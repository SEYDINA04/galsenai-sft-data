"""Builder end-to-end : datasets sources -> dataset SFT ChatML.

**Architecture mémoire (critique).** Le build traite des centaines de milliers
d'exemples : rien n'est accumulé en RAM. Chaque Sample produit est écrit
immédiatement dans tous les fichiers de sortie, puis oublié :

    ligne brute -> converter -> qualité -> LID -> décontamination -> écriture
                                                                     (+ stats
                                                                      incrémentales)

Conséquence : la mémoire du build est **constante** (quelques centaines de Mo),
quel que soit le volume. Un :class:`~galsenai_sft.core.memory.MemoryGuard`
surveille en plus la mémoire système et déclenche un **arrêt propre** (fichiers
fermés, manifest écrit, build marqué ``partial``) plutôt que de faire tomber la
machine.

Le ``Loader`` est injectable -> le builder est testable sans réseau.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from galsenai_sft.core.config import REPO_ROOT, Settings, get_settings
from galsenai_sft.core.logging import get_logger
from galsenai_sft.core.memory import MemoryGuard, MemoryPressure, describe_environment
from galsenai_sft.core.schema import Sample
from galsenai_sft.exporters import get_exporter
from galsenai_sft.loaders import HFLoader, Loader
from galsenai_sft.registry import get_converter
from galsenai_sft.validators.quality_validator import filter_quality
from galsenai_sft.validators.statistics import StatisticsAccumulator

log = get_logger(__name__)


def _rel(path: Path) -> str:
    """Chemin relatif au repo si possible, sinon chemin absolu (robuste aux tests)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class BuildEntryReport(BaseModel):
    dataset_id: str
    task: str
    split: str
    config: str | None = None
    n_raw: int = 0
    n_samples: int = 0
    n_filtered_lid: int = 0
    n_decontaminated: int = 0
    error: str | None = None


class BuildManifest(BaseModel):
    version: str
    created_at: str
    total_samples: int = 0
    entries: list[BuildEntryReport] = Field(default_factory=list)
    by_task: dict[str, int] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)  # fichier -> checksum
    # Traçabilité mémoire : un build interrompu reste exploitable, mais signalé.
    partial: bool = False
    stop_reason: str | None = None
    peak_rss_mb: float = 0.0


def load_build_plan(path: Path | None = None) -> list[dict[str, Any]]:
    """Charge le plan de build (liste de {id, split, config})."""
    path = path or REPO_ROOT / "configs" / "build.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("datasets", []) or []


# ════════════════════════════════════════════════════════════════════════
#  Écriture au fil de l'eau
# ════════════════════════════════════════════════════════════════════════
class _JsonlWriter:
    """Fichier JSONL ouvert en écriture, avec checksum calculé au vol.

    Hasher pendant l'écriture évite une seconde passe de lecture sur des
    fichiers de plusieurs Go à la fin du build.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.n = 0
        self._fh = path.open("w", encoding="utf-8")
        self._digest = hashlib.sha256()

    def write_line(self, text: str) -> None:
        data = text + "\n"
        self._fh.write(data)
        self._digest.update(data.encode("utf-8"))
        self.n += 1

    def close(self) -> str:
        if not self._fh.closed:
            self._fh.close()
        return self._digest.hexdigest()


class SampleSink:
    """Destination unique d'un Sample : le distribue dans tous les formats.

    Fichiers écrits simultanément (≈ 11 descripteurs, mémoire négligeable) :
      - ``chatml/all.jsonl``   : Samples canoniques bruts (rechargeables) ;
      - ``chatml/<tâche>.jsonl`` : ChatML exporté, un fichier par tâche ;
      - ``alpaca/all.jsonl``, ``sharegpt/all.jsonl`` : formats dérivés.
    """

    def __init__(self, settings: Settings, export_formats: tuple[str, ...]) -> None:
        self._chatml_dir = settings.paths.processed_chatml
        self._to_chatml = get_exporter("chatml")
        self._all = _JsonlWriter(self._chatml_dir / "all.jsonl")
        self._task_writers: dict[str, _JsonlWriter] = {}
        self._exports: dict[str, tuple[_JsonlWriter, Any]] = {}
        for fmt in export_formats:
            if fmt == "chatml":
                continue
            out_dir: Path = getattr(settings.paths, f"processed_{fmt}")
            self._exports[fmt] = (_JsonlWriter(out_dir / "all.jsonl"), get_exporter(fmt))

    def write(self, sample: Sample) -> None:
        """Écrit un Sample dans toutes les sorties, puis l'oublie."""
        self._all.write_line(sample.model_dump_json(exclude_none=True))

        task = sample.task.value
        writer = self._task_writers.get(task)
        if writer is None:
            writer = self._task_writers[task] = _JsonlWriter(self._chatml_dir / f"{task}.jsonl")
        writer.write_line(json.dumps(self._to_chatml(sample), ensure_ascii=False))

        for exporter_writer, exporter in self._exports.values():
            try:
                row = exporter(sample)
            except ValueError:
                continue  # ex. Alpaca ignore les multi-tours
            exporter_writer.write_line(json.dumps(row, ensure_ascii=False))

    def close(self) -> dict[str, str]:
        """Ferme tout et retourne {chemin relatif -> checksum}."""
        outputs: dict[str, str] = {}
        for w in (*self._task_writers.values(), self._all):
            outputs[_rel(w.path)] = w.close()
        for w, _ in self._exports.values():
            outputs[_rel(w.path)] = w.close()
        return outputs


# ════════════════════════════════════════════════════════════════════════
#  Pipeline d'un dataset (générateur, mémoire constante)
# ════════════════════════════════════════════════════════════════════════
def iter_entry_samples(
    entry: dict[str, Any],
    loader: Loader,
    report: BuildEntryReport,
    decontam_index: set[str] | None = None,
    limit: int | None = None,
    target_filter: Any | None = None,
) -> Iterator[Sample]:
    """Flux de Samples d'un dataset : charge -> convertit -> filtre.

    Générateur : aucune liste intermédiaire. ``report`` est mis à jour au fil de
    l'eau (compteurs, erreur éventuelle) — une panne réseau en milieu de dataset
    conserve donc les exemples déjà produits.

    Deux plafonds distincts :
      - ``limit`` (CLI ``--limit``) : lignes **source** lues, pour un smoke test ;
      - ``max_samples`` (clé du plan de build) : exemples **produits**, pour
        rééquilibrer un dataset surreprésenté.
    """
    dataset_id = entry["id"]
    max_samples = entry.get("max_samples")

    try:
        conv = get_converter(dataset_id)()
        rows = loader.load(
            dataset_id,
            split=entry.get("split", "train"),
            config=entry.get("config"),
            columns=entry.get("columns"),
        )

        def _counted() -> Iterator[dict[str, Any]]:
            for i, row in enumerate(rows):
                if limit and i >= limit:
                    break
                report.n_raw += 1
                yield row

        stream: Iterator[Sample] = filter_quality(conv.convert(_counted()))

        # Filtre LID de la cible wolof (opt-in par dataset : datasets bruités)
        if entry.get("lid_filter") and target_filter is not None:
            stream = _lid_filtered(stream, target_filter, report)

        if decontam_index:
            stream = _counted_decontamination(stream, decontam_index, report)

        for sample in stream:
            report.n_samples += 1
            yield sample
            if max_samples and report.n_samples >= max_samples:
                log.info("%s : plafond de %d exemples atteint", dataset_id, max_samples)
                break

    except Exception as e:  # noqa: BLE001 — un dataset en échec ne casse pas le build
        report.error = f"{type(e).__name__}: {e}"
        log.error("échec conversion %s : %s", dataset_id, report.error)


def _lid_filtered(
    stream: Iterator[Sample], target_filter: Any, report: BuildEntryReport
) -> Iterator[Sample]:
    for s in stream:
        if target_filter.keep(s):
            yield s
        else:
            report.n_filtered_lid += 1


def _counted_decontamination(
    stream: Iterator[Sample], index: set[str], report: BuildEntryReport
) -> Iterator[Sample]:
    from galsenai_sft.validators.decontamination import decontaminate

    kept = 0
    seen = 0

    def _counting(src: Iterator[Sample]) -> Iterator[Sample]:
        nonlocal seen
        for s in src:
            seen += 1
            yield s

    for s in decontaminate(_counting(stream), index):
        kept += 1
        yield s
    report.n_decontaminated = seen - kept


def _new_report(entry: dict[str, Any]) -> BuildEntryReport:
    """Rapport vierge d'une entrée du plan (tolère un dataset sans converter)."""
    try:
        task = get_converter(entry["id"]).task.value
    except KeyError:
        task = "unknown"
    return BuildEntryReport(
        dataset_id=entry["id"],
        task=task,
        split=entry.get("split", "train"),
        config=entry.get("config"),
    )


def convert_entry(
    entry: dict[str, Any],
    loader: Loader,
    decontam_index: set[str] | None = None,
    limit: int | None = None,
    target_filter: Any | None = None,
) -> tuple[list[Sample], BuildEntryReport]:
    """Version matérialisée de :func:`iter_entry_samples` (petits lots, tests).

    ⚠️ Charge tout en mémoire : à réserver au debug et aux datasets courts. Le
    build utilise le générateur.
    """
    report = _new_report(entry)
    samples = list(iter_entry_samples(entry, loader, report, decontam_index, limit, target_filter))
    return samples, report


# ════════════════════════════════════════════════════════════════════════
#  Build
# ════════════════════════════════════════════════════════════════════════
def build(
    plan: list[dict[str, Any]] | None = None,
    loader: Loader | None = None,
    settings: Settings | None = None,
    version: str = "0.1.0",
    limit: int | None = None,
    decontaminate_corpus: bool = False,
    export_formats: tuple[str, ...] = ("chatml", "alpaca", "sharegpt"),
    guard: MemoryGuard | None = None,
) -> BuildManifest:
    """Exécute le build complet et écrit les artefacts. Retourne le manifest.

    Mémoire constante : les Samples sont écrits puis libérés. En cas de pression
    mémoire (ou de ``Ctrl-C``), les fichiers sont fermés proprement et le
    manifest est écrit avec ``partial=True`` — les données déjà produites
    restent exploitables.
    """
    plan = plan if plan is not None else load_build_plan()
    settings = settings or get_settings()
    loader = loader or HFLoader(streaming=settings.build.streaming)
    guard = guard if guard is not None else MemoryGuard.from_settings(settings.memory)

    decontam_index: set[str] | None = None
    if decontaminate_corpus:
        from galsenai_sft.validators.decontamination import build_pretraining_index

        decontam_index = build_pretraining_index()

    # Filtre LID de la cible wolof, instancié une fois si au moins un dataset
    # du plan l'exige (le modèle GlotLID pèse ~1,6 Go en RAM : jamais deux fois).
    target_filter = None
    if any(e.get("lid_filter") for e in plan):
        from galsenai_sft.validators.content_filter import WolofTargetFilter

        target_filter = WolofTargetFilter(threshold=settings.lid.threshold)

    manifest = BuildManifest(version=version, created_at=datetime.now(UTC).isoformat())
    stats = StatisticsAccumulator()
    sink = SampleSink(settings, export_formats)
    log.info("build v%s — %s", version, describe_environment())

    # Index de la dernière entrée nécessitant le LID : au-delà, le modèle
    # (~1,6 Go) est libéré — il pèse plus que tout le reste du pipeline.
    last_lid_index = max((i for i, e in enumerate(plan) if e.get("lid_filter")), default=-1)

    try:
        with guard:
            for index, entry in enumerate(plan):
                report = _new_report(entry)
                manifest.entries.append(report)

                for sample in iter_entry_samples(
                    entry, loader, report, decontam_index, limit, target_filter
                ):
                    guard.check()  # arrêt propre si la mémoire système s'effondre
                    sink.write(sample)
                    stats.update(sample)
                    if stats.total % settings.build.log_every == 0:
                        _, rss = guard.sample()  # RSS courant (le pic est dans le manifest)
                        log.info(
                            "%s exemples écrits (RSS %.0f Mo, pic %.0f Mo)",
                            f"{stats.total:,}",
                            rss,
                            guard.peak_rss_mb,
                        )

                log.info("%s : %d samples", report.dataset_id, report.n_samples)

                if index == last_lid_index and target_filter is not None:
                    target_filter.release()
                    target_filter = None
    except (MemoryPressure, KeyboardInterrupt) as e:
        manifest.partial = True
        manifest.stop_reason = str(e) or type(e).__name__
        log.warning("build interrompu (%s) — artefacts partiels conservés", type(e).__name__)
    finally:
        manifest.outputs = sink.close()  # ferme et checksums, même en cas d'arrêt
        guard.stop()

    manifest.total_samples = stats.total
    manifest.by_task = dict(sorted(stats.by_task.items()))
    manifest.peak_rss_mb = round(guard.peak_rss_mb, 1)

    # --- Statistiques + manifest ---
    stats_path = settings.paths.interim / "build_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(stats.result().model_dump_json(indent=2), encoding="utf-8")

    manifest_path = settings.paths.interim / "build_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    log.info(
        "build terminé : %d samples · pic RSS %.0f Mo -> manifest %s",
        manifest.total_samples,
        manifest.peak_rss_mb,
        manifest_path,
    )

    return manifest
