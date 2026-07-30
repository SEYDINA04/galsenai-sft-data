"""Builder end-to-end : datasets sources -> dataset SFT ChatML.

Pour chaque dataset du plan de build :
  charge (Loader) -> convertit (converter) -> filtre qualité -> décontamine
  (optionnel) -> collecte les Samples.

Puis : écrit un JSONL ChatML par tâche + un JSONL global, exporte
alpaca/sharegpt, calcule les statistiques et un **manifest** reproductible
(datasets, versions, checksums, compteurs).

Le ``Loader`` est injectable -> le builder est testable sans réseau.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from galsenai_sft.core.config import REPO_ROOT, Settings, get_settings
from galsenai_sft.core.io import sha256_file, write_jsonl, write_samples_jsonl
from galsenai_sft.core.logging import get_logger
from galsenai_sft.core.schema import Sample
from galsenai_sft.exporters import get_exporter
from galsenai_sft.loaders import HFLoader, Loader
from galsenai_sft.registry import get_converter
from galsenai_sft.validators.quality_validator import filter_quality
from galsenai_sft.validators.statistics import compute_statistics

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
    error: str | None = None


class BuildManifest(BaseModel):
    version: str
    created_at: str
    total_samples: int = 0
    entries: list[BuildEntryReport] = Field(default_factory=list)
    by_task: dict[str, int] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)  # fichier -> checksum


def load_build_plan(path: Path | None = None) -> list[dict[str, Any]]:
    """Charge le plan de build (liste de {id, split, config})."""
    path = path or REPO_ROOT / "configs" / "build.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("datasets", []) or []


def convert_entry(
    entry: dict[str, Any],
    loader: Loader,
    decontam_index: set[str] | None = None,
    limit: int | None = None,
) -> tuple[list[Sample], BuildEntryReport]:
    """Charge + convertit + filtre un dataset. Retourne (samples, rapport)."""
    dataset_id = entry["id"]
    split = entry.get("split", "train")
    config = entry.get("config")
    cls = get_converter(dataset_id)
    conv = cls()
    report = BuildEntryReport(
        dataset_id=dataset_id, task=cls.task.value, split=split, config=config
    )

    try:
        rows = loader.load(dataset_id, split=split, config=config)
        n_raw = 0

        def _counted():
            nonlocal n_raw
            for i, row in enumerate(rows):
                if limit and i >= limit:
                    break
                n_raw += 1
                yield row

        samples = list(filter_quality(conv.convert(_counted())))

        if decontam_index:
            from galsenai_sft.validators.decontamination import decontaminate

            samples = list(decontaminate(samples, decontam_index))

        report.n_raw = n_raw
        report.n_samples = len(samples)
        return samples, report
    except Exception as e:  # noqa: BLE001 — on veut un build robuste par dataset
        report.error = f"{type(e).__name__}: {e}"
        log.error("échec conversion %s : %s", dataset_id, report.error)
        return [], report


def build(
    plan: list[dict[str, Any]] | None = None,
    loader: Loader | None = None,
    settings: Settings | None = None,
    version: str = "0.1.0",
    limit: int | None = None,
    decontaminate_corpus: bool = False,
    export_formats: tuple[str, ...] = ("chatml", "alpaca", "sharegpt"),
) -> BuildManifest:
    """Exécute le build complet et écrit les artefacts. Retourne le manifest."""
    plan = plan if plan is not None else load_build_plan()
    loader = loader or HFLoader()
    settings = settings or get_settings()

    decontam_index: set[str] | None = None
    if decontaminate_corpus:
        from galsenai_sft.validators.decontamination import build_pretraining_index

        decontam_index = build_pretraining_index()

    all_samples: list[Sample] = []
    by_task_samples: dict[str, list[Sample]] = defaultdict(list)
    manifest = BuildManifest(version=version, created_at=datetime.now(UTC).isoformat())

    for entry in plan:
        samples, rep = convert_entry(entry, loader, decontam_index, limit=limit)
        manifest.entries.append(rep)
        for s in samples:
            all_samples.append(s)
            by_task_samples[s.task.value].append(s)
        log.info("%s : %d samples", rep.dataset_id, rep.n_samples)

    manifest.total_samples = len(all_samples)
    manifest.by_task = {t: len(v) for t, v in sorted(by_task_samples.items())}

    # --- Écriture ChatML : un fichier par tâche + un global ---
    chatml_dir = settings.paths.processed_chatml
    to_chatml = get_exporter("chatml")

    for task, samples in by_task_samples.items():
        p = chatml_dir / f"{task}.jsonl"
        write_jsonl((to_chatml(s) for s in samples), p)
        manifest.outputs[_rel(p)] = sha256_file(p)

    all_path = chatml_dir / "all.jsonl"
    write_samples_jsonl(all_samples, all_path)  # Samples bruts (rechargeable)
    manifest.outputs[_rel(all_path)] = sha256_file(all_path)

    # --- Autres formats (global) ---
    for fmt in export_formats:
        if fmt == "chatml":
            continue
        exporter = get_exporter(fmt)
        out_dir = getattr(settings.paths, f"processed_{fmt}")
        p = out_dir / "all.jsonl"

        def _rows(exp=exporter):
            for s in all_samples:
                try:
                    yield exp(s)
                except ValueError:
                    continue  # ex. Alpaca ignore les multi-tours

        write_jsonl(_rows(), p)
        manifest.outputs[_rel(p)] = sha256_file(p)

    # --- Statistiques + manifest ---
    stats = compute_statistics(all_samples)
    stats_path = settings.paths.interim / "build_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(stats.model_dump_json(indent=2), encoding="utf-8")

    manifest_path = settings.paths.interim / "build_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    log.info("build terminé : %d samples -> manifest %s", manifest.total_samples, manifest_path)

    return manifest
