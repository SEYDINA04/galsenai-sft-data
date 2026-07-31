"""Inventaire **amont** : combien de données existent par tâche, avant tout build.

Complément indispensable des statistiques de build (``validators.statistics``),
qui ne répondent qu'*après* coup. Ici on interroge l'API ``datasets-server`` de
HuggingFace pour connaître le nombre de lignes **disponibles** à la source, sans
rien télécharger — de quoi arbitrer un périmètre (« a-t-on assez de NER ? »)
avant de payer un build complet.

Trois populations sont comptées :

``integrated``
    datasets du plan de build (``configs/build.yaml``) — ce qui entre réellement.
``candidate``
    datasets ciblés, converter non encore écrit (``metadata/candidates.yaml``).
``excluded``
    datasets ciblés puis écartés, avec le motif — le volume qu'on renonce à
    utiliser fait partie de l'inventaire : c'est ce qui rend la décision lisible.

Le réseau est isolé derrière le protocole :class:`SizeProbe`, ce qui rend
l'inventaire testable hors ligne.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml
from pydantic import BaseModel, Field

from galsenai_sft.core.logging import get_logger

log = get_logger(__name__)

_SIZE_API = "https://datasets-server.huggingface.co/size"

#: Statuts possibles d'un dataset ciblé.
INTEGRATED = "integrated"
CANDIDATE = "candidate"
EXCLUDED = "excluded"


class SplitSize(BaseModel):
    """Taille d'un couple (config, split) tel que rapporté par HuggingFace."""

    config: str
    split: str
    n_rows: int


@runtime_checkable
class SizeProbe(Protocol):
    def sizes(self, dataset_id: str) -> list[SplitSize]:
        """Retourne la taille de chaque (config, split) du dataset."""
        ...


class HFSizeProbe:
    """Sonde réelle : endpoint ``/size`` du datasets-server HuggingFace.

    Ne télécharge aucune donnée — une requête HTTP par dataset. Les datasets
    privés/gated nécessitent ``HF_TOKEN`` (ou une session ``huggingface-cli
    login``).
    """

    def __init__(self, timeout: int = 60) -> None:
        self.timeout = timeout

    def sizes(self, dataset_id: str) -> list[SplitSize]:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(f"{_SIZE_API}?dataset={dataset_id}")
        if token := _hf_token():
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            data = json.load(resp)
        splits = (data.get("size") or {}).get("splits") or []
        return [
            SplitSize(
                config=str(s.get("config", "default")),
                split=str(s.get("split", "train")),
                # estimated_num_rows : renseigné pour les datasets trop gros
                # pour être comptés exactement ; on l'accepte en repli.
                n_rows=int(s.get("num_rows") or s.get("estimated_num_rows") or 0),
            )
            for s in splits
        ]


def _hf_token() -> str | None:
    import os

    if token := os.environ.get("HF_TOKEN"):
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # pragma: no cover - huggingface_hub absent
        return None


class SourceVolume(BaseModel):
    """Volume disponible pour un dataset source ciblé."""

    dataset_id: str
    task: str
    status: str = INTEGRATED
    config: str | None = None
    split: str = "train"
    #: Lignes du (config, split) effectivement ciblé — ce que le build lira.
    n_targeted: int | None = None
    #: Lignes de tous les splits de la config ciblée (train + validation + test).
    n_all_splits: int | None = None
    #: Motif d'exclusion / note de ciblage.
    reason: str = ""
    error: str | None = None

    @property
    def n_unused(self) -> int:
        """Lignes disponibles mais non lues par le build (autres splits)."""
        if self.n_all_splits is None or self.n_targeted is None:
            return 0
        return max(0, self.n_all_splits - self.n_targeted)


class TaskVolume(BaseModel):
    """Agrégat par tâche — la réponse directe à « combien de NER, de MT… »."""

    task: str
    n_sources: int = 0
    n_integrated_rows: int = 0
    n_candidate_rows: int = 0
    n_excluded_rows: int = 0

    @property
    def n_reachable(self) -> int:
        """Volume atteignable sans changer de politique (intégré + candidat)."""
        return self.n_integrated_rows + self.n_candidate_rows


class Inventory(BaseModel):
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    sources: list[SourceVolume] = Field(default_factory=list)
    by_task: dict[str, TaskVolume] = Field(default_factory=dict)

    @property
    def total_integrated(self) -> int:
        return sum(t.n_integrated_rows for t in self.by_task.values())

    @property
    def total_reachable(self) -> int:
        return sum(t.n_reachable for t in self.by_task.values())


# ════════════════════════════════════════════════════════════════════
#  Résolution config/split
# ════════════════════════════════════════════════════════════════════
def _resolve_config(sizes: list[SplitSize], config: str | None) -> str | None:
    """Détermine la config visée quand le plan n'en déclare aucune.

    ``load_dataset(id, split=...)`` sans config utilise la config par défaut :
    on reproduit cette règle (``default``, sinon l'unique config existante),
    plutôt que de sommer aveuglément 200 langues (cas sib200).
    """
    if config is not None:
        return config
    names = {s.config for s in sizes}
    if "default" in names:
        return "default"
    if len(names) == 1:
        return next(iter(names))
    return None  # ambigu : on somme tout, et on le signale


def measure(
    dataset_id: str,
    probe: SizeProbe,
    config: str | None = None,
    split: str = "train",
) -> tuple[int | None, int | None, str | None]:
    """Retourne ``(n_targeted, n_all_splits, error)`` pour un dataset.

    Toute erreur (dataset gated, hors ligne, API en panne) est **capturée** :
    un inventaire partiel reste utile, il ne doit jamais faire échouer l'appelant.
    """
    try:
        sizes = probe.sizes(dataset_id)
    except Exception as e:  # réseau, 404, gated…
        log.warning("%s : taille indisponible (%s)", dataset_id, e)
        return None, None, f"{type(e).__name__}: {e}"
    if not sizes:
        return None, None, "aucune taille rapportée par l'API"

    resolved = _resolve_config(sizes, config)
    scoped = [s for s in sizes if resolved is None or s.config == resolved]
    if not scoped:
        return None, None, f"config '{config}' introuvable"

    targeted = sum(s.n_rows for s in scoped if s.split == split)
    all_splits = sum(s.n_rows for s in scoped)
    err = None if resolved is not None else "config ambiguë : somme de toutes les configs"
    return targeted, all_splits, err


# ════════════════════════════════════════════════════════════════════
#  Construction de l'inventaire
# ════════════════════════════════════════════════════════════════════
def load_candidates(path: Path | None = None) -> list[dict[str, Any]]:
    """Charge ``metadata/candidates.yaml`` (datasets ciblés non intégrés)."""
    from galsenai_sft.core.config import REPO_ROOT

    path = path or REPO_ROOT / "metadata" / "candidates.yaml"
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(raw.get("datasets", []) or [])


def build_inventory(
    plan: Iterable[dict[str, Any]] | None = None,
    probe: SizeProbe | None = None,
    candidates: list[dict[str, Any]] | None = None,
) -> Inventory:
    """Interroge HuggingFace et agrège le volume disponible **par tâche**."""
    from galsenai_sft.build import load_build_plan
    from galsenai_sft.metadata import load_registry

    probe = probe or HFSizeProbe()
    plan = list(plan) if plan is not None else load_build_plan()
    candidates = load_candidates() if candidates is None else candidates
    registry = load_registry()

    sources: list[SourceVolume] = []

    for entry in plan:
        dataset_id = entry["id"]
        meta = registry.get(dataset_id)
        config = entry.get("config")
        split = entry.get("split", "train")
        n_t, n_all, err = measure(dataset_id, probe, config, split)
        sources.append(
            SourceVolume(
                dataset_id=dataset_id,
                task=meta.task if meta else "unknown",
                status=INTEGRATED,
                config=config,
                split=split,
                n_targeted=n_t,
                n_all_splits=n_all,
                error=err,
            )
        )

    for cand in candidates:
        dataset_id = cand["id"]
        config = cand.get("config")
        split = cand.get("split", "train")
        n_t, n_all, err = measure(dataset_id, probe, config, split)
        sources.append(
            SourceVolume(
                dataset_id=dataset_id,
                task=cand.get("task", "unknown"),
                status=cand.get("status", CANDIDATE),
                config=config,
                split=split,
                n_targeted=n_t,
                n_all_splits=n_all,
                reason=cand.get("reason", ""),
                error=err,
            )
        )

    by_task: dict[str, TaskVolume] = {}
    for s in sources:
        tv = by_task.setdefault(s.task, TaskVolume(task=s.task))
        tv.n_sources += 1
        rows = s.n_targeted or 0
        if s.status == INTEGRATED:
            tv.n_integrated_rows += rows
        elif s.status == EXCLUDED:
            tv.n_excluded_rows += rows
        else:
            tv.n_candidate_rows += rows

    return Inventory(sources=sources, by_task=dict(sorted(by_task.items())))


# ════════════════════════════════════════════════════════════════════
#  Rendu
# ════════════════════════════════════════════════════════════════════
_STATUS_LABEL = {
    INTEGRATED: "✅ intégré",
    CANDIDATE: "🕐 candidat",
    EXCLUDED: "⛔ écarté",
}


def render_inventory(inv: Inventory) -> str:
    """Génère ``docs/inventory.md`` : volume disponible par tâche puis par source."""
    lines = [
        "# 📊 Inventaire — volume disponible par tâche",
        "",
        "> Généré automatiquement (`galsenai-sft inventory`). Ne pas éditer à la main.",
        "> Mesuré **à la source** via l'API `datasets-server` de HuggingFace,",
        "> sans rien télécharger. Ce sont des lignes **brutes** : un converter peut",
        "> en produire plusieurs exemples SFT (traduction bidirectionnelle) ou moins",
        "> (déduplication, filtre LID).",
        "",
        f"- **Généré le** : {inv.created_at}",
        f"- **Lignes intégrées au build** : {inv.total_integrated:,}",
        f"- **Lignes atteignables** (intégré + candidats) : {inv.total_reachable:,}",
        "",
        "## Par tâche",
        "",
        "| tâche | sources | intégré | candidat | écarté | atteignable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, tv in inv.by_task.items():
        lines.append(
            f"| {task} | {tv.n_sources} | {tv.n_integrated_rows:,} "
            f"| {tv.n_candidate_rows:,} | {tv.n_excluded_rows:,} | {tv.n_reachable:,} |"
        )

    lines += [
        "",
        "## Par source",
        "",
        "| dataset | tâche | statut | config/split | ciblé | tous splits | note |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for s in sorted(inv.sources, key=lambda x: (x.task, x.status, x.dataset_id)):
        scope = f"{s.config or '—'}/{s.split}"
        n_t = f"{s.n_targeted:,}" if s.n_targeted is not None else "—"
        n_a = f"{s.n_all_splits:,}" if s.n_all_splits is not None else "—"
        note = s.reason or (f"⚠️ {s.error}" if s.error else "")
        lines.append(
            f"| `{s.dataset_id}` | {s.task} | {_STATUS_LABEL.get(s.status, s.status)} "
            f"| {scope} | {n_t} | {n_a} | {note} |"
        )

    unused = sum(s.n_unused for s in inv.sources if s.status == INTEGRATED)
    if unused:
        lines += [
            "",
            f"> **{unused:,} lignes disponibles mais non lues** par le build : ce sont",
            "> les splits `validation`/`test` des sources intégrées. Elles constituent",
            "> la réserve naturelle pour un futur jeu d'évaluation.",
        ]
    lines.append("")
    return "\n".join(lines)


def write_inventory(
    inv: Inventory, json_path: Path | None = None, md_path: Path | None = None
) -> tuple[Path, Path]:
    """Écrit ``metadata/inventory.json`` (machine) et ``docs/inventory.md`` (humain)."""
    from galsenai_sft.core.config import REPO_ROOT

    json_path = json_path or REPO_ROOT / "metadata" / "inventory.json"
    md_path = md_path or REPO_ROOT / "docs" / "inventory.md"
    for p in (json_path, md_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(inv.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_inventory(inv), encoding="utf-8")
    return json_path, md_path


def load_inventory(path: Path | None = None) -> Inventory | None:
    """Relit ``metadata/inventory.json`` s'il existe (sans refaire de réseau)."""
    from galsenai_sft.core.config import REPO_ROOT

    path = path or REPO_ROOT / "metadata" / "inventory.json"
    if not path.exists():
        return None
    return Inventory.model_validate(json.loads(path.read_text(encoding="utf-8")))
