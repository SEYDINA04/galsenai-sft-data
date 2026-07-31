"""Registre des métadonnées + génération du catalogue Markdown.

Fusionne :
  - la tâche déclarée par chaque converter enregistré (source de vérité code) ;
  - les infos statiques de ``metadata/datasets_registry.yaml`` (licence, url…).

Produit ``docs/dataset_catalog.md`` (tableau lisible) et permet de mettre à jour
les compteurs/checksum après conversion.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from galsenai_sft.core.config import get_settings
from galsenai_sft.metadata.models import DatasetMeta, LicenseStatus
from galsenai_sft.registry import available, get_converter


def load_registry(path: Path | None = None) -> dict[str, DatasetMeta]:
    """Charge le registre : 1 DatasetMeta par converter enregistré.

    Les valeurs du YAML surchargent les valeurs par défaut ; la tâche est
    toujours prise du converter (le code fait foi). Si un inventaire existe
    (``metadata/inventory.json``, produit par ``galsenai-sft inventory``), les
    volumes disponibles à la source y sont repris — sinon ``n_samples`` reste
    ``None`` et le catalogue affiche ``—``.
    """
    path = path or get_settings().paths.datasets_registry
    raw: dict = {}
    if path.exists():
        raw = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("datasets", {}) or {}

    volumes = _source_volumes()
    out: dict[str, DatasetMeta] = {}
    for dataset_id in available():
        cls = get_converter(dataset_id)
        entry = dict(raw.get(dataset_id, {}))
        entry["dataset_id"] = dataset_id
        entry["task"] = cls.task.value  # le converter fait foi
        if (n := volumes.get(dataset_id)) is not None:
            entry.setdefault("n_samples", n)
        out[dataset_id] = DatasetMeta.model_validate(entry)
    return out


def _source_volumes() -> dict[str, int]:
    """Volumes mesurés à la source, depuis l'inventaire s'il a été généré.

    Import local et tolérant : le catalogue doit rester générable sans réseau
    et sans inventaire préalable.
    """
    try:
        from galsenai_sft.inventory import load_inventory

        inv = load_inventory()
    except Exception:  # pragma: no cover - inventaire illisible
        return {}
    if inv is None:
        return {}
    return {s.dataset_id: s.n_targeted for s in inv.sources if s.n_targeted is not None}


def render_catalog(registry: dict[str, DatasetMeta]) -> str:
    """Génère le catalogue Markdown des datasets."""
    lines = [
        "# 📖 Catalogue des datasets",
        "",
        "> Généré automatiquement (`galsenai-sft catalog`). Ne pas éditer à la main.",
        "",
        f"**{len(registry)} datasets** intégrés.",
        "",
        "La colonne *lignes source* provient de `galsenai-sft inventory` (mesure à",
        "la source, sans téléchargement). Ce n'est **pas** le nombre d'exemples SFT",
        "produits : un converter peut en générer plusieurs par ligne (traduction",
        "bidirectionnelle) ou moins (déduplication, filtre LID). Voir",
        "[`inventory.md`](inventory.md) pour le volume par tâche et les sources",
        "ciblées mais non intégrées.",
        "",
        "| dataset | tâche | licence | commercial | statut | lignes source |",
        "|---|---|---|:---:|---|---:|",
    ]
    for ds, m in sorted(registry.items(), key=lambda kv: (kv[1].task, kv[0])):
        commercial = "✅" if m.commercial_ok else "⚠️"
        n = f"{m.n_samples:,}" if m.n_samples is not None else "—"
        url = f"[`{ds}`]({m.source_url})" if m.source_url else f"`{ds}`"
        lines.append(
            f"| {url} | {m.task} | {m.license} ({m.license_status.value}) "
            f"| {commercial} | {m.conversion_status.value} | {n} |"
        )

    # Récap licences (track recherche vs commercial)
    commercial_ok = [d for d, m in registry.items() if m.commercial_ok]
    nc = [d for d, m in registry.items() if m.license_status is LicenseStatus.NON_COMMERCIAL]
    unverified = [d for d, m in registry.items() if m.license_status is LicenseStatus.UNVERIFIED]
    lines += [
        "",
        "## Conformité licences",
        "",
        f"- **Utilisables en commercial** : {len(commercial_ok)}",
        f"- **Non-commercial (track recherche)** : {len(nc)} — {', '.join(f'`{d}`' for d in nc) or 'aucun'}",
        f"- **Licence non vérifiée** : {len(unverified)} — à auditer avant usage commercial",
        "",
    ]
    return "\n".join(lines)


def write_catalog(
    out_path: Path | None = None, registry: dict[str, DatasetMeta] | None = None
) -> Path:
    """Écrit le catalogue Markdown (par défaut ``docs/dataset_catalog.md``)."""
    from galsenai_sft.core.config import REPO_ROOT

    registry = registry or load_registry()
    out = out_path or REPO_ROOT / "docs" / "dataset_catalog.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_catalog(registry), encoding="utf-8")
    return out
