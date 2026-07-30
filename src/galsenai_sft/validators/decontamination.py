"""Décontamination : retire les Samples dont le contenu apparaît déjà dans le
corpus de **pré-entraînement**, pour éviter la fuite d'évaluation (le modèle a
déjà vu ces textes en pré-entraînement — les garder en SFT/éval fausse la mesure).

Stratégie : empreintes de hash exactes (normalisées) des textes du corpus de
pré-entraînement, chargées en streaming (parquet/jsonl, colonne ``text``), puis
filtrage des Samples dont un tour utilisateur ou assistant matche.

Le corpus de référence est déclaré dans la config
(``pretraining_corpus_paths``). Optionnel : si aucun chemin, no-op.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path

from galsenai_sft.core.config import get_settings
from galsenai_sft.core.logging import get_logger
from galsenai_sft.core.schema import Sample

log = get_logger(__name__)


def _norm_hash(text: str) -> str:
    return hashlib.sha1(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def _iter_texts(path: Path) -> Iterator[str]:
    """Itère la colonne 'text' d'un parquet ou d'un jsonl (streaming)."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=50_000, columns=["text"]):
            yield from (t or "" for t in batch.column("text").to_pylist())
    elif path.suffix in {".jsonl", ".json"}:
        import json

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    yield obj.get("text", "") if isinstance(obj, dict) else ""
    else:
        raise ValueError(f"format non supporté pour la décontamination : {path}")


def build_pretraining_index(paths: Iterable[Path] | None = None) -> set[str]:
    """Construit l'ensemble des empreintes des textes de pré-entraînement."""
    from galsenai_sft.core.config import REPO_ROOT

    if paths is None:
        paths = get_settings().pretraining_corpus_paths
    index: set[str] = set()
    for p in paths:
        p = Path(p)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()  # chemins relatifs = relatifs au repo
        if not p.exists():
            log.warning("corpus de pré-entraînement introuvable : %s", p)
            continue
        n = 0
        for text in _iter_texts(p):
            if text.strip():
                index.add(_norm_hash(text))
                n += 1
        log.info("décontamination : %d textes indexés depuis %s", n, p.name)
    return index


def decontaminate(samples: Iterable[Sample], index: set[str] | None = None) -> Iterator[Sample]:
    """Retire les Samples dont un tour matche le corpus de pré-entraînement.

    Si ``index`` est vide/None, aucun filtrage n'est appliqué (no-op).
    """
    if index is None:
        index = build_pretraining_index()
    if not index:
        yield from samples
        return

    removed = 0
    for s in samples:
        hit = any(_norm_hash(m.content) in index for m in s.messages if m.content.strip())
        if hit:
            removed += 1
            continue
        yield s
    if removed:
        log.info("décontamination : %d samples retirés (fuite de pré-entraînement)", removed)
