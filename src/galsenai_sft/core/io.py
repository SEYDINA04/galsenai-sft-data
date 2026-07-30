"""Entrées/sorties : lecture/écriture de :class:`Sample` en JSONL, checksums."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from galsenai_sft.core.schema import Sample


def write_samples_jsonl(samples: Iterable[Sample], path: str | Path) -> int:
    """Écrit des Samples en JSONL (un objet pydantic par ligne). Retourne le compte."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json(exclude_none=True) + "\n")
            n += 1
    return n


def read_samples_jsonl(path: str | Path) -> Iterator[Sample]:
    """Lit un JSONL de Samples (streaming, une ligne à la fois)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Sample.model_validate_json(line)


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    """Écrit des dicts arbitraires en JSONL (pour les formats exportés)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256_file(path: str | Path) -> str:
    """Checksum SHA-256 d'un fichier (pour la traçabilité des artefacts)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
