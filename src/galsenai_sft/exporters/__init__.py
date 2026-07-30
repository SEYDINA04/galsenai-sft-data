"""Exporters : sérialisation d'un :class:`Sample` vers les formats de sortie.

Un seul registre pour choisir un format par nom (utilisé par la CLI).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from galsenai_sft.core.schema import Sample
from galsenai_sft.exporters.alpaca import to_alpaca
from galsenai_sft.exporters.chatml import to_chatml
from galsenai_sft.exporters.sharegpt import to_sharegpt

EXPORTERS: dict[str, Callable[[Sample], dict[str, Any]]] = {
    "chatml": to_chatml,
    "alpaca": to_alpaca,
    "sharegpt": to_sharegpt,
}

__all__ = ["EXPORTERS", "to_alpaca", "to_chatml", "to_sharegpt"]


def get_exporter(fmt: str) -> Callable[[Sample], dict[str, Any]]:
    """Retourne la fonction d'export pour un format nommé."""
    if fmt not in EXPORTERS:
        raise KeyError(f"format inconnu '{fmt}'. Disponibles : {sorted(EXPORTERS)}")
    return EXPORTERS[fmt]
