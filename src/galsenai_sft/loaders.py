"""Chargement des lignes brutes d'un dataset (backend injectable).

Abstraction ``Loader`` pour découpler le builder de HuggingFace ``datasets`` :
  - ``HFLoader`` : implémentation réelle (télécharge via ``datasets.load_dataset``) ;
  - un loader factice peut être injecté dans les tests (aucune dépendance réseau).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Loader(Protocol):
    def load(
        self, dataset_id: str, split: str = "train", config: str | None = None
    ) -> Iterable[dict[str, Any]]:
        """Retourne un itérable de lignes brutes (dicts)."""
        ...


class HFLoader:
    """Charge un dataset depuis le Hub HuggingFace (streaming possible)."""

    def __init__(self, streaming: bool = False) -> None:
        self.streaming = streaming

    def load(
        self, dataset_id: str, split: str = "train", config: str | None = None
    ) -> Iterable[dict[str, Any]]:
        from datasets import load_dataset

        ds = (
            load_dataset(dataset_id, config, split=split, streaming=self.streaming)
            if config
            else load_dataset(dataset_id, split=split, streaming=self.streaming)
        )
        return ds
