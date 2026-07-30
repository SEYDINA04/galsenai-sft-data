"""Chargement des lignes brutes d'un dataset (backend injectable).

Abstraction ``Loader`` pour découpler le builder de HuggingFace ``datasets`` :
  - ``HFLoader`` : implémentation réelle (télécharge via ``datasets.load_dataset``) ;
  - un loader factice peut être injecté dans les tests (aucune dépendance réseau).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from galsenai_sft.core.logging import get_logger

log = get_logger(__name__)

# Endpoint d'auto-conversion parquet de HuggingFace (datasets-server).
_PARQUET_API = "https://datasets-server.huggingface.co/parquet"


@runtime_checkable
class Loader(Protocol):
    def load(
        self, dataset_id: str, split: str = "train", config: str | None = None
    ) -> Iterable[dict[str, Any]]:
        """Retourne un itérable de lignes brutes (dicts)."""
        ...


def _parquet_urls(dataset_id: str, split: str, config: str | None) -> list[str]:
    """Récupère les URLs parquet auto-converties pour (dataset, config, split)."""
    import os
    import urllib.request

    token = os.environ.get("HF_TOKEN")
    req = urllib.request.Request(f"{_PARQUET_API}?dataset={dataset_id}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        import json

        data = json.load(resp)
    files = data.get("parquet_files", [])
    return [
        f["url"]
        for f in files
        if f.get("split") == split and (config is None or f.get("config") == config)
    ]


class HFLoader:
    """Charge un dataset depuis le Hub HuggingFace (streaming possible).

    Robustesse : si le dataset repose sur un **script de chargement** (désormais
    refusé par ``datasets``), on bascule automatiquement sur la branche
    **parquet auto-convertie** par HuggingFace — solution générale, sans logique
    spécifique à un dataset.
    """

    def __init__(self, streaming: bool = False) -> None:
        self.streaming = streaming

    def load(
        self, dataset_id: str, split: str = "train", config: str | None = None
    ) -> Iterable[dict[str, Any]]:
        from datasets import load_dataset

        try:
            ds = (
                load_dataset(dataset_id, config, split=split, streaming=self.streaming)
                if config
                else load_dataset(dataset_id, split=split, streaming=self.streaming)
            )
            return ds
        except (RuntimeError, ValueError) as e:
            msg = str(e).lower()
            if "script" not in msg and "no longer supported" not in msg:
                raise
            log.warning(
                "%s utilise un script de chargement — bascule sur la branche parquet",
                dataset_id,
            )
            urls = _parquet_urls(dataset_id, split, config)
            if not urls:
                raise RuntimeError(
                    f"aucun parquet auto-converti trouvé pour {dataset_id} "
                    f"(config={config}, split={split})"
                ) from e
            return load_dataset("parquet", data_files=urls, split="train", streaming=self.streaming)
