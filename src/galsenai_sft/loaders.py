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
        self,
        dataset_id: str,
        split: str = "train",
        config: str | None = None,
        columns: list[str] | None = None,
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


def _iter_parquet_columns(
    urls: list[str], columns: list[str], batch_size: int = 2000
) -> Iterable[dict[str, Any]]:
    """Lit UNIQUEMENT ``columns`` depuis des parquets distants (lecture colonnaire).

    Évite de télécharger les colonnes lourdes (audio/image) : sur un parquet, la
    projection de colonnes ne récupère que les blocs des colonnes demandées.
    """
    import fsspec
    import pyarrow.parquet as pq

    fs = fsspec.filesystem("https")
    for url in urls:
        pf = pq.ParquetFile(fs.open(url))
        present = [c for c in columns if c in pf.schema_arrow.names]
        if not present:
            continue
        for batch in pf.iter_batches(batch_size=batch_size, columns=present):
            yield from batch.to_pylist()


def _project(ds: Any, columns: list[str] | None) -> Any:
    """Restreint un dataset HF aux colonnes demandées, si l'API le permet.

    Filet de sécurité quand la branche parquet n'est pas disponible : évite de
    matérialiser les colonnes lourdes (audio/image) ligne par ligne.
    """
    if not columns:
        return ds
    select = getattr(ds, "select_columns", None)
    names = getattr(ds, "column_names", None)
    if select is None or not names:
        return ds
    present = [c for c in columns if c in names]
    if not present or len(present) == len(names):
        return ds
    log.info("projection des colonnes : %s", ",".join(present))
    return select(present)


class HFLoader:
    """Charge un dataset depuis le Hub HuggingFace, **en streaming par défaut**.

    Pourquoi le streaming par défaut : ``load_dataset`` classique télécharge le
    dataset *entier* (cache Arrow) avant la première ligne — des dizaines de Go
    pour les corpus audio, et une pression mémoire/disque inutile puisque le
    pipeline ne lit chaque ligne qu'une fois. En streaming, la mémoire reste
    bornée à un lot.

    Robustesse : si le dataset repose sur un **script de chargement** (désormais
    refusé par ``datasets``), on bascule automatiquement sur la branche
    **parquet auto-convertie** par HuggingFace — solution générale, sans logique
    spécifique à un dataset.
    """

    def __init__(self, streaming: bool = True, batch_size: int = 1000) -> None:
        self.streaming = streaming
        self.batch_size = batch_size

    def load(
        self,
        dataset_id: str,
        split: str = "train",
        config: str | None = None,
        columns: list[str] | None = None,
    ) -> Iterable[dict[str, Any]]:
        from datasets import load_dataset

        # Projection de colonnes : lit seulement les colonnes texte via la branche
        # parquet (évite de télécharger l'audio/image des datasets lourds).
        if columns:
            urls = _parquet_urls(dataset_id, split, config)
            if urls:
                log.info(
                    "%s : lecture colonnaire (%s) sur %d parquet(s)",
                    dataset_id,
                    ",".join(columns),
                    len(urls),
                )
                return _iter_parquet_columns(urls, columns, batch_size=self.batch_size)
            log.warning("%s : pas de parquet pour projection, chargement standard", dataset_id)

        try:
            ds = (
                load_dataset(dataset_id, config, split=split, streaming=self.streaming)
                if config
                else load_dataset(dataset_id, split=split, streaming=self.streaming)
            )
            return _project(ds, columns)
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
