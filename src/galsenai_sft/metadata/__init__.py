"""Métadonnées : modèles, registre, génération du catalogue."""

from __future__ import annotations

from galsenai_sft.metadata.catalog import load_registry, render_catalog, write_catalog
from galsenai_sft.metadata.models import (
    ConversionStatus,
    DatasetMeta,
    LicenseStatus,
)

__all__ = [
    "ConversionStatus",
    "DatasetMeta",
    "LicenseStatus",
    "load_registry",
    "render_catalog",
    "write_catalog",
]
