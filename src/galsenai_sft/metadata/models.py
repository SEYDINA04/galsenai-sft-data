"""Modèles de métadonnées des datasets.

Chaque dataset expose automatiquement : name, version, language, task, license,
citation, source_url, n_samples, splits, checksum, conversion_status.
Les infos statiques (licence, url, citation) viennent de
``metadata/datasets_registry.yaml`` ; la tâche vient du converter enregistré ;
les compteurs/checksum sont renseignés après conversion.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ConversionStatus(StrEnum):
    PLANNED = "planned"  # converter écrit, pas encore exécuté
    CONVERTED = "converted"  # samples produits
    VALIDATED = "validated"  # passé les validators
    FAILED = "failed"


class LicenseStatus(StrEnum):
    PERMISSIVE = "permissive"
    NON_COMMERCIAL = "non-commercial"
    RESTRICTED = "restricted"
    UNVERIFIED = "unverified"


class DatasetMeta(BaseModel):
    """Métadonnées d'un dataset source intégré à la plateforme."""

    dataset_id: str
    task: str
    language: str = "wo"
    license: str = "unverified"
    license_status: LicenseStatus = LicenseStatus.UNVERIFIED
    commercial_ok: bool = False
    source_url: str = ""
    citation: str = ""
    version: str = "1.0"
    # Renseignés après conversion :
    n_samples: int | None = None
    splits: dict[str, int] = Field(default_factory=dict)
    checksum: str | None = None
    conversion_status: ConversionStatus = ConversionStatus.PLANNED
    notes: str = ""
