"""Validators : structure ChatML, qualité, LID, décontamination, statistiques.

Le LID (``lid``) et la décontamination (``decontamination``) ont des dépendances
lourdes (fasttext, pyarrow) importées paresseusement dans leurs modules ; ils ne
sont donc pas ré-exportés ici pour garder l'import du package léger.
"""

from __future__ import annotations

from galsenai_sft.validators.quality_validator import (
    filter_quality,
    sample_fingerprint,
    validate_quality,
)
from galsenai_sft.validators.report import Issue, Severity, ValidationReport
from galsenai_sft.validators.schema_validator import (
    validate_raw_chatml,
    validate_sample_structure,
)
from galsenai_sft.validators.statistics import Statistics, compute_statistics

__all__ = [
    "Issue",
    "Severity",
    "ValidationReport",
    "Statistics",
    "compute_statistics",
    "filter_quality",
    "sample_fingerprint",
    "validate_quality",
    "validate_raw_chatml",
    "validate_sample_structure",
]
