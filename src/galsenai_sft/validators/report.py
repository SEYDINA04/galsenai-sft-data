"""Types communs aux validators : problèmes détectés et rapport agrégé.

Chaque validator produit une liste de :class:`Issue`. Un :class:`ValidationReport`
agrège les issues sur un lot de Samples et distingue erreurs (bloquantes) et
avertissements (non bloquants), à la manière des quality gates.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, Field


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class Issue(BaseModel):
    """Un problème détecté sur un Sample (ou globalement)."""

    code: str  # ex. "empty_content", "duplicate", "role_ordering"
    severity: Severity
    message: str
    sample_index: int | None = None  # position dans le lot (None = global)
    source: str | None = None


class ValidationReport(BaseModel):
    """Rapport agrégé d'une passe de validation."""

    total: int = 0
    issues: list[Issue] = Field(default_factory=list)

    # --- construction ---
    def add(self, issue: Issue) -> None:
        self.issues.append(issue)

    def extend(self, issues: list[Issue]) -> None:
        self.issues.extend(issues)

    # --- lecture ---
    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """Vrai si aucune erreur bloquante."""
        return not self.errors

    def counts_by_code(self) -> dict[str, int]:
        return dict(Counter(i.code for i in self.issues))

    def summary(self) -> str:
        c = self.counts_by_code()
        parts = ", ".join(f"{k}={v}" for k, v in sorted(c.items())) or "aucun"
        return (
            f"{self.total} samples · {len(self.errors)} erreurs · "
            f"{len(self.warnings)} avertissements · [{parts}]"
        )
