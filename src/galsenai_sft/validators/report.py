"""Types communs aux validators : problèmes détectés et rapport agrégé.

Chaque validator produit une liste de :class:`Issue`. Un :class:`ValidationReport`
agrège les issues sur un lot de Samples et distingue erreurs (bloquantes) et
avertissements (non bloquants), à la manière des quality gates.

**Mémoire bornée** : sur un dataset de plusieurs centaines de milliers
d'exemples, conserver *chaque* problème coûterait des Go. Le rapport garde donc
tous les **compteurs** (exacts) mais seulement les ``max_issues`` premiers
exemples détaillés (``truncated=True`` au-delà).
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
    """Rapport agrégé d'une passe de validation (mémoire bornée)."""

    total: int = 0
    #: Échantillon détaillé des problèmes (borné par ``max_issues``).
    issues: list[Issue] = Field(default_factory=list)
    #: Compteurs exacts, quel que soit le volume.
    counts: dict[str, int] = Field(default_factory=dict)
    n_errors: int = 0
    n_warnings: int = 0
    #: Nombre maximal de problèmes détaillés conservés.
    max_issues: int = 1000
    truncated: bool = False

    # --- construction ---
    def add(self, issue: Issue) -> None:
        self.counts[issue.code] = self.counts.get(issue.code, 0) + 1
        if issue.severity is Severity.ERROR:
            self.n_errors += 1
        else:
            self.n_warnings += 1
        if len(self.issues) < self.max_issues:
            self.issues.append(issue)
        else:
            self.truncated = True

    def extend(self, issues: list[Issue]) -> None:
        for issue in issues:
            self.add(issue)

    # --- lecture ---
    @property
    def errors(self) -> list[Issue]:
        """Erreurs détaillées **conservées** (voir ``n_errors`` pour le total)."""
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        """Avertissements détaillés conservés (total exact : ``n_warnings``)."""
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """Vrai si aucune erreur bloquante (compteur exact, pas l'échantillon)."""
        return self.n_errors == 0

    def counts_by_code(self) -> dict[str, int]:
        """Compteurs exacts par code de problème."""
        return dict(Counter(self.counts))

    def summary(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items())) or "aucun"
        suffix = f" (détail limité aux {self.max_issues} premiers)" if self.truncated else ""
        return (
            f"{self.total} samples · {self.n_errors} erreurs · "
            f"{self.n_warnings} avertissements · [{parts}]{suffix}"
        )
