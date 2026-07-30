"""Estimation de qualité (QE) d'une traduction vers le wolof.

Backend-agnostique (Protocol) ; l'implémentation par défaut s'appuie sur le LID
(GlotLID v3) : une traduction est *suspecte* si le texte produit n'est pas
détecté comme wolof au-dessus du seuil. Des backends plus fins (SSA-COMET-QE)
pourront être branchés plus tard.

Cette QE légère sert de premier filtre automatique avant la revue humaine.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from galsenai_sft.translators.base import TranslationResult


class QualityScore(BaseModel):
    score: float  # 0..1
    passed: bool
    reason: str = ""


@runtime_checkable
class QualityEstimator(Protocol):
    def score(self, result: TranslationResult) -> QualityScore: ...


class LIDQualityEstimator:
    """QE basée sur la détection de langue de la sortie (wolof attendu)."""

    def __init__(self, identifier=None, min_confidence: float = 0.5) -> None:
        self._identifier = identifier
        self.min_confidence = min_confidence

    def _get_identifier(self):
        if self._identifier is None:
            from galsenai_sft.validators.lid import get_identifier

            self._identifier = get_identifier()
        return self._identifier

    def score(self, result: TranslationResult) -> QualityScore:
        if not result.translated.strip():
            return QualityScore(score=0.0, passed=False, reason="traduction vide")
        label, conf = self._get_identifier().predict(result.translated)
        target = (
            f"{result.target_lang}_Latn" if "_" not in result.target_lang else result.target_lang
        )
        passed = label == target and conf >= self.min_confidence
        reason = "" if passed else f"langue détectée={label} (conf={conf:.2f}), attendu={target}"
        return QualityScore(score=conf if passed else 0.0, passed=passed, reason=reason)
