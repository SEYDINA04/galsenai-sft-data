"""Traducteurs : interface pluggable + QE + revue humaine (étape 3, moteur différé).

Le moteur réel (LLM frontière, pivot FR) n'est PAS branché : seul un
``EchoTranslator`` de test est fourni. Le pipeline est prêt à l'accueillir.
"""

from __future__ import annotations

from galsenai_sft.translators.base import (
    CachingTranslator,
    EchoTranslator,
    TranslationRequest,
    TranslationResult,
    Translator,
)
from galsenai_sft.translators.quality import (
    LIDQualityEstimator,
    QualityEstimator,
    QualityScore,
)
from galsenai_sft.translators.review import (
    ReviewItem,
    ReviewStatus,
    approved_translations,
    read_review_queue,
    write_review_queue,
)

__all__ = [
    "CachingTranslator",
    "EchoTranslator",
    "LIDQualityEstimator",
    "QualityEstimator",
    "QualityScore",
    "ReviewItem",
    "ReviewStatus",
    "TranslationRequest",
    "TranslationResult",
    "Translator",
    "approved_translations",
    "read_review_queue",
    "write_review_queue",
]
