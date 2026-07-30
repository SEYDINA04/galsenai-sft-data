"""Pipeline de traduction : batch -> QE -> file de revue.

Orchestre les briques du lot 5. Ne dépend PAS d'un moteur concret : on lui passe
un ``Translator`` (EchoTranslator par défaut, ou un vrai backend plus tard) et un
``QualityEstimator``. La revue humaine se fait ensuite sur la file JSONL produite.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from galsenai_sft.core.logging import get_logger
from galsenai_sft.translators.base import (
    EchoTranslator,
    TranslationRequest,
    Translator,
)
from galsenai_sft.translators.quality import LIDQualityEstimator, QualityEstimator
from galsenai_sft.translators.review import ReviewItem, make_review_item, write_review_queue

log = get_logger(__name__)


def translate_batch(
    texts: Iterable[str],
    source_lang: str,
    target_lang: str = "wo",
    translator: Translator | None = None,
    estimator: QualityEstimator | None = None,
) -> Iterator[ReviewItem]:
    """Traduit un lot de textes et produit des items de revue (avec score QE)."""
    translator = translator or EchoTranslator()
    estimator = estimator or LIDQualityEstimator()
    for text in texts:
        text = (text or "").strip()
        if not text:
            continue
        result = translator.translate(
            TranslationRequest(text=text, source_lang=source_lang, target_lang=target_lang)
        )
        qe = estimator.score(result)
        yield make_review_item(result, qe)


def run_translation_pipeline(
    texts: Iterable[str],
    source_lang: str,
    out_queue: str | Path,
    target_lang: str = "wo",
    translator: Translator | None = None,
    estimator: QualityEstimator | None = None,
) -> int:
    """Traduit + écrit la file de revue. Retourne le nombre d'items."""
    items = list(
        translate_batch(texts, source_lang, target_lang, translator=translator, estimator=estimator)
    )
    n = write_review_queue(items, out_queue)
    passed = sum(1 for i in items if i.qe_passed)
    log.info("traduction : %d items (%d QE-ok) -> %s", n, passed, out_queue)
    return n
