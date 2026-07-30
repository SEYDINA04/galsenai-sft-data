"""File de revue humaine + workflow d'approbation des traductions.

Les traductions qui échouent la QE automatique (ou un échantillon des réussies)
sont écrites dans une **file de revue** JSONL. Un relecteur (communauté GalsenAI)
marque chaque item ``approved`` / ``rejected`` (+ correction éventuelle). Seuls
les items approuvés sont intégrés au dataset final.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from galsenai_sft.core.io import write_jsonl
from galsenai_sft.translators.base import TranslationResult
from galsenai_sft.translators.quality import QualityScore


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewItem(BaseModel):
    source_text: str
    source_lang: str
    target_lang: str
    translation: str
    engine: str
    qe_score: float
    qe_passed: bool
    qe_reason: str = ""
    status: ReviewStatus = ReviewStatus.PENDING
    correction: str | None = None  # traduction corrigée par le relecteur

    @property
    def final_text(self) -> str:
        """Texte retenu : la correction si fournie, sinon la traduction."""
        return self.correction if self.correction else self.translation


def make_review_item(result: TranslationResult, qe: QualityScore) -> ReviewItem:
    return ReviewItem(
        source_text=result.text,
        source_lang=result.source_lang,
        target_lang=result.target_lang,
        translation=result.translated,
        engine=result.engine,
        qe_score=qe.score,
        qe_passed=qe.passed,
        qe_reason=qe.reason,
        # Auto-approbation possible des items ayant passé la QE (configurable).
        status=ReviewStatus.PENDING,
    )


def write_review_queue(items: Iterable[ReviewItem], path: str | Path) -> int:
    """Écrit la file de revue en JSONL."""
    return write_jsonl((i.model_dump(mode="json") for i in items), path)


def read_review_queue(path: str | Path) -> Iterator[ReviewItem]:
    """Relit une file de revue (après annotation)."""
    import json

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield ReviewItem.model_validate(json.loads(line))


def approved_translations(path: str | Path) -> Iterator[ReviewItem]:
    """Itère uniquement les items approuvés (avec correction éventuelle)."""
    for item in read_review_queue(path):
        if item.status is ReviewStatus.APPROVED:
            yield item
