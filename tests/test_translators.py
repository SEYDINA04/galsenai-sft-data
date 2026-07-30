"""Tests des traducteurs (interface, cache, QE, review) — sans moteur réel."""

from __future__ import annotations

from galsenai_sft.translators import (
    CachingTranslator,
    EchoTranslator,
    ReviewStatus,
    TranslationRequest,
    approved_translations,
)
from galsenai_sft.translators.pipeline import translate_batch
from galsenai_sft.translators.quality import LIDQualityEstimator, QualityScore
from galsenai_sft.translators.review import (
    ReviewItem,
    read_review_queue,
    write_review_queue,
)


class _StubEstimator:
    """QE factice : passe si le texte contient 'ok'."""

    def score(self, result) -> QualityScore:
        passed = "ok" in result.translated.lower()
        return QualityScore(score=1.0 if passed else 0.0, passed=passed)


def test_echo_translator():
    t = EchoTranslator()
    r = t.translate(TranslationRequest(text="Bonjour", source_lang="fr"))
    assert r.translated == "Bonjour"
    assert r.engine == "echo"


def test_caching_translator(tmp_path):
    cache = tmp_path / "cache.jsonl"
    t = CachingTranslator(EchoTranslator(), cache)
    req = TranslationRequest(text="Merci", source_lang="fr")
    t.translate(req)
    # Recharger depuis le cache disque
    t2 = CachingTranslator(EchoTranslator(), cache)
    r = t2.translate(req)
    assert "cache" in r.engine
    assert r.translated == "Merci"


def test_pipeline_with_stub_estimator(tmp_path):
    items = list(
        translate_batch(
            ["texte ok", "texte mauvais", ""],
            source_lang="fr",
            translator=EchoTranslator(),
            estimator=_StubEstimator(),
        )
    )
    assert len(items) == 2  # vide ignoré
    assert items[0].qe_passed is True
    assert items[1].qe_passed is False


def test_review_workflow(tmp_path):
    queue = tmp_path / "review.jsonl"
    items = [
        ReviewItem(
            source_text="Bonjour",
            source_lang="fr",
            target_lang="wo",
            translation="Salaam",
            engine="echo",
            qe_score=1.0,
            qe_passed=True,
        ),
        ReviewItem(
            source_text="Merci",
            source_lang="fr",
            target_lang="wo",
            translation="???",
            engine="echo",
            qe_score=0.0,
            qe_passed=False,
        ),
    ]
    write_review_queue(items, queue)

    # Simuler l'annotation humaine
    annotated = list(read_review_queue(queue))
    annotated[0].status = ReviewStatus.APPROVED
    annotated[1].status = ReviewStatus.APPROVED
    annotated[1].correction = "Jërëjëf"
    write_review_queue(annotated, queue)

    approved = list(approved_translations(queue))
    assert len(approved) == 2
    assert approved[1].final_text == "Jërëjëf"  # correction retenue


def test_lid_estimator_empty():
    qe = LIDQualityEstimator(identifier=object())  # ne sera pas appelé sur vide
    from galsenai_sft.translators.base import TranslationResult

    res = TranslationResult(
        text="x", source_lang="fr", target_lang="wo", translated="  ", engine="echo"
    )
    score = qe.score(res)
    assert score.passed is False
