"""Tests du filtre de contenu (bruit + LID cible wolof)."""

from __future__ import annotations

from galsenai_sft.core.schema import Message, Role, Sample, TaskType
from galsenai_sft.validators.content_filter import WolofTargetFilter, is_noisy_text


def test_is_noisy_text():
    assert is_noisy_text("màà caro kii dormaiit et tout !!!!") is True  # ponctuation
    assert is_noisy_text("aaaa bbbb") is True  # répétition de caractères
    assert is_noisy_text("ok") is True  # trop court
    assert is_noisy_text("1234 5678 9999") is True  # trop peu de lettres
    assert is_noisy_text("Maa ngi fi rekk ci sama kër") is False  # propre


class _FakeLID:
    """LID factice : renvoie wol_Latn seulement pour les textes 'wolof:...'."""

    def __init__(self, mapping):
        self.mapping = mapping

    def predict(self, text):
        return self.mapping.get(text, ("fra_Latn", 0.9))


def _sample(assistant: str) -> Sample:
    return Sample(
        messages=[
            Message(role=Role.USER, content="Consigne quelconque ici"),
            Message(role=Role.ASSISTANT, content=assistant),
        ],
        task=TaskType.INSTRUCTION,
        source="test/wori",
    )


def test_wolof_target_filter_keeps_wolof():
    good = "Maa ngi fi rekk ci sama kër"
    lid = _FakeLID({good: ("wol_Latn", 0.99)})
    f = WolofTargetFilter(identifier=lid)
    assert f.keep(_sample(good)) is True


def test_wolof_target_filter_drops_non_wolof():
    fr = "Le jeudi 16 octobre l'étude est terminée"
    lid = _FakeLID({fr: ("fra_Latn", 0.98)})
    f = WolofTargetFilter(identifier=lid)
    assert f.keep(_sample(fr)) is False


def test_wolof_target_filter_drops_noisy_before_lid():
    # bruit détecté sans même appeler le LID
    f = WolofTargetFilter(identifier=_FakeLID({}))
    assert f.keep(_sample("dormaiit !!!!")) is False


def test_release_libere_le_modele_lid():
    """Le LID (~1,6 Go) doit être libérable : c'est l'essentiel de l'empreinte."""
    from galsenai_sft.validators.content_filter import WolofTargetFilter
    from galsenai_sft.validators.lid import get_identifier

    class FakeLID:
        def predict(self, text):
            return ("wol_Latn", 0.99)

    f = WolofTargetFilter(identifier=FakeLID())
    f.release()
    assert f._identifier is None
    assert get_identifier.cache_info().currsize == 0
