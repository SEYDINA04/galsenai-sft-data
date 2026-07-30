"""Tests du registry (plugin) et du converter de traduction patron."""

from __future__ import annotations

from galsenai_sft.converters.translation.base_translation import TranslationConverter
from galsenai_sft.core.schema import Role, TaskType
from galsenai_sft.registry import available, get_converter


def test_discovery_finds_translation_converters():
    ids = available()
    assert "galsenai/french-wolof-translation" in ids


def test_get_converter_returns_class():
    cls = get_converter("galsenai/french-wolof-translation")
    assert issubclass(cls, TranslationConverter)
    assert cls.task is TaskType.TRANSLATION


def test_translation_conversion_forward_and_backward():
    cls = get_converter("galsenai/french-wolof-translation")
    conv = cls(seed=1)
    rows = [{"french": "Bonjour", "wolof": "Salaamaalekum"}]
    samples = list(conv.convert(rows))
    # bidirectionnel -> 2 samples (fr->wo et wo->fr)
    assert len(samples) == 2
    directions = {s.meta["direction"] for s in samples}
    assert directions == {"fr->wo", "wo->fr"}
    for s in samples:
        assert s.messages[0].role is Role.USER
        assert s.messages[-1].role is Role.ASSISTANT
        assert s.source == "galsenai/french-wolof-translation"


def test_translation_skips_empty():
    cls = get_converter("galsenai/french-wolof-translation")
    conv = cls()
    samples = list(conv.convert([{"french": "", "wolof": ""}]))
    assert samples == []
