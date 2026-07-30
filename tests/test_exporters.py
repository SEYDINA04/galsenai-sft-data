"""Tests des exporters (ChatML, Alpaca, ShareGPT)."""

from __future__ import annotations

import pytest

from galsenai_sft.exporters import get_exporter, to_alpaca, to_chatml, to_sharegpt
from galsenai_sft.exporters.alpaca import is_alpaca_compatible


def test_chatml(simple_sample):
    row = to_chatml(simple_sample)
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][1]["role"] == "assistant"
    assert row["source"] == "test/dataset"
    assert row["prompt_lang"] == "wo"


def test_alpaca_mono_turn(simple_sample):
    row = to_alpaca(simple_sample)
    assert "Bonjour" in row["instruction"]
    assert row["output"] == "Salaamaalekum"
    assert row["input"] == ""


def test_alpaca_with_system(system_sample):
    assert is_alpaca_compatible(system_sample)
    row = to_alpaca(system_sample)
    # system préfixé à l'instruction
    assert "jàngalekat" in row["instruction"]
    assert "Naka nga def?" in row["instruction"]
    assert row["output"] == "Maa ngi fi rekk."


def test_sharegpt(system_sample):
    row = to_sharegpt(system_sample)
    froms = [c["from"] for c in row["conversations"]]
    assert froms == ["system", "human", "gpt"]


def test_get_exporter_unknown():
    with pytest.raises(KeyError):
        get_exporter("parquetxyz")
