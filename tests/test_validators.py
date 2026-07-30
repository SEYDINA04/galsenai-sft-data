"""Tests des validators (schéma, qualité, décontamination, statistiques)."""

from __future__ import annotations

from galsenai_sft.core.schema import Message, Role, Sample, TaskType, ToolCall
from galsenai_sft.validators import (
    compute_statistics,
    sample_fingerprint,
    validate_quality,
    validate_raw_chatml,
)
from galsenai_sft.validators.decontamination import decontaminate
from galsenai_sft.validators.quality_validator import filter_quality


def _s(user: str, assistant: str, **kw) -> Sample:
    return Sample(
        messages=[
            Message(role=Role.USER, content=user),
            Message(role=Role.ASSISTANT, content=assistant),
        ],
        task=kw.get("task", TaskType.TRANSLATION),
        source=kw.get("source", "test/ds"),
    )


# --- schema_validator ---
def test_raw_chatml_valid():
    row = {
        "messages": [
            {"role": "user", "content": "Naka?"},
            {"role": "assistant", "content": "Maa ngi fi."},
        ]
    }
    assert validate_raw_chatml(row) == []


def test_raw_chatml_no_final_assistant():
    row = {"messages": [{"role": "user", "content": "Naka?"}]}
    codes = {i.code for i in validate_raw_chatml(row)}
    assert "no_final_assistant" in codes


def test_raw_chatml_invalid_role_and_empty():
    row = {"messages": [{"role": "robot", "content": ""}, {"role": "assistant", "content": "ok"}]}
    codes = {i.code for i in validate_raw_chatml(row)}
    assert "invalid_role" in codes and "empty_content" in codes


# --- quality_validator ---
def test_duplicate_detection():
    rep = validate_quality([_s("Bonjour", "Salaam"), _s("Bonjour", "Salaam")])
    assert not rep.ok
    assert rep.counts_by_code().get("duplicate") == 1


def test_echo_answer_warning():
    rep = validate_quality([_s("Dakar", "Dakar")])
    assert rep.ok  # warning seulement
    assert "echo_answer" in rep.counts_by_code()


def test_filter_quality_removes_duplicates():
    kept = list(filter_quality([_s("a", "b"), _s("a", "b"), _s("c", "d")]))
    assert len(kept) == 2


def test_fingerprint_stable():
    assert sample_fingerprint(_s("X", "Y")) == sample_fingerprint(_s("x", "y "))


# --- decontamination ---
def test_decontaminate_removes_seen_text():
    import hashlib

    def h(t):
        return hashlib.sha1(" ".join(t.lower().split()).encode()).hexdigest()

    index = {h("Salaamaalekum")}
    kept = list(decontaminate([_s("Bonjour", "Salaamaalekum"), _s("Merci", "Jërëjëf")], index))
    assert len(kept) == 1
    assert kept[0].messages[-1].content == "Jërëjëf"


def test_decontaminate_noop_without_index():
    samples = [_s("a", "b"), _s("c", "d")]
    assert len(list(decontaminate(samples, set()))) == 2


# --- statistics ---
def test_statistics():
    tool_sample = Sample(
        messages=[
            Message(role=Role.USER, content="Météo à Dakar ?"),
            Message(
                role=Role.ASSISTANT,
                tool_calls=[ToolCall(name="weather", arguments={"city": "Dakar"})],
            ),
        ],
        task=TaskType.TOOL_USE,
        source="test/tools",
    )
    stats = compute_statistics([_s("a", "bb"), tool_sample])
    assert stats.total == 2
    assert stats.by_task["translation"] == 1
    assert stats.with_tool_calls == 1
    assert stats.by_source["test/ds"] == 1
