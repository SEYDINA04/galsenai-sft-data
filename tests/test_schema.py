"""Tests du schéma canonique (validation ChatML)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from galsenai_sft.core.schema import Message, Role, Sample, TaskType, ToolCall


def test_valid_sample(simple_sample):
    assert simple_sample.n_turns() == 1
    assert simple_sample.messages[-1].role is Role.ASSISTANT


def test_must_end_with_assistant():
    with pytest.raises(ValidationError):
        Sample(
            messages=[Message(role=Role.USER, content="Naka?")],
            task=TaskType.QA,
            source="s",
        )


def test_system_must_be_first():
    with pytest.raises(ValidationError):
        Sample(
            messages=[
                Message(role=Role.USER, content="Naka?"),
                Message(role=Role.SYSTEM, content="sys"),
                Message(role=Role.ASSISTANT, content="Maa ngi fi."),
            ],
            task=TaskType.QA,
            source="s",
        )


def test_requires_user_turn():
    with pytest.raises(ValidationError):
        Sample(
            messages=[Message(role=Role.ASSISTANT, content="Maa ngi fi.")],
            task=TaskType.QA,
            source="s",
        )


def test_empty_message_rejected():
    with pytest.raises(ValidationError):
        Message(role=Role.USER, content="   ")


def test_tool_call_message_allowed():
    m = Message(
        role=Role.ASSISTANT,
        tool_calls=[ToolCall(name="get_weather", arguments={"city": "Dakar"})],
    )
    assert m.tool_calls[0].name == "get_weather"


def test_tool_name_only_for_tool_role():
    with pytest.raises(ValidationError):
        Message(role=Role.USER, content="x", name="tool")
