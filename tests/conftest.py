"""Fixtures partagées : rend le package importable et fournit des Samples."""

from __future__ import annotations

import pytest

from galsenai_sft.core.schema import Message, Role, Sample, TaskType


@pytest.fixture
def simple_sample() -> Sample:
    return Sample(
        messages=[
            Message(role=Role.USER, content="Tekkil lii ci wolof: Bonjour"),
            Message(role=Role.ASSISTANT, content="Salaamaalekum"),
        ],
        task=TaskType.TRANSLATION,
        source="test/dataset",
    )


@pytest.fixture
def system_sample() -> Sample:
    return Sample(
        messages=[
            Message(role=Role.SYSTEM, content="Yaw mi, jàngalekat nga wolof."),
            Message(role=Role.USER, content="Naka nga def?"),
            Message(role=Role.ASSISTANT, content="Maa ngi fi rekk."),
        ],
        task=TaskType.INSTRUCTION,
        source="test/dataset",
    )
