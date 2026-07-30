"""Fixtures partagées : rend le package importable et fournit des Samples."""

from __future__ import annotations

import pytest

from galsenai_sft.core.config import get_settings
from galsenai_sft.core.schema import Message, Role, Sample, TaskType


@pytest.fixture(autouse=True)
def _garde_fou_neutre():
    """Neutralise le plancher mémoire : les tests ne doivent pas dépendre de la
    RAM libre de la machine d'exécution (le garde-fou a ses propres tests)."""
    settings = get_settings()
    previous = settings.memory.min_available_mb
    settings.memory.min_available_mb = 0
    yield
    settings.memory.min_available_mb = previous


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
