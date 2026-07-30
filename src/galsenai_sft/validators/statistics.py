"""Statistiques d'un lot de Samples : volumétrie par tâche / source / langue de
consigne, distribution de longueurs, part multi-tours. Sert au rapport de build
et à la data card.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel, Field

from galsenai_sft.core.schema import Role, Sample


class Statistics(BaseModel):
    total: int = 0
    by_task: dict[str, int] = Field(default_factory=dict)
    by_source: dict[str, int] = Field(default_factory=dict)
    by_prompt_lang: dict[str, int] = Field(default_factory=dict)
    multi_turn: int = 0
    with_tool_calls: int = 0
    total_user_chars: int = 0
    total_assistant_chars: int = 0

    @property
    def avg_assistant_chars(self) -> float:
        return round(self.total_assistant_chars / self.total, 1) if self.total else 0.0


def compute_statistics(samples: Iterable[Sample]) -> Statistics:
    """Calcule les statistiques agrégées sur un flux de Samples (une passe)."""
    task = Counter()
    source = Counter()
    lang = Counter()
    total = 0
    multi_turn = 0
    with_tools = 0
    user_chars = 0
    asst_chars = 0

    for s in samples:
        total += 1
        task[s.task.value] += 1
        source[s.source] += 1
        lang[s.prompt_lang.value] += 1
        if s.n_turns() > 1:
            multi_turn += 1
        for m in s.messages:
            if m.tool_calls:
                with_tools += 1
            if m.role is Role.USER:
                user_chars += len(m.content)
            elif m.role is Role.ASSISTANT:
                asst_chars += len(m.content)

    return Statistics(
        total=total,
        by_task=dict(task.most_common()),
        by_source=dict(source.most_common()),
        by_prompt_lang=dict(lang.most_common()),
        multi_turn=multi_turn,
        with_tool_calls=with_tools,
        total_user_chars=user_chars,
        total_assistant_chars=asst_chars,
    )
