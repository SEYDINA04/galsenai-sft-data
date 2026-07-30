"""Statistiques d'un lot de Samples : volumétrie par tâche / source / langue de
consigne, distribution de longueurs, part multi-tours. Sert au rapport de build
et à la data card.

Les statistiques sont **incrémentales** (:class:`StatisticsAccumulator`) : le
build les met à jour sample par sample, sans jamais garder le lot en mémoire.
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


class StatisticsAccumulator:
    """Agrégateur en une passe, à mémoire constante.

    Ne retient que des compteurs (bornés par le nombre de tâches/sources/langues),
    jamais les Samples : on peut agréger des millions d'exemples sans risque.
    """

    def __init__(self) -> None:
        self._task: Counter[str] = Counter()
        self._source: Counter[str] = Counter()
        self._lang: Counter[str] = Counter()
        self.total = 0
        self.multi_turn = 0
        self.with_tool_calls = 0
        self.user_chars = 0
        self.assistant_chars = 0

    def update(self, sample: Sample) -> None:
        """Intègre un Sample supplémentaire."""
        self.total += 1
        self._task[sample.task.value] += 1
        self._source[sample.source] += 1
        self._lang[sample.prompt_lang.value] += 1
        if sample.n_turns() > 1:
            self.multi_turn += 1
        for m in sample.messages:
            if m.tool_calls:
                self.with_tool_calls += 1
            if m.role is Role.USER:
                self.user_chars += len(m.content)
            elif m.role is Role.ASSISTANT:
                self.assistant_chars += len(m.content)

    @property
    def by_task(self) -> dict[str, int]:
        return dict(self._task.most_common())

    def result(self) -> Statistics:
        """Fige les compteurs en un objet :class:`Statistics`."""
        return Statistics(
            total=self.total,
            by_task=self.by_task,
            by_source=dict(self._source.most_common()),
            by_prompt_lang=dict(self._lang.most_common()),
            multi_turn=self.multi_turn,
            with_tool_calls=self.with_tool_calls,
            total_user_chars=self.user_chars,
            total_assistant_chars=self.assistant_chars,
        )


def compute_statistics(samples: Iterable[Sample]) -> Statistics:
    """Calcule les statistiques agrégées sur un flux de Samples (une passe)."""
    acc = StatisticsAccumulator()
    for s in samples:
        acc.update(s)
    return acc.result()
