"""Classe de base des converters + gabarits de consignes bilingues (wo/fr).

Un converter transforme une **ligne brute** d'un dataset source en une liste de
:class:`Sample` canoniques. C'est le seul point d'extension à écrire pour
intégrer un nouveau dataset.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any, ClassVar

from galsenai_sft.core.schema import PromptLang, Sample, TaskType


class BaseConverter(ABC):
    """Contrat commun à tous les converters (plugin).

    Sous-classe minimale à implémenter :
        - ``task`` : la famille de tâche produite ;
        - ``convert_row(row)`` : ligne brute -> 0..n Samples.

    ``dataset_id`` est renseigné automatiquement par le décorateur ``@register``.
    """

    #: renseigné par @register
    dataset_id: ClassVar[str] = ""
    #: famille de tâche produite (à définir en sous-classe)
    task: ClassVar[TaskType] = TaskType.OTHER
    #: langues de consigne à générer (bilingue par défaut)
    prompt_langs: ClassVar[tuple[PromptLang, ...]] = (PromptLang.WO, PromptLang.FR)

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    # --- API publique ----------------------------------------------------- #
    @abstractmethod
    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        """Convertit une ligne brute en 0, 1 ou plusieurs Samples canoniques."""

    def convert(self, rows: Iterable[dict[str, Any]]) -> Iterator[Sample]:
        """Convertit un flux de lignes (streaming), en ignorant les lignes vides."""
        for row in rows:
            yield from self.convert_row(row)

    # --- Helpers pour les sous-classes ------------------------------------ #
    def pick_lang(self) -> PromptLang:
        """Choisit une langue de consigne de façon déterministe (seed)."""
        return self._rng.choice(self.prompt_langs)

    def make_sample(
        self,
        messages: list,
        *,
        prompt_lang: PromptLang,
        sample_id: str | None = None,
        **meta: Any,
    ) -> Sample:
        """Fabrique un Sample en injectant tâche + source automatiquement."""
        return Sample(
            messages=messages,
            task=self.task,
            source=self.dataset_id,
            prompt_lang=prompt_lang,
            id=sample_id,
            meta=meta,
        )
