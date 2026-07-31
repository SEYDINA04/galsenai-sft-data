"""Converter d'inférence textuelle (NLI) : AfriXNLI wolof.

Le raisonnement sur une paire de phrases (implication / neutre / contradiction)
n'existait pas dans le dataset. Rangé sous ``classification`` : la sortie est un
label unique pris dans un ensemble fermé, comme le sentiment ou la thématique.

Les libellés de sortie restent en anglais (`entailment`, `neutral`,
`contradiction`), conformément à la règle du projet : on localise la consigne,
jamais le schéma de réponse.
"""

from __future__ import annotations

from typing import Any, ClassVar

from galsenai_sft.converters.base import BaseConverter
from galsenai_sft.converters.prompts import NLI_TEMPLATES
from galsenai_sft.core.schema import Message, Role, Sample, TaskType
from galsenai_sft.registry import register

#: Ordre officiel XNLI des étiquettes entières.
NLI_LABELS: tuple[str, ...] = ("entailment", "neutral", "contradiction")


@register("masakhane/afrixnli")
class AfriXNLI(BaseConverter):
    task: ClassVar[TaskType] = TaskType.CLASSIFICATION

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        premise = str(row.get("premise") or "").strip()
        hypothesis = str(row.get("hypothesis") or "").strip()
        raw_label = row.get("label")
        if not premise or not hypothesis or raw_label is None:
            return []

        if isinstance(raw_label, int):
            if not 0 <= raw_label < len(NLI_LABELS):
                return []
            label = NLI_LABELS[raw_label]
        else:
            label = str(raw_label).strip().lower()
            if label not in NLI_LABELS:
                return []

        lang = self.pick_lang()
        user = self._rng.choice(NLI_TEMPLATES[lang]).format(premise=premise, hypothesis=hypothesis)
        return [
            self.make_sample(
                [
                    Message(role=Role.USER, content=user),
                    Message(role=Role.ASSISTANT, content=label),
                ],
                prompt_lang=lang,
                classify_task="nli",
            )
        ]
