"""Converter QA : AfriQA (wolof).

Question wolof -> réponse wolof. La colonne ``answers`` est une liste
sérialisée en chaîne (ex. ``"['réponse']"``) : on la parse prudemment.
"""

from __future__ import annotations

import ast
from typing import Any, ClassVar

from galsenai_sft.converters.base import BaseConverter
from galsenai_sft.converters.prompts import QA_TEMPLATES
from galsenai_sft.core.schema import Message, Role, Sample, TaskType
from galsenai_sft.registry import register


def _first_answer(raw: Any) -> str:
    """Extrait la première réponse d'un champ answers (liste ou str-liste)."""
    if raw is None:
        return ""
    if isinstance(raw, list):
        return str(raw[0]).strip() if raw else ""
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0]).strip()
        except (ValueError, SyntaxError):
            pass
    return s


@register("masakhane/afriqa")
class AfriQA(BaseConverter):
    task: ClassVar[TaskType] = TaskType.QA
    question_col: ClassVar[str] = "question"
    answers_col: ClassVar[str] = "answers"

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        question = str(row.get(self.question_col) or "").strip()
        answer = _first_answer(row.get(self.answers_col))
        if not question or not answer:
            return []
        lang = self.pick_lang()
        user = self._rng.choice(QA_TEMPLATES[lang]).format(question=question)
        return [
            self.make_sample(
                [
                    Message(role=Role.USER, content=user),
                    Message(role=Role.ASSISTANT, content=answer),
                ],
                prompt_lang=lang,
            )
        ]
