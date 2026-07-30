"""Converters de classification de texte (sentiment, émotion, thème).

Un converter générique paramétrable : colonne texte, colonne label, et une clé
de tâche (``sentiment`` | ``topic`` | ``emotion`` | …) qui sélectionne la
formulation bilingue de la consigne.
"""

from __future__ import annotations

from typing import Any, ClassVar

from galsenai_sft.converters.base import BaseConverter
from galsenai_sft.converters.prompts import CLASSIFY_TASKS, CLASSIFY_TEMPLATES
from galsenai_sft.core.schema import Message, Role, Sample, TaskType
from galsenai_sft.registry import register


class ClassificationConverter(BaseConverter):
    """Texte -> label. Paramétré par colonnes et clé de tâche de consigne."""

    task: ClassVar[TaskType] = TaskType.CLASSIFICATION

    text_col: ClassVar[str] = "text"
    label_col: ClassVar[str] = "label"
    #: clé dans CLASSIFY_TASKS (sentiment/topic/emotion/intent)
    classify_task: ClassVar[str] = "topic"

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        text = str(row.get(self.text_col) or "").strip()
        label = str(row.get(self.label_col) or "").strip()
        if not text or not label:
            return []
        lang = self.pick_lang()
        task_desc = CLASSIFY_TASKS[self.classify_task][lang]
        template = self._rng.choice(CLASSIFY_TEMPLATES[lang])
        user = template.format(task_desc=task_desc, text=text)
        messages = [
            Message(role=Role.USER, content=user),
            Message(role=Role.ASSISTANT, content=label),
        ]
        return [self.make_sample(messages, prompt_lang=lang, classify_task=self.classify_task)]


@register("michsethowusu/wolof-sentiments-corpus")
class WolofSentiments(ClassificationConverter):
    text_col = "Wolof"
    label_col = "sentiment"
    classify_task = "sentiment"


@register("michsethowusu/wolof-emotions-corpus")
class WolofEmotions(ClassificationConverter):
    text_col = "Wolof"
    label_col = "emotion"
    classify_task = "emotion"


@register("Davlan/sib200")
class Sib200Topic(ClassificationConverter):
    text_col = "text"
    label_col = "category"
    classify_task = "topic"
