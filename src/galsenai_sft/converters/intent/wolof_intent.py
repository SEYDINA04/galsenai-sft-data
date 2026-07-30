"""Converters d'intent classification (+ slot filling).

- INJONGO (wolof, gold, localisé) : intent + slots (colonne ``target``).
- WolBanking77 : intent bancaire (colonne ``input_wo``).
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from galsenai_sft.converters.base import BaseConverter
from galsenai_sft.converters.prompts import (
    INTENT_TEMPLATES,
    SLOT_TEMPLATES,
)
from galsenai_sft.core.schema import Message, Role, Sample, TaskType
from galsenai_sft.registry import register


def _parse_injongo_target(target: str) -> dict[str, str]:
    """Parse 'CITY: Dakaar $$ TIME: 11:55' -> {'CITY': 'Dakaar', 'TIME': '11:55'}."""
    slots: dict[str, str] = {}
    if not target:
        return slots
    for part in target.split("$$"):
        if ":" in part:
            key, val = part.split(":", 1)
            key, val = key.strip(), val.strip()
            if key and val:
                slots[key] = val
    return slots


@register("masakhane/InjongoIntent")
class InjongoIntent(BaseConverter):
    """INJONGO : produit un sample d'intent + (si slots) un sample de slot filling."""

    task: ClassVar[TaskType] = TaskType.INTENT
    text_col: ClassVar[str] = "text"
    intent_col: ClassVar[str] = "intent"
    target_col: ClassVar[str] = "target"

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        text = str(row.get(self.text_col) or "").strip()
        intent = str(row.get(self.intent_col) or "").strip()
        if not text or not intent:
            return []

        out: list[Sample] = []

        # 1) Détection d'intention
        lang = self.pick_lang()
        user = self._rng.choice(INTENT_TEMPLATES[lang]).format(text=text)
        out.append(
            self.make_sample(
                [
                    Message(role=Role.USER, content=user),
                    Message(role=Role.ASSISTANT, content=intent),
                ],
                prompt_lang=lang,
                subtask="intent",
            )
        )

        # 2) Slot filling (si des slots existent) -> réponse JSON
        slots = _parse_injongo_target(str(row.get(self.target_col) or ""))
        if slots:
            lang2 = self.pick_lang()
            user2 = self._rng.choice(SLOT_TEMPLATES[lang2]).format(text=text)
            answer = json.dumps(slots, ensure_ascii=False)
            out.append(
                self.make_sample(
                    [
                        Message(role=Role.USER, content=user2),
                        Message(role=Role.ASSISTANT, content=answer),
                    ],
                    prompt_lang=lang2,
                    subtask="slot_filling",
                )
            )
        return out


@register("karim155/WolBanking77")
class WolBanking77(BaseConverter):
    """Intent bancaire (77 classes). Énoncé wolof -> label d'intention."""

    task: ClassVar[TaskType] = TaskType.INTENT
    text_col: ClassVar[str] = "input_wo"
    label_col: ClassVar[str] = "label"

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        text = str(row.get(self.text_col) or "").strip()
        label = str(row.get(self.label_col) or "").strip()
        if not text or not label:
            return []
        lang = self.pick_lang()
        user = self._rng.choice(INTENT_TEMPLATES[lang]).format(text=text)
        return [
            self.make_sample(
                [
                    Message(role=Role.USER, content=user),
                    Message(role=Role.ASSISTANT, content=label),
                ],
                prompt_lang=lang,
                subtask="intent",
                domain="banking",
            )
        ]
