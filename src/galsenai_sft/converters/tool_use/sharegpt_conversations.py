"""Converter de conversations ShareGPT -> ChatML.

Générique pour tout dataset au format ``conversations = [{"from", "value"}]``.
Utilisé par Code-170k-wolof (assistant de code, signal agentique/tool use le
plus proche disponible en wolof). Multi-tours conservés tels quels.
"""

from __future__ import annotations

from typing import Any, ClassVar

from galsenai_sft.converters.base import BaseConverter
from galsenai_sft.core.schema import Message, PromptLang, Role, Sample, TaskType
from galsenai_sft.registry import register

_FROM_TO_ROLE = {
    "system": Role.SYSTEM,
    "human": Role.USER,
    "user": Role.USER,
    "gpt": Role.ASSISTANT,
    "assistant": Role.ASSISTANT,
    "tool": Role.TOOL,
}


class ShareGPTConverter(BaseConverter):
    """Convertit une colonne ``conversations`` ShareGPT en Sample ChatML."""

    task: ClassVar[TaskType] = TaskType.TOOL_USE
    conversations_col: ClassVar[str] = "conversations"
    # Consigne = celle du dataset (pas de gabarit ajouté) -> langue neutre.
    prompt_langs: ClassVar[tuple[PromptLang, ...]] = (PromptLang.WO,)

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        turns = row.get(self.conversations_col) or []
        messages: list[Message] = []
        for t in turns:
            if not isinstance(t, dict):
                continue
            role = _FROM_TO_ROLE.get(str(t.get("from") or "").lower())
            value = str(t.get("value") or "").strip()
            if role is None or not value:
                continue
            messages.append(Message(role=role, content=value))

        # Un Sample valide doit contenir user + finir par assistant.
        if len(messages) < 2:
            return []
        while messages and messages[-1].role is not Role.ASSISTANT:
            messages.pop()
        if not messages or not any(m.role is Role.USER for m in messages):
            return []

        return [self.make_sample(messages, prompt_lang=PromptLang.WO, n_turns=len(messages))]


@register("michsethowusu/Code-170k-wolof")
class Code170kWolof(ShareGPTConverter):
    task: ClassVar[TaskType] = TaskType.TOOL_USE
