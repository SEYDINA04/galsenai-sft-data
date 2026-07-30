"""Converter d'instructions natives : WORI (reverse instructions).

WORI fournit un texte wolof natif (``text_wo``) et une instruction générée
(``instruction_wo``). La paire instruction -> texte constitue directement un
exemple SFT sans translationese (l'output est du wolof natif).
"""

from __future__ import annotations

from typing import Any, ClassVar

from galsenai_sft.converters.base import BaseConverter
from galsenai_sft.core.schema import Message, PromptLang, Role, Sample, TaskType
from galsenai_sft.registry import register


@register("m-a-d-i/wori-wolof-instructions")
class WoriInstructions(BaseConverter):
    task: ClassVar[TaskType] = TaskType.INSTRUCTION
    # Consignes déjà fournies dans les deux langues par le dataset.
    prompt_langs: ClassVar[tuple[PromptLang, ...]] = (PromptLang.WO, PromptLang.FR)

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        text_wo = str(row.get("text_wo") or "").strip()
        instr_wo = str(row.get("instruction_wo") or "").strip()
        instr_fr = str(row.get("instruction_fr") or "").strip()
        if not text_wo:
            return []

        out: list[Sample] = []
        # Consigne wolof -> réponse wolof
        if instr_wo:
            out.append(
                self.make_sample(
                    [
                        Message(role=Role.USER, content=instr_wo),
                        Message(role=Role.ASSISTANT, content=text_wo),
                    ],
                    prompt_lang=PromptLang.WO,
                )
            )
        # Consigne française -> réponse wolof (transfert bilingue)
        if instr_fr:
            out.append(
                self.make_sample(
                    [
                        Message(role=Role.USER, content=instr_fr),
                        Message(role=Role.ASSISTANT, content=text_wo),
                    ],
                    prompt_lang=PromptLang.FR,
                )
            )
        return out
