"""Converters d'instructions : format Alpaca et collection Aya.

Complètent WORI (reverse instructions) par les deux autres familles de données
d'instruction wolof disponibles :

``AlpacaConverter``
    format ``instruction`` / ``input`` / ``output``. Le champ ``input`` est un
    **contexte facultatif** : quand il est présent, il est concaténé à la
    consigne plutôt que perdu (le format ChatML n'a pas de champ dédié).

``AyaCollection``
    paires ``inputs`` / ``targets`` de la collection Aya (sous-ensemble wolof),
    agrégat de plusieurs jeux annotés dont la consigne est déjà rédigée.

Ces sources sont issues de traduction automatique pour partie : le filtre LID
de la cible wolof est activé pour elles dans ``configs/build.yaml``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from galsenai_sft.converters.base import BaseConverter
from galsenai_sft.core.schema import Message, PromptLang, Role, Sample, TaskType
from galsenai_sft.registry import register


class AlpacaConverter(BaseConverter):
    """Convertit une ligne Alpaca (instruction/input/output) en Sample ChatML."""

    task: ClassVar[TaskType] = TaskType.INSTRUCTION
    instruction_col: ClassVar[str] = "instruction"
    input_col: ClassVar[str] = "input"
    output_col: ClassVar[str] = "output"
    #: consigne fournie par le dataset (wolof) : aucun gabarit n'est ajouté.
    prompt_langs: ClassVar[tuple[PromptLang, ...]] = (PromptLang.WO,)

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        instruction = str(row.get(self.instruction_col) or "").strip()
        context = str(row.get(self.input_col) or "").strip()
        output = str(row.get(self.output_col) or "").strip()
        if not instruction or not output:
            return []

        # Le contexte n'a pas de place propre en ChatML : on l'attache à la
        # consigne (perdre `input` changerait le sens de plusieurs exemples).
        user = f"{instruction}\n\n{context}" if context else instruction
        return [
            self.make_sample(
                [
                    Message(role=Role.USER, content=user),
                    Message(role=Role.ASSISTANT, content=output),
                ],
                prompt_lang=PromptLang.WO,
                has_context=bool(context),
            )
        ]


@register("ngia/alpaca-data-in-wolof")
class AlpacaWolof(AlpacaConverter):
    """Alpaca traduit en wolof (traduction automatique — LID activé au build)."""


@register("bilalfaye/wolof-sft")
class BilalfayeWolofSFT(AlpacaConverter):
    """Jeu SFT wolof au format Alpaca."""


@register("CohereLabs/aya_collection_language_split")
class AyaCollectionWolof(BaseConverter):
    """Sous-ensemble wolof de la collection Aya (``inputs`` -> ``targets``)."""

    task: ClassVar[TaskType] = TaskType.INSTRUCTION
    prompt_langs: ClassVar[tuple[PromptLang, ...]] = (PromptLang.WO,)

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        inputs = str(row.get("inputs") or "").strip()
        targets = str(row.get("targets") or "").strip()
        if not inputs or not targets:
            return []
        return [
            self.make_sample(
                [
                    Message(role=Role.USER, content=inputs),
                    Message(role=Role.ASSISTANT, content=targets),
                ],
                prompt_lang=PromptLang.WO,
                # Aya agrège plusieurs jeux : on garde la provenance fine.
                aya_dataset=str(row.get("dataset_name") or ""),
                aya_task=str(row.get("task_type") or ""),
            )
        ]
