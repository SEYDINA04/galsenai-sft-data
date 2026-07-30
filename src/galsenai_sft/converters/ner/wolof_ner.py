"""Converter NER : WolofEntityLinking.

Chaque phrase -> liste d'entités ``[{"text", "type"}]`` en JSON (format
machine-vérifiable, cohérent avec la discipline JSON du tool use). Les *types*
d'entités restent en anglais (PER/ORG/LOC…), seule la consigne est localisée
(cf. stratégie du survey : localiser les consignes, pas les schémas).
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from galsenai_sft.converters.base import BaseConverter
from galsenai_sft.converters.prompts import NER_TEMPLATES
from galsenai_sft.core.schema import Message, Role, Sample, TaskType
from galsenai_sft.registry import register


@register("mbaye930/WolofEntityLinking")
class WolofEntityLinking(BaseConverter):
    task: ClassVar[TaskType] = TaskType.NER
    text_col: ClassVar[str] = "text"
    entities_col: ClassVar[str] = "entities"

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        text = str(row.get(self.text_col) or "").strip()
        raw_entities = row.get(self.entities_col) or []
        if not text:
            return []

        entities: list[dict[str, str]] = []
        for e in raw_entities:
            if not isinstance(e, dict):
                continue
            ent_text = str(e.get("text") or "").strip()
            ent_type = str(e.get("ner_type") or e.get("type") or "").strip()
            if ent_text and ent_type:
                entities.append({"text": ent_text, "type": ent_type})

        lang = self.pick_lang()
        user = self._rng.choice(NER_TEMPLATES[lang]).format(text=text)
        answer = json.dumps(entities, ensure_ascii=False)
        return [
            self.make_sample(
                [
                    Message(role=Role.USER, content=user),
                    Message(role=Role.ASSISTANT, content=answer),
                ],
                prompt_lang=lang,
                n_entities=len(entities),
            )
        ]
