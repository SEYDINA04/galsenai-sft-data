"""Converter NER : MasakhaNER 2.0 (wolof), annotations BIO au format tokens.

Seul jeu NER wolof annoté humainement à cette échelle. Le format source est du
*token classification* (une étiquette entière par token) : il faut le
**reconstruire en entités** pour produire une consigne d'instruction, sinon on
demanderait au modèle de générer une liste d'entiers alignée sur une
tokenisation qu'il ne voit pas.

La sortie est le même JSON que le converter WolofEntityLinking
(``[{"text": ..., "type": ...}]``), pour que les deux sources NER entraînent
exactement le même format de réponse.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from galsenai_sft.converters.base import BaseConverter
from galsenai_sft.converters.prompts import NER_TEMPLATES
from galsenai_sft.core.schema import Message, Role, Sample, TaskType
from galsenai_sft.registry import register

#: Étiquettes MasakhaNER 2.0, dans l'ordre officiel des entiers du dataset.
MASAKHANER2_LABELS: tuple[str, ...] = (
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-DATE",
    "I-DATE",
)


def spans_from_bio(tokens: list[str], tags: list[str]) -> list[dict[str, str]]:
    """Reconstruit les entités d'une séquence BIO.

    Tolérant aux séquences mal formées : un ``I-X`` orphelin (sans ``B-X``
    précédent) ouvre une entité au lieu d'être ignoré — les corpus annotés en
    contiennent, et les perdre reviendrait à apprendre au modèle à les oublier.
    """
    entities: list[dict[str, str]] = []
    cur_type: str | None = None
    cur_tokens: list[str] = []

    def flush() -> None:
        nonlocal cur_type, cur_tokens
        if cur_type and cur_tokens:
            entities.append({"text": " ".join(cur_tokens), "type": cur_type})
        cur_type, cur_tokens = None, []

    for token, tag in zip(tokens, tags, strict=False):
        if tag == "O" or not tag:
            flush()
            continue
        prefix, _, ent_type = tag.partition("-")
        if not ent_type:
            flush()
            continue
        if prefix == "B" or ent_type != cur_type:
            flush()
            cur_type = ent_type
            cur_tokens = [token]
        else:  # I- de la même entité : on prolonge
            cur_tokens.append(token)
    flush()
    return entities


@register("masakhane/masakhaner2")
class MasakhaNER2(BaseConverter):
    task: ClassVar[TaskType] = TaskType.NER

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        tokens = [str(t) for t in (row.get("tokens") or [])]
        raw_tags = row.get("ner_tags") or []
        if not tokens or not raw_tags:
            return []

        # Les tags arrivent en entiers (ClassLabel) ou déjà en chaînes.
        tags: list[str] = []
        for t in raw_tags:
            if isinstance(t, int):
                tags.append(MASAKHANER2_LABELS[t] if 0 <= t < len(MASAKHANER2_LABELS) else "O")
            else:
                tags.append(str(t))

        text = " ".join(tokens).strip()
        if not text:
            return []
        entities = spans_from_bio(tokens, tags)

        lang = self.pick_lang()
        user = self._rng.choice(NER_TEMPLATES[lang]).format(text=text)
        return [
            self.make_sample(
                [
                    Message(role=Role.USER, content=user),
                    Message(role=Role.ASSISTANT, content=json.dumps(entities, ensure_ascii=False)),
                ],
                prompt_lang=lang,
                n_entities=len(entities),
            )
        ]
