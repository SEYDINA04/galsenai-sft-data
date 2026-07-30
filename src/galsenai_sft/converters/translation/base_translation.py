"""Converter de traduction générique (piloté par configuration).

Sert de **patron** pour la famille traduction : au lieu d'un fichier par
dataset parallèle, on paramètre un converter unique par les noms de colonnes
source/cible et les codes de langue. Un dataset concret l'enregistre via une
sous-classe minimale décorée par ``@register`` (voir ``galsenai_translation``).

Il produit, pour une paire (texte_source, texte_cible), des Samples
d'instruction bilingues :
  - user : « Traduis en {cible} : {source} »  (consigne wo ou fr)
  - assistant : {cible}

Par défaut, la direction **vers le wolof** est privilégiée (le wolof est
toujours la sortie), ce qui correspond à l'objectif d'un LLM wolof. La
direction inverse (wolof -> autre) est optionnelle via ``bidirectional``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from galsenai_sft.converters.base import BaseConverter
from galsenai_sft.converters.prompts import TRANSLATION_TEMPLATES, lang_name
from galsenai_sft.core.schema import Message, Role, Sample, TaskType


class TranslationConverter(BaseConverter):
    """Converter paramétrable pour un corpus parallèle à 2 colonnes texte."""

    task: ClassVar[TaskType] = TaskType.TRANSLATION

    #: nom de colonne + code langue du texte source (ex. ("fr", "fra"))
    source_col: ClassVar[str] = ""
    source_lang: ClassVar[str] = ""
    #: nom de colonne + code langue du texte cible (typiquement le wolof)
    target_col: ClassVar[str] = ""
    target_lang: ClassVar[str] = "wo"
    #: si vrai, génère aussi la direction inverse (cible -> source)
    bidirectional: ClassVar[bool] = False

    def _one_direction(
        self, src_text: str, tgt_text: str, src_code: str, tgt_code: str
    ) -> Sample | None:
        src_text = (src_text or "").strip()
        tgt_text = (tgt_text or "").strip()
        if not src_text or not tgt_text:
            return None
        lang = self.pick_lang()
        template = self._rng.choice(TRANSLATION_TEMPLATES[lang])
        tgt_name = lang_name(lang, tgt_code)
        user = template.format(tgt=tgt_name, text=src_text)
        messages = [
            Message(role=Role.USER, content=user),
            Message(role=Role.ASSISTANT, content=tgt_text),
        ]
        return self.make_sample(
            messages,
            prompt_lang=lang,
            direction=f"{src_code}->{tgt_code}",
        )

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        src = row.get(self.source_col)
        tgt = row.get(self.target_col)
        out: list[Sample] = []

        # Direction principale : source -> wolof (cible).
        s = self._one_direction(src, tgt, self.source_lang, self.target_lang)
        if s is not None:
            out.append(s)

        # Direction inverse optionnelle : wolof -> source.
        if self.bidirectional:
            s2 = self._one_direction(tgt, src, self.target_lang, self.source_lang)
            if s2 is not None:
                out.append(s2)

        return out
