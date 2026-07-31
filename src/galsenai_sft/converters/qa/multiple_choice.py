"""Converters QA à choix multiples et problèmes de mathématiques.

Trois sources africaines couvrant le wolof, qui font passer la tâche QA d'une
source unique à quatre :

``Belebele``
    compréhension de lecture : passage FLORES + question + 4 options.
``AfriMMLU``
    connaissances scolaires : question + 4 options, sans passage.
``AfriMGSM``
    problèmes de mathématiques rédigés, réponse numérique.

Choix de conception : la réponse attendue est **la lettre** (`A`–`D`), pas le
texte de l'option. C'est vérifiable automatiquement, indépendant de la
formulation, et cohérent avec la façon dont ces jeux sont évalués.

Ces jeux sont d'abord des **jeux d'évaluation** : leurs splits d'origine sont
`test`/`validation`. Les intégrer à l'entraînement retire la possibilité de
s'en servir comme mesure — c'est un arbitrage assumé au profit du volume, noté
dans la data card.
"""

from __future__ import annotations

import ast
from typing import Any, ClassVar

from galsenai_sft.converters.base import BaseConverter
from galsenai_sft.converters.prompts import (
    MATH_TEMPLATES,
    MCQ_PASSAGE_TEMPLATES,
    MCQ_TEMPLATES,
)
from galsenai_sft.core.schema import Message, Role, Sample, TaskType
from galsenai_sft.registry import register

_LETTERS = ("A", "B", "C", "D", "E", "F")


def format_choices(options: list[str]) -> str:
    """Met en forme les options : ``A. …`` une par ligne."""
    return "\n".join(f"{_LETTERS[i]}. {opt}" for i, opt in enumerate(options) if i < len(_LETTERS))


def parse_choices(raw: Any) -> list[str]:
    """Lit une liste d'options, qu'elle soit déjà une liste ou une chaîne Python.

    AfriMMLU sérialise ses options en chaîne (``"['a', 'b']"``) : la parser à la
    main serait fragile, ``literal_eval`` refuse tout ce qui n'est pas un
    littéral (pas d'exécution de code).
    """
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw or "").strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            return []
    return []


@register("facebook/belebele")
class Belebele(BaseConverter):
    """Compréhension de lecture : passage + question + 4 options."""

    task: ClassVar[TaskType] = TaskType.QA

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        passage = str(row.get("flores_passage") or "").strip()
        question = str(row.get("question") or "").strip()
        options = [str(row.get(f"mc_answer{i}") or "").strip() for i in range(1, 5)]
        if not passage or not question or not all(options):
            return []
        try:
            correct = int(row.get("correct_answer_num"))
        except (TypeError, ValueError):
            return []
        if not 1 <= correct <= len(options):
            return []

        lang = self.pick_lang()
        user = self._rng.choice(MCQ_PASSAGE_TEMPLATES[lang]).format(
            passage=passage, question=question, choices=format_choices(options)
        )
        return [
            self.make_sample(
                [
                    Message(role=Role.USER, content=user),
                    Message(role=Role.ASSISTANT, content=_LETTERS[correct - 1]),
                ],
                prompt_lang=lang,
                subtask="reading_comprehension",
            )
        ]


@register("masakhane/afrimmlu")
class AfriMMLU(BaseConverter):
    """Connaissances scolaires à choix multiples (réponse déjà en lettre)."""

    task: ClassVar[TaskType] = TaskType.QA

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        question = str(row.get("question") or "").strip()
        options = parse_choices(row.get("choices"))
        answer = str(row.get("answer") or "").strip().upper()
        if not question or len(options) < 2 or answer not in _LETTERS[: len(options)]:
            return []

        lang = self.pick_lang()
        user = self._rng.choice(MCQ_TEMPLATES[lang]).format(
            question=question, choices=format_choices(options)
        )
        return [
            self.make_sample(
                [
                    Message(role=Role.USER, content=user),
                    Message(role=Role.ASSISTANT, content=answer),
                ],
                prompt_lang=lang,
                subject=str(row.get("subject") or ""),
            )
        ]


@register("masakhane/afrimgsm")
class AfriMGSM(BaseConverter):
    """Problèmes de mathématiques rédigés ; la réponse est un nombre."""

    task: ClassVar[TaskType] = TaskType.QA

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        question = str(row.get("question") or "").strip()
        # `answer` est souvent nul dans les splits traduits : `answer_number`
        # est le seul champ systématiquement renseigné.
        number = row.get("answer_number")
        if not question or number is None:
            return []

        lang = self.pick_lang()
        user = self._rng.choice(MATH_TEMPLATES[lang]).format(question=question)
        return [
            self.make_sample(
                [
                    Message(role=Role.USER, content=user),
                    Message(role=Role.ASSISTANT, content=str(number).strip()),
                ],
                prompt_lang=lang,
                subtask="math_word_problem",
            )
        ]
