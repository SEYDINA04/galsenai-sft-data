"""Gabarits de consignes bilingues (wolof / français).

Centralise les formulations d'instruction par tâche pour éviter toute logique
dataset-spécifique dispersée. Chaque tâche expose plusieurs variantes ; le
choix est délégué au converter (déterministe via seed).
"""

from __future__ import annotations

from galsenai_sft.core.schema import PromptLang

# Noms de langues affichés dans les consignes, par langue de consigne.
LANG_NAMES: dict[PromptLang, dict[str, str]] = {
    PromptLang.WO: {
        "wo": "wolof",
        "fr": "farañse",
        "en": "àngale",
        "ar": "araab",
    },
    PromptLang.FR: {
        "wo": "wolof",
        "fr": "français",
        "en": "anglais",
        "ar": "arabe",
    },
}

# Consignes de traduction : {src} et {tgt} = noms de langues déjà localisés.
TRANSLATION_TEMPLATES: dict[PromptLang, list[str]] = {
    PromptLang.WO: [
        "Tekkil lii ci {tgt}: {text}",
        "Tekkil mbind mii ci {tgt}: {text}",
        "Firi lii ci {tgt}: {text}",
    ],
    PromptLang.FR: [
        "Traduis ce texte en {tgt} : {text}",
        "Traduis en {tgt} : {text}",
        "Donne la traduction en {tgt} de : {text}",
    ],
}


def lang_name(prompt_lang: PromptLang, code: str) -> str:
    """Nom localisé d'une langue (code ISO court) dans la langue de consigne."""
    return LANG_NAMES[prompt_lang].get(code, code)
