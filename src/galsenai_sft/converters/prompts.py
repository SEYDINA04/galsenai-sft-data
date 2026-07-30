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


# --- Intent classification : {text} = énoncé utilisateur -------------------
INTENT_TEMPLATES: dict[PromptLang, list[str]] = {
    PromptLang.WO: [
        "Lan mooy jubluwaayu kàddu gii? {text}",
        "Wan jubluwaay la kàddu gii wund? {text}",
    ],
    PromptLang.FR: [
        "Quelle est l'intention de cet énoncé ? {text}",
        "Classe l'intention de la phrase suivante : {text}",
    ],
}

# --- Slot filling : extraire les entités/valeurs d'un énoncé ----------------
SLOT_TEMPLATES: dict[PromptLang, list[str]] = {
    PromptLang.WO: [
        "Génne slots yi ci kàddu gii: {text}",
    ],
    PromptLang.FR: [
        "Extrais les slots (entités) de cet énoncé : {text}",
    ],
}

# --- NER : {text} = phrase -------------------------------------------------
NER_TEMPLATES: dict[PromptLang, list[str]] = {
    PromptLang.WO: [
        "Génne tur yu am solo (PER, ORG, LOC…) ci mbind mii: {text}",
        "Wone entités yi nekk ci kàddu gii: {text}",
    ],
    PromptLang.FR: [
        "Extrais les entités nommées (PER, ORG, LOC…) du texte : {text}",
        "Identifie les entités nommées de la phrase : {text}",
    ],
}

# --- QA : {question} -------------------------------------------------------
QA_TEMPLATES: dict[PromptLang, list[str]] = {
    PromptLang.WO: [
        "Tontul laaj bii: {question}",
        "Jox tontu laaj bii: {question}",
    ],
    PromptLang.FR: [
        "Réponds à la question : {question}",
        "Donne la réponse à : {question}",
    ],
}

# --- Classification thématique/sentiment : {text}, {task_desc}, {labels} ----
CLASSIFY_TEMPLATES: dict[PromptLang, list[str]] = {
    PromptLang.WO: [
        "{task_desc} ci mbind mii: {text}",
    ],
    PromptLang.FR: [
        "{task_desc} : {text}",
    ],
}

# Descriptions de tâche de classification (par langue de consigne).
CLASSIFY_TASKS: dict[str, dict[PromptLang, str]] = {
    "sentiment": {
        PromptLang.WO: "Wan xalaat la (baax / bon / digg-dóomu)?",
        PromptLang.FR: "Quel est le sentiment (positif / négatif / neutre)",
    },
    "topic": {
        PromptLang.WO: "Ci wan wàll la mbind mii bokk?",
        PromptLang.FR: "Quel est le thème du texte",
    },
    "emotion": {
        PromptLang.WO: "Wan yëg-yëg la kàddu gii wund?",
        PromptLang.FR: "Quelle émotion exprime ce texte",
    },
    "intent": {
        PromptLang.WO: "Lan mooy jubluwaayu kàddu gii?",
        PromptLang.FR: "Quelle est l'intention de cet énoncé",
    },
}
