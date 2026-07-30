"""Schéma canonique interne.

Toute donnée SFT — quelle que soit la tâche (NER, traduction, intent, QA, tool
use…) — est représentée par un :class:`Sample` : une conversation ChatML typée.
C'est la **seule** représentation de vérité ; les formats de sortie (ChatML,
Alpaca, ShareGPT) en sont des sérialisations produites par les exporters.

Choix de conception :
- On ne stocke **jamais** du ChatML brut (texte) : on manipule des objets typés
  (pydantic) validables, transformables et testables.
- Le champ ``tool_calls`` permet de représenter l'usage d'outils (agentique)
  sans changer de structure — même schéma pour toutes les capacités.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Role(StrEnum):
    """Rôles ChatML autorisés dans une conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TaskType(StrEnum):
    """Familles de tâches SFT ciblées (extensible)."""

    NER = "ner"
    TRANSLATION = "translation"
    INTENT = "intent"
    QA = "qa"
    RETRIEVAL = "retrieval"
    CLASSIFICATION = "classification"
    TOOL_USE = "tool_use"
    INSTRUCTION = "instruction"
    OTHER = "other"


class PromptLang(StrEnum):
    """Langue de la consigne (le contenu cible reste du wolof)."""

    WO = "wo"
    FR = "fr"


class ToolCall(BaseModel):
    """Appel d'outil émis par l'assistant (capacité agentique / tool use)."""

    name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """Un tour de conversation ChatML."""

    role: Role
    content: str = ""
    # Renseigné uniquement pour les tours assistant qui appellent un outil.
    tool_calls: list[ToolCall] | None = None
    # Nom de l'outil pour un message de rôle ``tool`` (résultat d'exécution).
    name: str | None = None

    @model_validator(mode="after")
    def _check_content_or_tool_calls(self) -> Message:
        has_content = bool(self.content and self.content.strip())
        has_tools = bool(self.tool_calls)
        if not has_content and not has_tools:
            raise ValueError("un message doit avoir un content non vide ou des tool_calls")
        if self.name is not None and self.role is not Role.TOOL:
            raise ValueError("le champ 'name' n'est valable que pour un message de rôle 'tool'")
        return self


class Sample(BaseModel):
    """Un exemple SFT canonique : une conversation + ses métadonnées de traçabilité."""

    messages: list[Message] = Field(..., min_length=1)
    task: TaskType
    source: str = Field(
        ..., min_length=1, description="repo_id ou identifiant du dataset d'origine"
    )
    prompt_lang: PromptLang = PromptLang.WO
    id: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_role_ordering(self) -> Sample:
        """Garantit une structure ChatML valide.

        Règles :
        - un éventuel ``system`` est en première position uniquement ;
        - hors system, les rôles alternent user/assistant (tool inséré après
          un appel d'outil de l'assistant) ;
        - la conversation se termine par un tour ``assistant`` (la cible SFT).
        """
        roles = [m.role for m in self.messages]

        if Role.SYSTEM in roles and roles.index(Role.SYSTEM) != 0:
            raise ValueError("le message 'system' doit être en première position")
        if roles.count(Role.SYSTEM) > 1:
            raise ValueError("un seul message 'system' est autorisé")

        if roles[-1] is not Role.ASSISTANT:
            raise ValueError("la conversation doit se terminer par un tour 'assistant'")

        # Au moins un échange user -> assistant.
        if Role.USER not in roles:
            raise ValueError("la conversation doit contenir au moins un tour 'user'")

        return self

    def n_turns(self) -> int:
        """Nombre de tours utilisateur (approx. de la longueur du dialogue)."""
        return sum(1 for m in self.messages if m.role is Role.USER)
