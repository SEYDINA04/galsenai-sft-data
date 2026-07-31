"""Converters de *function calling* réel (capacité agentique).

Comblent le trou le plus visible du dataset : la tâche ``tool_use`` ne
contenait jusqu'ici aucun appel d'outil — uniquement du questions/réponses de
code. Ces quatre sources apportent de vrais appels, dans quatre formats
différents :

===================================  ==========================================
Source                               Format d'origine
===================================  ==========================================
``Agent-Ark/Toucan-1.5M``            ``messages`` (JSON) avec rôles
                                     ``tool_call`` / ``tool_response``
``NousResearch/hermes-…``            ShareGPT + balises ``<tool_call>`` XML
``Team-ACE/ToolACE``                 ShareGPT + DSL ``[Nom(arg="v")]``
``nvidia/When2Call``                 ``messages`` OpenAI + ``tools``
===================================  ==========================================

Tous convergent vers la même représentation canonique : un message ``system``
qui déclare les outils disponibles, des ``tool_calls`` typés sur les tours
assistant, et des messages de rôle ``tool`` pour les résultats.

Ces jeux sont **anglophones** : ils n'apportent pas de wolof, mais la capacité
d'appeler un outil, que le transfert multilingue propage. Ils sont donc
étiquetés ``prompt_lang=en``, et jamais comptés comme de la donnée wolof.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, ClassVar

from galsenai_sft.converters.base import BaseConverter
from galsenai_sft.core.schema import Message, PromptLang, Role, Sample, TaskType, ToolCall
from galsenai_sft.registry import register

_SYSTEM_PREFIX = (
    "You are a helpful assistant with access to the following tools. "
    "Call a tool when it is needed to answer the request; otherwise answer directly.\n"
)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
#: DSL ToolACE : ``[Nom De Fonction(arg="v", n=1), Autre()]``
_TOOLACE_CALL_RE = re.compile(r"([A-Za-z_][\w .-]*)\((.*?)\)\s*(?:,|$)", re.DOTALL)
#: Noms de fonctions déclarés dans un schéma JSON noyé dans de la prose.
_NAME_FIELD_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


# ════════════════════════════════════════════════════════════════════
#  Parsing tolérant
# ════════════════════════════════════════════════════════════════════
def loads_loose(raw: Any) -> Any:
    """Parse du JSON, ou une repr Python, ou rien.

    Les jeux de function calling mélangent les deux sérialisations (Toucan
    stocke des dicts en repr Python, avec ``arguments`` lui-même en JSON).
    ``literal_eval`` n'exécute aucun code : il refuse tout ce qui n'est pas un
    littéral.
    """
    if isinstance(raw, dict | list):
        return raw
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        pass
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return None


def to_tool_call(raw: Any) -> ToolCall | None:
    """Construit un :class:`ToolCall` depuis un dict ``{name, arguments}``."""
    data = loads_loose(raw)
    if not isinstance(data, dict):
        return None
    name = str(data.get("name") or "").strip()
    if not name:
        return None
    args = data.get("arguments", data.get("parameters", {}))
    # `arguments` est fréquemment une chaîne JSON imbriquée.
    if isinstance(args, str):
        args = loads_loose(args)
    if not isinstance(args, dict):
        args = {}
    return ToolCall(name=name, arguments=args)


def tool_names(raw_tools: Any) -> list[str]:
    """Extrait les noms d'outils déclarés, quel que soit l'emballage."""
    tools = loads_loose(raw_tools)
    if isinstance(tools, dict):
        tools = [tools]
    if not isinstance(tools, list):
        return []
    names: list[str] = []
    for t in tools:
        item = loads_loose(t) if isinstance(t, str) else t
        if not isinstance(item, dict):
            continue
        # Format OpenAI ({"type": "function", "function": {...}}) ou plat.
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        if name := str(fn.get("name") or "").strip():
            names.append(name)
    return names


def system_message(raw_tools: Any) -> Message | None:
    """Message système déclarant les outils disponibles (compact et lisible).

    On ne recopie pas les schémas JSON complets : ils font souvent plusieurs
    kilo-octets par exemple, ce qui multiplierait par dix la taille du dataset
    sans apporter de signal supplémentaire sur *quand* appeler un outil.
    """
    names = tool_names(raw_tools)
    if not names:
        return None
    listed = "\n".join(f"- {n}" for n in dict.fromkeys(names))
    return Message(role=Role.SYSTEM, content=_SYSTEM_PREFIX + listed)


# ════════════════════════════════════════════════════════════════════
#  Base commune
# ════════════════════════════════════════════════════════════════════
class FunctionCallingConverter(BaseConverter):
    """Squelette commun : construit le système, délègue l'extraction des tours."""

    task: ClassVar[TaskType] = TaskType.TOOL_USE
    prompt_langs: ClassVar[tuple[PromptLang, ...]] = (PromptLang.EN,)
    tools_col: ClassVar[str] = "tools"

    def extract_turns(self, row: dict[str, Any]) -> list[Message]:
        """À implémenter : les tours de la conversation, hors système."""
        raise NotImplementedError

    def raw_tools(self, row: dict[str, Any]) -> Any:
        return row.get(self.tools_col)

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        try:
            turns = self.extract_turns(row)
        except (ValueError, TypeError, KeyError):
            return []
        if not turns:
            return []

        messages: list[Message] = []
        if (sys_msg := system_message(self.raw_tools(row))) is not None:
            messages.append(sys_msg)
        messages.extend(turns)

        # Un Sample valide contient au moins un `user` et finit par `assistant`.
        while messages and messages[-1].role is not Role.ASSISTANT:
            messages.pop()
        if not any(m.role is Role.USER for m in messages):
            return []
        if not any(m.role is Role.ASSISTANT for m in messages):
            return []

        n_calls = sum(len(m.tool_calls or []) for m in messages)
        return [
            self.make_sample(
                messages,
                prompt_lang=PromptLang.EN,
                n_turns=len(messages),
                n_tool_calls=n_calls,
            )
        ]


# ════════════════════════════════════════════════════════════════════
#  Toucan — rôles tool_call / tool_response explicites
# ════════════════════════════════════════════════════════════════════
@register("Agent-Ark/Toucan-1.5M")
class Toucan(FunctionCallingConverter):
    """Trajectoires multi-tours réellement exécutées (serveurs MCP)."""

    def extract_turns(self, row: dict[str, Any]) -> list[Message]:
        raw = loads_loose(row.get("messages"))
        if not isinstance(raw, list):
            return []

        out: list[Message] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            content = item.get("content")

            if role == "tool_call":
                call = to_tool_call(content)
                if call is None:
                    continue
                # Appels consécutifs : on les regroupe sur le même tour
                # assistant (appels parallèles), comme le fait l'inférence.
                if out and out[-1].role is Role.ASSISTANT and out[-1].tool_calls:
                    out[-1].tool_calls.append(call)
                else:
                    out.append(Message(role=Role.ASSISTANT, tool_calls=[call]))
            elif role == "tool_response":
                text = str(content or "").strip()
                if text:
                    out.append(Message(role=Role.TOOL, content=text))
            elif role in {"user", "assistant"}:
                text = str(content or "").strip()
                if text:
                    out.append(
                        Message(role=Role.USER if role == "user" else Role.ASSISTANT, content=text)
                    )
        return out


# ════════════════════════════════════════════════════════════════════
#  Hermes — ShareGPT + balises <tool_call> / <tool_response>
# ════════════════════════════════════════════════════════════════════
@register("NousResearch/hermes-function-calling-v1")
class HermesFunctionCalling(FunctionCallingConverter):
    """Format de référence de l'écosystème (balises XML dans le texte)."""

    def raw_tools(self, row: dict[str, Any]) -> Any:
        return row.get("tools")

    def extract_turns(self, row: dict[str, Any]) -> list[Message]:
        turns = row.get("conversations") or []
        out: list[Message] = []
        for t in turns:
            if not isinstance(t, dict):
                continue
            frm = str(t.get("from") or "").lower()
            value = str(t.get("value") or "").strip()
            if not value:
                continue

            if frm == "system":
                continue  # remplacé par notre système normalisé
            if frm == "tool":
                # Le résultat est encapsulé ; on garde le contenu utile.
                inner = _TOOL_RESPONSE_RE.findall(value)
                out.append(Message(role=Role.TOOL, content="\n".join(inner) if inner else value))
            elif frm in {"human", "user"}:
                out.append(Message(role=Role.USER, content=value))
            elif frm in {"gpt", "assistant"}:
                calls = [c for raw in _TOOL_CALL_RE.findall(value) if (c := to_tool_call(raw))]
                text = _TOOL_CALL_RE.sub("", value).strip()
                if calls:
                    out.append(Message(role=Role.ASSISTANT, content=text, tool_calls=calls))
                elif text:
                    out.append(Message(role=Role.ASSISTANT, content=text))
        return out


# ════════════════════════════════════════════════════════════════════
#  ToolACE — DSL [Nom(arg="v")]
# ════════════════════════════════════════════════════════════════════
def parse_toolace_calls(value: str) -> list[ToolCall]:
    """Parse le DSL ToolACE ``[Nom(arg="v", n=1), Autre()]``.

    Les valeurs sont des littéraux Python : ``literal_eval`` les lit sans rien
    exécuter. Un argument illisible est ignoré plutôt que de faire perdre
    l'appel entier.
    """
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return []
    calls: list[ToolCall] = []
    for name, arg_str in _TOOLACE_CALL_RE.findall(text[1:-1]):
        name = name.strip()
        if not name:
            continue
        args: dict[str, Any] = {}
        for part in re.split(r",\s*(?=[A-Za-z_]\w*\s*=)", arg_str):
            key, sep, raw_val = part.partition("=")
            if not sep:
                continue
            try:
                args[key.strip()] = ast.literal_eval(raw_val.strip())
            except (ValueError, SyntaxError):
                args[key.strip()] = raw_val.strip()
        calls.append(ToolCall(name=name, arguments=args))
    return calls


@register("Team-ACE/ToolACE")
class ToolACE(FunctionCallingConverter):
    """Grande diversité d'API ; les outils sont décrits dans le champ system."""

    def raw_tools(self, row: dict[str, Any]) -> Any:
        # Le champ system mêle prose et JSON, sans délimiteur fiable : extraire
        # la sous-chaîne « du premier [ au dernier ] » échoue dès qu'un schéma
        # contient un crochet. On lit donc directement les noms déclarés.
        text = str(row.get("system") or "")
        return [{"name": n} for n in _NAME_FIELD_RE.findall(text)]

    def extract_turns(self, row: dict[str, Any]) -> list[Message]:
        turns = row.get("conversations") or []
        out: list[Message] = []
        for t in turns:
            if not isinstance(t, dict):
                continue
            frm = str(t.get("from") or "").lower()
            value = str(t.get("value") or "").strip()
            if not value:
                continue
            if frm in {"human", "user"}:
                out.append(Message(role=Role.USER, content=value))
            elif frm == "tool":
                out.append(Message(role=Role.TOOL, content=value))
            elif frm in {"gpt", "assistant"}:
                if calls := parse_toolace_calls(value):
                    out.append(Message(role=Role.ASSISTANT, tool_calls=calls))
                else:
                    out.append(Message(role=Role.ASSISTANT, content=value))
        return out


# ════════════════════════════════════════════════════════════════════
#  When2Call — décision d'appel / abstention
# ════════════════════════════════════════════════════════════════════
@register("nvidia/When2Call")
class When2Call(FunctionCallingConverter):
    """Apprend **quand ne pas** appeler d'outil — contre-poids au sur-appel.

    Une bonne partie de ses exemples est justement une réponse directe alors
    que des outils sont disponibles : c'est le signal recherché.
    """

    def extract_turns(self, row: dict[str, Any]) -> list[Message]:
        raw = row.get("messages") or []
        if isinstance(raw, str):
            raw = loads_loose(raw) or []
        out: list[Message] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").lower()
            content = str(item.get("content") or "").strip()
            if role == "system":
                continue
            if role == "user" and content:
                out.append(Message(role=Role.USER, content=content))
            elif role == "tool" and content:
                out.append(Message(role=Role.TOOL, content=content))
            elif role == "assistant":
                # Un contenu qui est un appel sérialisé devient un vrai ToolCall.
                if (call := to_tool_call(content)) is not None:
                    out.append(Message(role=Role.ASSISTANT, tool_calls=[call]))
                elif content:
                    out.append(Message(role=Role.ASSISTANT, content=content))
        return out
