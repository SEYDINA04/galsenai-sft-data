"""Validation de la structure ChatML d'un Sample.

Le schéma pydantic (``core.schema``) garantit déjà l'essentiel à la construction.
Ce validator sert à :
  - **re-vérifier** des Samples chargés depuis un JSONL (données externes,
    éventuellement produites hors plateforme) sans faire crasher tout le lot ;
  - détecter des anomalies non couvertes par le schéma (UTF-8 invalide,
    échange user↔assistant incohérent, tool sans appel préalable).

Il n'utilise donc PAS uniquement la validation pydantic : il inspecte des dicts
bruts pour rapporter des :class:`Issue` au lieu de lever une exception.
"""

from __future__ import annotations

from typing import Any

from galsenai_sft.core.schema import Role, Sample
from galsenai_sft.validators.report import Issue, Severity

_VALID_ROLES = {r.value for r in Role}


def _has_valid_utf8(text: str) -> bool:
    try:
        text.encode("utf-8").decode("utf-8")
        return True
    except (UnicodeDecodeError, UnicodeEncodeError):
        return False


def validate_raw_chatml(row: dict[str, Any], index: int | None = None) -> list[Issue]:
    """Valide un dict ChatML brut (``{"messages": [...]}``) chargé de l'extérieur."""
    issues: list[Issue] = []

    def err(code: str, msg: str) -> None:
        issues.append(Issue(code=code, severity=Severity.ERROR, message=msg, sample_index=index))

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        err("empty_messages", "clé 'messages' absente ou vide")
        return issues

    roles: list[str] = []
    for j, m in enumerate(messages):
        if not isinstance(m, dict):
            err("bad_message", f"message #{j} n'est pas un objet")
            continue
        role = m.get("role")
        content = m.get("content", "")
        if role not in _VALID_ROLES:
            err("invalid_role", f"rôle invalide '{role}' au message #{j}")
        else:
            roles.append(role)
        has_tools = bool(m.get("tool_calls"))
        if not (content and str(content).strip()) and not has_tools:
            err("empty_content", f"message #{j} sans content ni tool_calls")
        if isinstance(content, str) and not _has_valid_utf8(content):
            err("invalid_utf8", f"UTF-8 invalide au message #{j}")

    if roles:
        if roles[-1] != Role.ASSISTANT.value:
            err("no_final_assistant", "la conversation ne se termine pas par 'assistant'")
        if Role.USER.value not in roles:
            err("no_user_turn", "aucun tour 'user'")
        if roles.count(Role.SYSTEM.value) > 1:
            err("multiple_system", "plusieurs messages 'system'")
        if Role.SYSTEM.value in roles and roles.index(Role.SYSTEM.value) != 0:
            err("system_not_first", "message 'system' hors première position")

    return issues


def validate_sample_structure(sample: Sample, index: int | None = None) -> list[Issue]:
    """Vérifie des invariants supplémentaires sur un Sample déjà typé."""
    issues: list[Issue] = []
    for j, m in enumerate(sample.messages):
        if m.content and not _has_valid_utf8(m.content):
            issues.append(
                Issue(
                    code="invalid_utf8",
                    severity=Severity.ERROR,
                    message=f"UTF-8 invalide au message #{j}",
                    sample_index=index,
                    source=sample.source,
                )
            )
    return issues
