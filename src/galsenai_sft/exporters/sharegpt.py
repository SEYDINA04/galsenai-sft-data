"""Exporter ShareGPT : Sample -> ``{"conversations": [{"from", "value"}]}``.

Format historique (celui de nombreux datasets, ex. Code-170k). Mapping des
rôles : system->system, user->human, assistant->gpt, tool->tool.
"""

from __future__ import annotations

from typing import Any

from galsenai_sft.core.schema import Role, Sample

_ROLE_TO_FROM = {
    Role.SYSTEM: "system",
    Role.USER: "human",
    Role.ASSISTANT: "gpt",
    Role.TOOL: "tool",
}


def to_sharegpt(sample: Sample) -> dict[str, Any]:
    """Sérialise un Sample au format ShareGPT (``conversations``)."""
    conversations = []
    for m in sample.messages:
        value = m.content
        if m.tool_calls:
            # Représentation textuelle simple des appels d'outil pour ShareGPT.
            calls = "; ".join(f"{tc.name}({tc.arguments})" for tc in m.tool_calls)
            value = (value + "\n" + calls).strip() if value else calls
        conversations.append({"from": _ROLE_TO_FROM[m.role], "value": value})
    return {
        "conversations": conversations,
        "task": sample.task.value,
        "source": sample.source,
    }
