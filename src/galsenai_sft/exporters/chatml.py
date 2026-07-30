"""Exporter ChatML : Sample -> dict ``{"messages": [{"role", "content", ...}]}``.

Format natif attendu par TRL / axolotl / LlamaFactory et les chat templates.
"""

from __future__ import annotations

from typing import Any

from galsenai_sft.core.schema import Role, Sample


def to_chatml(sample: Sample) -> dict[str, Any]:
    """Sérialise un Sample au format ChatML (messages avec rôles)."""
    messages: list[dict[str, Any]] = []
    for m in sample.messages:
        msg: dict[str, Any] = {"role": m.role.value, "content": m.content}
        if m.role is Role.TOOL and m.name:
            msg["name"] = m.name
        if m.tool_calls:
            msg["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls
            ]
        messages.append(msg)
    row: dict[str, Any] = {"messages": messages}
    # Métadonnées de traçabilité conservées à plat pour l'analyse.
    row["task"] = sample.task.value
    row["source"] = sample.source
    row["prompt_lang"] = sample.prompt_lang.value
    if sample.id:
        row["id"] = sample.id
    return row
