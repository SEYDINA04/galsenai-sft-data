"""Exporter Alpaca : Sample -> ``{"instruction", "input", "output"}``.

Adapté aux conversations mono-tour (1 user -> 1 assistant). Un ``system``
éventuel est préfixé à l'instruction. Les dialogues multi-tours ne sont pas
représentables en Alpaca : ``to_alpaca`` lève alors ``ValueError`` (à filtrer
en amont selon le besoin).
"""

from __future__ import annotations

from typing import Any

from galsenai_sft.core.schema import Role, Sample


def is_alpaca_compatible(sample: Sample) -> bool:
    """Vrai si le Sample est mono-tour (exportable en Alpaca)."""
    non_system = [m for m in sample.messages if m.role is not Role.SYSTEM]
    return (
        len(non_system) == 2
        and non_system[0].role is Role.USER
        and non_system[1].role is Role.ASSISTANT
        and not non_system[1].tool_calls
    )


def to_alpaca(sample: Sample) -> dict[str, Any]:
    """Sérialise un Sample mono-tour au format Alpaca."""
    if not is_alpaca_compatible(sample):
        raise ValueError(
            f"Sample non exportable en Alpaca (multi-tours ou tool use) : source={sample.source}"
        )
    system = next((m.content for m in sample.messages if m.role is Role.SYSTEM), "")
    user = next(m.content for m in sample.messages if m.role is Role.USER)
    assistant = next(m.content for m in sample.messages if m.role is Role.ASSISTANT)

    instruction = f"{system}\n\n{user}".strip() if system else user
    return {
        "instruction": instruction,
        "input": "",
        "output": assistant,
        "task": sample.task.value,
        "source": sample.source,
    }
