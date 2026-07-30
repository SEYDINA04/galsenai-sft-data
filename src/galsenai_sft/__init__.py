"""galsenai_sft — Plateforme de préparation de datasets d'instruction (SFT) en wolof.

Représentation canonique interne : :class:`galsenai_sft.core.schema.Sample`
(sérialisable en ChatML, Alpaca, ShareGPT via les exporters dédiés).
"""

from __future__ import annotations

__version__ = "0.1.0"

from galsenai_sft.core.schema import Message, Role, Sample, TaskType, ToolCall

__all__ = ["Message", "Role", "Sample", "TaskType", "ToolCall", "__version__"]
