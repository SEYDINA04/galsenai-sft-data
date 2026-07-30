"""Système de plugins : enregistrement et découverte des converters.

Principe **Open/Closed** : ajouter un dataset = écrire UN converter décoré par
``@register(...)`` dans ``galsenai_sft/converters/<tâche>/``. Aucune modification
du pipeline existant n'est nécessaire ; la découverte est automatique.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

from galsenai_sft.core.logging import get_logger

if TYPE_CHECKING:
    from galsenai_sft.converters.base import BaseConverter

log = get_logger(__name__)

# Registre global : identifiant de dataset -> classe de converter.
_REGISTRY: dict[str, type[BaseConverter]] = {}
_DISCOVERED = False


def register(dataset_id: str):
    """Décorateur : enregistre une classe de converter pour un ``dataset_id``.

    Exemple::

        @register("galsenai/french-wolof-translation")
        class FrenchWolofTranslation(BaseConverter):
            ...
    """

    def _wrap(cls: type[BaseConverter]) -> type[BaseConverter]:
        if dataset_id in _REGISTRY:
            raise ValueError(f"converter déjà enregistré pour '{dataset_id}'")
        cls.dataset_id = dataset_id
        _REGISTRY[dataset_id] = cls
        return cls

    return _wrap


def discover(force: bool = False) -> None:
    """Importe récursivement ``galsenai_sft.converters`` pour déclencher les ``@register``."""
    global _DISCOVERED
    if _DISCOVERED and not force:
        return
    import galsenai_sft.converters as pkg

    for mod in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg.__name__}."):
        importlib.import_module(mod.name)
    _DISCOVERED = True
    log.debug("Converters découverts : %d", len(_REGISTRY))


def get_converter(dataset_id: str) -> type[BaseConverter]:
    """Retourne la classe de converter enregistrée pour ``dataset_id``."""
    discover()
    if dataset_id not in _REGISTRY:
        raise KeyError(f"aucun converter pour '{dataset_id}'. Disponibles : {sorted(_REGISTRY)}")
    return _REGISTRY[dataset_id]


def available() -> list[str]:
    """Liste triée des dataset_id disposant d'un converter."""
    discover()
    return sorted(_REGISTRY)
