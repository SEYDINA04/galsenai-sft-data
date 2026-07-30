"""Détection de langue (LID) — interface pluggable + implémentation GlotLID v3.

Conçu comme un :class:`Protocol` (``LanguageIdentifier``) pour pouvoir brancher
d'autres backends (ConLID, OpenLID-v3, ensemble) sans toucher au reste du code —
principe Open/Closed. Le défaut retenu (cf. recherche) est **GlotLID v3 épinglé**
(``model_v3.bin``), rapide (fastText) et meilleur global sur évaluation externe.

``fasttext`` est une dépendance optionnelle (extra ``lid``) : l'import est
paresseux pour ne pas alourdir la plateforme quand le LID n'est pas utilisé.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, runtime_checkable

from galsenai_sft.core.config import LIDConfig, get_settings
from galsenai_sft.core.logging import get_logger

log = get_logger(__name__)


@runtime_checkable
class LanguageIdentifier(Protocol):
    """Contrat minimal d'un identificateur de langue."""

    def predict(self, text: str) -> tuple[str, float]:
        """Retourne (label, confiance) pour le top-1. Label ex. ``wol_Latn``."""
        ...


class GlotLIDv3:
    """Backend fastText GlotLID v3 (épinglé)."""

    def __init__(self, config: LIDConfig | None = None) -> None:
        self.config = config or get_settings().lid
        self._model = None

    def _load(self):
        if self._model is None:
            import fasttext
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=self.config.repo_id, filename=self.config.filename)
            log.info("GlotLID chargé : %s/%s", self.config.repo_id, self.config.filename)
            self._model = fasttext.load_model(path)
        return self._model

    def predict(self, text: str) -> tuple[str, float]:
        model = self._load()
        clean = (text or "").replace("\n", " ").replace("\r", " ").strip()
        if not clean:
            return ("__empty__", 0.0)
        # Appel direct au prédicteur C++ (évite le bug numpy 2.x de .predict).
        preds = model.f.predict(clean + "\n", 1, 0.0, "strict")
        if not preds:
            return ("__empty__", 0.0)
        prob, label = preds[0]
        return (label.replace("__label__", ""), float(prob))

    def is_target(self, text: str) -> bool:
        """Vrai si le texte est dans la langue cible au-dessus du seuil."""
        label, conf = self.predict(text)
        return label == self.config.target_label and conf >= self.config.threshold


@lru_cache(maxsize=1)
def get_identifier() -> GlotLIDv3:
    """Instance partagée du LID par défaut (chargement paresseux)."""
    return GlotLIDv3()
