"""Filtre de contenu : retient les exemples dont la **cible wolof** est propre.

Certains datasets (ex. WORI) contiennent du texte cible bruité ou dans une autre
langue (français, ourdou, créole…). Ce filtre vérifie que le dernier tour
assistant est :
  - non vide et non « bruité » (répétitions de caractères, ponctuation
    excessive, trop court) ;
  - détecté comme **wolof** par le LID (GlotLID v3) au-dessus du seuil.

⚠️ À n'appliquer QU'aux datasets dont la cible est du wolof libre (instruction,
QA, traduction→wolof). Il ne doit PAS filtrer les tâches dont la réponse n'est
pas du wolof : labels EN (classification/intent), JSON (NER), code (tool_use),
traduction→français. Le pilotage se fait par dataset dans le plan de build.
"""

from __future__ import annotations

import re

from galsenai_sft.core.schema import Role, Sample

_CHAR_RUN = re.compile(r"(.)\1{3,}")  # même caractère répété >= 4 fois


def is_noisy_text(text: str, max_punct: int = 4) -> bool:
    """Heuristiques bon marché de détection de bruit (sans LID)."""
    t = text.strip()
    if len(t.split()) < 3:
        return True
    if _CHAR_RUN.search(t):  # ex. "!!!!", "aaaaa"
        return True
    if (t.count("!") + t.count("?")) >= max_punct:
        return True
    # Trop peu de lettres (majoritairement symboles/chiffres)
    letters = sum(c.isalpha() for c in t)
    if letters < 0.5 * len(t):
        return True
    return False


class WolofTargetFilter:
    """Garde un Sample si sa cible (dernier assistant) est du wolof propre."""

    def __init__(self, identifier=None, threshold: float = 0.5, target_label: str = "wol_Latn"):
        self._identifier = identifier
        self.threshold = threshold
        self.target_label = target_label

    def _lid(self):
        if self._identifier is None:
            from galsenai_sft.validators.lid import get_identifier

            self._identifier = get_identifier()
        return self._identifier

    def keep(self, sample: Sample) -> bool:
        assistant = next(
            (m.content for m in reversed(sample.messages) if m.role is Role.ASSISTANT), ""
        )
        if not assistant.strip() or is_noisy_text(assistant):
            return False
        label, conf = self._lid().predict(assistant)
        return label == self.target_label and conf >= self.threshold

    def release(self) -> None:
        """Libère le modèle LID (~1,6 Go) : plus aucun dataset à filtrer."""
        from galsenai_sft.validators.lid import release_identifier

        self._identifier = None
        release_identifier()
