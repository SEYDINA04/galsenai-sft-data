"""Converters de traduction concrets (un par dataset source).

Chaque classe ne fait que déclarer ses colonnes/langues et s'enregistrer :
l'ajout d'un nouveau corpus parallèle = ~5 lignes, sans toucher au pipeline.
"""

from __future__ import annotations

from galsenai_sft.converters.translation.base_translation import TranslationConverter
from galsenai_sft.registry import register


@register("galsenai/french-wolof-translation")
class FrenchWolofTranslation(TranslationConverter):
    source_col = "french"
    source_lang = "fr"
    target_col = "wolof"
    target_lang = "wo"
    bidirectional = True


@register("bilalfaye/english-wolof-french-dataset")
class EnglishWolofFrench(TranslationConverter):
    # Direction anglais -> wolof (colonnes du dataset source).
    source_col = "en"
    source_lang = "en"
    target_col = "wo"
    target_lang = "wo"
    bidirectional = True
