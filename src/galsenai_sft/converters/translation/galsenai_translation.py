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


# ════════════════════════════════════════════════════════════════════
#  Corpus parallèles supplémentaires (montée en volume de la traduction)
#
#  Ces corpus se recouvrent partiellement : `centralized_…` agrège déjà
#  plusieurs des autres (sa colonne `source` le dit), et `Alwaly/…-gs` vient
#  de MAFAND. C'est sans danger depuis que la déduplication est **globale**
#  au build (index partagé) : un doublon inter-sources est écarté à l'écriture.
# ════════════════════════════════════════════════════════════════════


@register("galsenai/centralized_wolof_french_translation_data")
class CentralizedWolofFrench(TranslationConverter):
    """Corpus fr↔wo centralisé de GalsenAI (le plus gros disponible)."""

    source_col = "fr"
    source_lang = "fr"
    target_col = "wo"
    target_lang = "wo"
    bidirectional = True


@register("sudoping01/english-wolof-translation")
class SudopingEnglishWolof(TranslationConverter):
    source_col = "en"
    source_lang = "en"
    target_col = "wo"
    target_lang = "wo"
    bidirectional = True


@register("MaroneAI/Wolof-to-French_Translation-Dataset")
class MaroneWolofToFrench(TranslationConverter):
    # Colonnes capitalisées côté source ; ici l'entrée est le wolof.
    source_col = "Input"
    source_lang = "wo"
    target_col = "Target"
    target_lang = "fr"
    bidirectional = True


@register("MaroneAI/French-Wolof_Translation-Dataset")
class MaroneFrenchToWolof(TranslationConverter):
    # Même corpus que le précédent, sens inverse et colonnes en minuscules.
    source_col = "input"
    source_lang = "fr"
    target_col = "target"
    target_lang = "wo"
    bidirectional = True


@register("Alwaly/french-wolof-translation-gs")
class AlwalyFrenchWolof(TranslationConverter):
    source_col = "french"
    source_lang = "fr"
    target_col = "wolof"
    target_lang = "wo"
    bidirectional = True


@register("galsenai/english-wolof-smol-translation")
class EnglishWolofSmol(TranslationConverter):
    source_col = "en"
    source_lang = "en"
    target_col = "wo"
    target_lang = "wo"
    bidirectional = True


@register("dofbi/jolof")
class Jolof(TranslationConverter):
    """Lexique fr↔wo : entrées courtes (mot à mot), utile pour le vocabulaire."""

    source_col = "french"
    source_lang = "fr"
    target_col = "wolof"
    target_lang = "wo"
    bidirectional = True
