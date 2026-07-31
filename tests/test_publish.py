"""Tests de la data card publiée sur HuggingFace.

La carte est le seul document que verra un consommateur du dataset : ce qui
n'y figure pas n'existe pas pour lui. Ces tests verrouillent donc la présence
des statistiques **et** des défauts connus.
"""

from __future__ import annotations

import json

import pytest

from galsenai_sft.build import BuildEntryReport, BuildManifest
from galsenai_sft.publish import TASK_DOC, build_datacard

STATS = {
    "total": 1000,
    "by_task": {"classification": 700, "tool_use": 200, "ner": 100},
    "by_prompt_lang": {"wo": 600, "fr": 400},
    "multi_turn": 0,
    "with_tool_calls": 0,
    "total_tool_calls": 0,
    "total_user_chars": 150_000,
    "total_assistant_chars": 250_000,
}


@pytest.fixture
def manifest() -> BuildManifest:
    return BuildManifest(
        version="1.2.3",
        created_at="2026-07-30T18:01:55+00:00",
        total_samples=1000,
        by_task={"classification": 700, "tool_use": 200, "ner": 100},
        entries=[
            BuildEntryReport(
                dataset_id="org/classif",
                task="classification",
                split="train",
                n_raw=750,
                n_samples=700,
            ),
            BuildEntryReport(
                dataset_id="org/code",
                task="tool_use",
                split="train",
                n_raw=200,
                n_samples=200,
            ),
            BuildEntryReport(
                dataset_id="org/ner",
                task="ner",
                split="train",
                n_raw=50,
                n_samples=100,
            ),
            # Entrée sans exemple : ne doit pas polluer le tableau des sources.
            BuildEntryReport(
                dataset_id="org/vide",
                task="qa",
                split="train",
                n_raw=0,
                n_samples=0,
            ),
        ],
    )


@pytest.fixture
def card(manifest, tmp_path, monkeypatch) -> str:
    """Data card générée avec des statistiques contrôlées (pas celles du disque)."""
    from galsenai_sft.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings.paths, "interim", tmp_path)
    (tmp_path / "build_stats.json").write_text(json.dumps(STATS), encoding="utf-8")
    return build_datacard(manifest, "org/mon_sft")


# ════════════════════════════════════════════════════════════════════
#  En-tête : le viewer HF doit savoir où sont les données
# ════════════════════════════════════════════════════════════════════
def test_entete_declare_les_data_files(card):
    """Sans bloc configs/data_files, le viewer HF ne trouve pas le fichier."""
    header = card.split("---")[1]
    assert "configs:" in header
    assert "path: data/train.jsonl" in header
    assert "split: train" in header


def test_entete_ne_declare_aucune_licence(card):
    """Choix assumé : les sources ont des licences hétérogènes."""
    header = card.split("---")[1]
    assert "license" not in header


# ════════════════════════════════════════════════════════════════════
#  Consigne : statistiques
# ════════════════════════════════════════════════════════════════════
def test_chaque_tache_est_expliquee(card, manifest):
    """Une part sans explication n'informe pas : chaque tâche porte sa description."""
    for task in manifest.by_task:
        assert f"`{task}`" in card
        assert TASK_DOC[task][:40] in card


def test_parts_par_tache(card):
    assert "70.0%" in card  # classification 700/1000
    assert "20.0%" in card  # tool_use


def test_langue_de_consigne_presente(card):
    assert "wolof (`wo`)" in card and "60.0%" in card
    assert "français (`fr`)" in card and "40.0%" in card


def test_longueurs_moyennes_calculees(card):
    assert "150 car." in card  # 150_000 / 1000 consignes
    assert "250 car." in card  # 250_000 / 1000 réponses


def test_sources_avec_taux_de_conservation(card):
    """Le rapport lignes lues -> exemples explique les écarts de volume."""
    assert "| `org/classif` | classification | 750 | 700 | 93% |" in card
    assert "| `org/ner` | ner | 50 | 100 | 200% |" in card  # NER dupliqué


def test_source_sans_exemple_est_omise(card):
    assert "org/vide" not in card


# ════════════════════════════════════════════════════════════════════
#  Consigne : explications et défauts connus
# ════════════════════════════════════════════════════════════════════
def test_desequilibre_signale(card):
    assert "Déséquilibre des tâches" in card
    assert "classification" in card


def test_absence_de_tool_call_signalee(card):
    """La tâche s'appelle tool_use mais ne contient aucun appel d'outil."""
    assert "aucun appel d'outil" in card


def test_absence_de_split_signalee(card):
    assert "aucun jeu de validation ou de test" in card
    assert "Pas de jeu d'évaluation" in card
    assert "décontamination" in card.lower()


def test_limites_qualite_signalees(card):
    assert "Vérification de langue partielle" in card
    assert "Déduplication exacte uniquement" in card


def test_renvoi_vers_licences_et_ciblage(card):
    """La carte ne porte pas les licences : elle doit dire où les trouver."""
    assert "dataset_catalog.md" in card
    assert "targeting.md" in card


def test_exemple_de_chargement_utilisable(card):
    assert "load_dataset" in card


# ════════════════════════════════════════════════════════════════════
#  Robustesse
# ════════════════════════════════════════════════════════════════════
def test_card_generable_sans_statistiques(manifest, tmp_path, monkeypatch):
    """build_stats.json absent : la carte doit rester générable, sans la section."""
    from galsenai_sft.core.config import get_settings

    monkeypatch.setattr(get_settings().paths, "interim", tmp_path)
    card = build_datacard(manifest, "org/mon_sft")
    assert "## Tâches" in card
    assert "## Statistiques" not in card
    assert "Pas de jeu d'évaluation" in card  # les limites structurelles restent


def test_build_partiel_signale(manifest, tmp_path, monkeypatch):
    from galsenai_sft.core.config import get_settings

    monkeypatch.setattr(get_settings().paths, "interim", tmp_path)
    manifest.partial = True
    manifest.stop_reason = "mémoire disponible sous le plancher"
    card = build_datacard(manifest, "org/mon_sft")
    assert "Build partiel" in card
    assert "plancher" in card


def test_manifest_vide_ne_leve_pas(tmp_path, monkeypatch):
    """Division par le total : un manifest à 0 exemple ne doit pas planter."""
    from galsenai_sft.core.config import get_settings

    monkeypatch.setattr(get_settings().paths, "interim", tmp_path)
    empty = BuildManifest(version="0", created_at="2026-01-01T00:00:00+00:00")
    card = build_datacard(empty, "org/vide")
    assert "org/vide" in card


# ════════════════════════════════════════════════════════════════════
#  v0.2 : la carte doit suivre la réalité du build
# ════════════════════════════════════════════════════════════════════
def test_langues_declarees_suivent_les_statistiques(card):
    """Les jeux de function calling sont anglophones : l'omettre serait faux."""
    header = card.split("---")[1]
    assert "- wo" in header and "- fr" in header
    assert header.index("- wo") < header.index("- fr"), "le wolof reste la langue principale"


def test_jeux_d_evaluation_verses_dans_train_sont_signales(tmp_path, monkeypatch):
    from galsenai_sft.core.config import get_settings

    monkeypatch.setattr(get_settings().paths, "interim", tmp_path)
    m = BuildManifest(
        version="0.2.0",
        created_at="2026-07-31T01:04:09+00:00",
        total_samples=100,
        by_task={"qa": 100},
        entries=[
            BuildEntryReport(
                dataset_id="facebook/belebele", task="qa", split="test", n_raw=100, n_samples=100
            )
        ],
    )
    card = build_datacard(m, "org/x")
    assert "versés dans `train`" in card
    assert "facebook/belebele" in card


def test_pas_d_alerte_eval_quand_tout_vient_de_train(card):
    assert "versés dans `train`" not in card


def test_dedup_decrite_comme_globale_et_exacte(card):
    """La v0.1 annonçait une déduplication locale : ce n'est plus vrai."""
    assert "globalement" in card
    assert "index partagé" in card
    assert "MinHash" in card
