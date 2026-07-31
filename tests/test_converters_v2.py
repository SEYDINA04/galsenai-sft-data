"""Tests des converters ajoutés en v0.2 (montée en volume des tâches).

Les lignes de test reproduisent le **format réel** de chaque source, relevé sur
l'API HuggingFace : c'est ce qui fait la valeur de ces tests, un format
supposé ne prouverait rien.
"""

from __future__ import annotations

import json

from galsenai_sft.core.schema import PromptLang, Role, TaskType
from galsenai_sft.registry import get_converter


def convert(dataset_id: str, row: dict):
    return get_converter(dataset_id)().convert_row(row)


# ════════════════════════════════════════════════════════════════════
#  Traduction : colonnes hétérogènes d'un corpus à l'autre
# ════════════════════════════════════════════════════════════════════
def test_corpus_centralise_produit_les_deux_directions():
    out = convert(
        "galsenai/centralized_wolof_french_translation_data",
        {"wo": "Arme yi wóor nañu leen!", "fr": "Les armes sont en place!", "source": "x"},
    )
    assert len(out) == 2
    assert {s.meta["direction"] for s in out} == {"fr->wo", "wo->fr"}
    assert all(s.task is TaskType.TRANSLATION for s in out)


def test_maroneai_colonnes_capitalisees():
    """Un des deux jeux MaroneAI utilise Input/Target, l'autre input/target."""
    wo_fr = convert(
        "MaroneAI/Wolof-to-French_Translation-Dataset",
        {"Input": "Omar a lekk ceeb bi", "Target": "C'est Omar qui a mangé le riz."},
    )
    fr_wo = convert(
        "MaroneAI/French-Wolof_Translation-Dataset",
        {"input": "C'est Omar qui a mangé le riz.", "target": "Omar a lekk ceeb bi"},
    )
    assert {s.meta["direction"] for s in wo_fr} == {"wo->fr", "fr->wo"}
    assert {s.meta["direction"] for s in fr_wo} == {"fr->wo", "wo->fr"}


def test_traduction_ignore_une_ligne_incomplete():
    assert convert("dofbi/jolof", {"wolof": "aakimu", "french": ""}) == []


# ════════════════════════════════════════════════════════════════════
#  Instruction : Alpaca (avec/sans contexte) et Aya
# ════════════════════════════════════════════════════════════════════
def test_alpaca_attache_le_contexte_a_la_consigne():
    """`input` n'a pas d'équivalent ChatML : le perdre changerait le sens."""
    out = convert(
        "bilalfaye/wolof-sft",
        {
            "instruction": "Bindal benn mbind bu am 3 baat",
            "input": "Kaaraange reso",
            "output": "Aar lëkkaloo yi ci net bi",
        },
    )
    assert len(out) == 1
    user = out[0].messages[0].content
    assert "Bindal benn mbind" in user and "Kaaraange reso" in user
    assert out[0].meta["has_context"] is True


def test_alpaca_sans_contexte():
    out = convert(
        "ngia/alpaca-data-in-wolof",
        {"instruction": "Jox ñatti xeeti feem", "input": None, "output": "1. Lekk regime"},
    )
    assert out[0].messages[0].content == "Jox ñatti xeeti feem"
    assert out[0].meta["has_context"] is False


def test_alpaca_sans_sortie_est_ignore():
    assert convert("ngia/alpaca-data-in-wolof", {"instruction": "x", "output": ""}) == []


def test_aya_garde_la_provenance_fine():
    """Aya agrège plusieurs jeux : sans `dataset_name`, la trace est perdue."""
    out = convert(
        "CohereLabs/aya_collection_language_split",
        {
            "inputs": "Fowub F.C Barsaa nu mu tudd ?",
            "targets": "Kàmpunóo",
            "dataset_name": "AfriQA-inst",
            "task_type": "question-answering",
        },
    )
    assert out[0].meta["aya_dataset"] == "AfriQA-inst"
    assert out[0].task is TaskType.INSTRUCTION


# ════════════════════════════════════════════════════════════════════
#  NER : reconstruction des entités depuis les étiquettes BIO
# ════════════════════════════════════════════════════════════════════
def test_masakhaner_reconstruit_les_entites_multi_tokens():
    """L'enjeu : « Ëmmë Seen » (B-PER, I-PER) est UNE entité, pas deux."""
    out = convert(
        "masakhane/masakhaner2",
        {
            "tokens": ["Ñëwoon", "ca", "Kawlag", ",", "Ëmmë", "Seen", "."],
            "ner_tags": [0, 0, 5, 0, 1, 2, 0],
        },
    )
    entities = json.loads(out[0].messages[-1].content)
    assert {"text": "Kawlag", "type": "LOC"} in entities
    assert {"text": "Ëmmë Seen", "type": "PER"} in entities
    assert len(entities) == 2


def test_masakhaner_accepte_des_tags_deja_textuels():
    out = convert(
        "masakhane/masakhaner2",
        {"tokens": ["Dakar", "la"], "ner_tags": ["B-LOC", "O"]},
    )
    assert json.loads(out[0].messages[-1].content) == [{"text": "Dakar", "type": "LOC"}]


def test_masakhaner_phrase_sans_entite_reste_un_exemple():
    """Les phrases négatives apprennent au modèle à répondre `[]`."""
    out = convert("masakhane/masakhaner2", {"tokens": ["Dama", "bëgg"], "ner_tags": [0, 0]})
    assert json.loads(out[0].messages[-1].content) == []


def test_bio_tolere_un_i_orphelin():
    from galsenai_sft.converters.ner.masakhaner import spans_from_bio

    spans = spans_from_bio(["Macky", "Sall"], ["I-PER", "I-PER"])
    assert spans == [{"text": "Macky Sall", "type": "PER"}]


# ════════════════════════════════════════════════════════════════════
#  QA : QCM et mathématiques
# ════════════════════════════════════════════════════════════════════
def test_belebele_repond_par_la_lettre():
    out = convert(
        "facebook/belebele",
        {
            "flores_passage": "Un passage.",
            "question": "Laaj bi?",
            "mc_answer1": "a",
            "mc_answer2": "b",
            "mc_answer3": "c",
            "mc_answer4": "d",
            "correct_answer_num": 3,
        },
    )
    assert out[0].messages[-1].content == "C"
    assert "Un passage." in out[0].messages[0].content
    assert "C. c" in out[0].messages[0].content


def test_belebele_index_invalide_est_ignore():
    row = {
        "flores_passage": "p",
        "question": "q",
        "mc_answer1": "a",
        "mc_answer2": "b",
        "mc_answer3": "c",
        "mc_answer4": "d",
        "correct_answer_num": 9,
    }
    assert convert("facebook/belebele", row) == []


def test_afrimmlu_parse_les_choix_serialises():
    """Le champ `choices` arrive en chaîne Python, pas en liste."""
    out = convert(
        "masakhane/afrimmlu",
        {
            "subject": "maths",
            "question": "Lan mooy solos p ci 24 = 2p ?",
            "choices": "['p = 4', 'p = 8', 'p = 12', 'p = 24']",
            "answer": "C",
        },
    )
    assert out[0].messages[-1].content == "C"
    assert "C. p = 12" in out[0].messages[0].content


def test_afrimgsm_utilise_answer_number():
    """`answer` est nul dans les splits traduits ; `answer_number` ne l'est pas."""
    out = convert(
        "masakhane/afrimgsm",
        {"question": "Ñaata?", "answer": None, "answer_number": 18, "equation_solution": None},
    )
    assert out[0].messages[-1].content == "18"


# ════════════════════════════════════════════════════════════════════
#  NLI
# ════════════════════════════════════════════════════════════════════
def test_afrixnli_mappe_l_entier_vers_le_libelle():
    out = convert("masakhane/afrixnli", {"premise": "a", "hypothesis": "b", "label": 2})
    assert out[0].messages[-1].content == "contradiction"
    assert out[0].task is TaskType.CLASSIFICATION


def test_afrixnli_label_hors_bornes_ignore():
    assert convert("masakhane/afrixnli", {"premise": "a", "hypothesis": "b", "label": 7}) == []


# ════════════════════════════════════════════════════════════════════
#  Function calling : le trou que la v0.1 laissait béant
# ════════════════════════════════════════════════════════════════════
TOUCAN_ROW = {
    "tools": json.dumps(
        [
            {"type": "function", "function": {"name": "find_rhymes", "description": "d"}},
            {"type": "function", "function": {"name": "count_syllables", "description": "d"}},
        ]
    ),
    "messages": json.dumps(
        [
            {"role": "user", "content": "Rhymes for smile?"},
            {"role": "assistant", "content": "Let me look."},
            # Toucan sérialise en repr Python, avec `arguments` en JSON imbriqué.
            {
                "role": "tool_call",
                "content": "{'name': 'find_rhymes', 'arguments': '{\"w\": \"smile\"}'}",
            },
            {
                "role": "tool_call",
                "content": "{'name': 'count_syllables', 'arguments': '{\"s\": \"x\"}'}",
            },
            {"role": "tool_response", "content": "aisle, bile"},
            {"role": "assistant", "content": "Here are rhymes."},
        ]
    ),
}


def test_toucan_produit_de_vrais_tool_calls():
    out = convert("Agent-Ark/Toucan-1.5M", TOUCAN_ROW)
    assert len(out) == 1
    calls = [c for m in out[0].messages for c in (m.tool_calls or [])]
    assert [c.name for c in calls] == ["find_rhymes", "count_syllables"]
    # `arguments` était une chaîne JSON imbriquée : elle doit être désérialisée.
    assert calls[0].arguments == {"w": "smile"}
    assert out[0].meta["n_tool_calls"] == 2


def test_toucan_groupe_les_appels_paralleles():
    """Deux tool_call consécutifs = un seul tour assistant, comme à l'inférence."""
    out = convert("Agent-Ark/Toucan-1.5M", TOUCAN_ROW)
    tool_turns = [m for m in out[0].messages if m.tool_calls]
    assert len(tool_turns) == 1
    assert len(tool_turns[0].tool_calls) == 2


def test_toucan_declare_les_outils_en_systeme():
    out = convert("Agent-Ark/Toucan-1.5M", TOUCAN_ROW)
    system = out[0].messages[0]
    assert system.role is Role.SYSTEM
    assert "find_rhymes" in system.content and "count_syllables" in system.content


def test_function_calling_est_etiquete_anglais():
    """Ces jeux apportent la capacité agentique, pas du wolof."""
    out = convert("Agent-Ark/Toucan-1.5M", TOUCAN_ROW)
    assert out[0].prompt_lang is PromptLang.EN


def test_hermes_extrait_les_balises_tool_call():
    out = convert(
        "NousResearch/hermes-function-calling-v1",
        {
            "tools": '[{"type": "function", "function": {"name": "get_camera_live_feed"}}]',
            "conversations": [
                {"from": "system", "value": "You are a function calling AI model."},
                {"from": "human", "value": "Check the front door camera."},
                {
                    "from": "gpt",
                    "value": 'Sure.\n<tool_call>\n{"name": "get_camera_live_feed", '
                    '"arguments": {"camera_id": "front_door"}}\n</tool_call>',
                },
                {"from": "tool", "value": '<tool_response>\n{"url": "x"}\n</tool_response>'},
                {"from": "gpt", "value": "Here is the feed."},
            ],
        },
    )
    calls = [c for m in out[0].messages for c in (m.tool_calls or [])]
    assert calls[0].name == "get_camera_live_feed"
    assert calls[0].arguments == {"camera_id": "front_door"}
    # Le texte hors balise est conservé, la balise retirée.
    call_turn = next(m for m in out[0].messages if m.tool_calls)
    assert call_turn.content == "Sure."
    assert "<tool_call>" not in call_turn.content
    assert any(m.role is Role.TOOL for m in out[0].messages)


def test_toolace_parse_le_dsl():
    out = convert(
        "Team-ACE/ToolACE",
        {
            "system": 'Functions: [{"name": "Market Trends API", "description": "d"}]',
            "conversations": [
                {"from": "user", "value": "Top market trends?"},
                {
                    "from": "assistant",
                    "value": '[Market Trends API(trend_type="MARKET_INDEXES", country="us")]',
                },
                {"from": "tool", "value": '[{"results": {}}]'},
                {"from": "assistant", "value": "Here they are."},
            ],
        },
    )
    calls = [c for m in out[0].messages for c in (m.tool_calls or [])]
    assert calls[0].name == "Market Trends API"
    assert calls[0].arguments == {"trend_type": "MARKET_INDEXES", "country": "us"}
    assert out[0].messages[0].role is Role.SYSTEM


def test_when2call_garde_les_abstentions():
    """Répondre sans appeler alors que des outils existent EST le signal visé."""
    out = convert(
        "nvidia/When2Call",
        {
            "tools": ['{"name": "get_stations_within_1_km", "description": "d"}'],
            "messages": [
                {"role": "user", "content": "Trending topics in NYC?"},
                {"role": "assistant", "content": "I can't provide real-time information."},
            ],
        },
    )
    assert len(out) == 1
    assert out[0].meta["n_tool_calls"] == 0
    assert out[0].messages[0].role is Role.SYSTEM  # outils déclarés malgré tout


def test_conversation_sans_reponse_assistant_est_rejetee():
    row = {"tools": "[]", "messages": json.dumps([{"role": "user", "content": "hi"}])}
    assert convert("Agent-Ark/Toucan-1.5M", row) == []


def test_parsing_tolerant_ne_leve_jamais():
    from galsenai_sft.converters.tool_use.function_calling import loads_loose, to_tool_call

    assert loads_loose("pas du json {{{") is None
    assert loads_loose("") is None
    assert to_tool_call("{}") is None
    assert to_tool_call('{"name": "f"}').arguments == {}
