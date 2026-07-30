"""Tests des converters wolof (lot 3), avec lignes brutes fidèles aux schémas HF."""

from __future__ import annotations

import json

from galsenai_sft.core.schema import Role, TaskType
from galsenai_sft.registry import available, get_converter


def _convert(dataset_id: str, rows: list[dict], seed: int = 0):
    conv = get_converter(dataset_id)(seed=seed)
    return list(conv.convert(rows))


def test_all_expected_converters_registered():
    ids = set(available())
    expected = {
        "masakhane/InjongoIntent",
        "karim155/WolBanking77",
        "mbaye930/WolofEntityLinking",
        "Davlan/sib200",
        "michsethowusu/wolof-sentiments-corpus",
        "masakhane/afriqa",
        "m-a-d-i/wori-wolof-instructions",
        "michsethowusu/Code-170k-wolof",
    }
    assert expected.issubset(ids)


def test_injongo_intent_and_slots():
    row = {
        "intent": "alarm",
        "text": "Ndax mën nga def ab alarm ci Dakaar ci 11:55?",
        "target": "CITY_OR_PROVINCE: Dakaar $$ TIME: 11:55",
    }
    samples = _convert("masakhane/InjongoIntent", [row])
    subtasks = {s.meta["subtask"] for s in samples}
    assert subtasks == {"intent", "slot_filling"}
    intent_s = next(s for s in samples if s.meta["subtask"] == "intent")
    assert intent_s.messages[-1].content == "alarm"
    assert intent_s.task is TaskType.INTENT
    slot_s = next(s for s in samples if s.meta["subtask"] == "slot_filling")
    slots = json.loads(slot_s.messages[-1].content)
    assert slots["CITY_OR_PROVINCE"] == "Dakaar" and slots["TIME"] == "11:55"


def test_wolbanking77():
    row = {"input_wo": "Jotuma xaalis waaye terewul.", "label": "wrong amount of cash received"}
    s = _convert("karim155/WolBanking77", [row])[0]
    assert s.messages[-1].content == "wrong amount of cash received"
    assert s.meta["domain"] == "banking"


def test_ner_json_output():
    row = {
        "text": "Móritani ak Madagaskaar ñu ngi ci mbuus 3 gi.",
        "entities": [
            {"text": "Móritani", "ner_type": "LOC"},
            {"text": "Madagaskaar", "ner_type": "LOC"},
        ],
    }
    s = _convert("mbaye930/WolofEntityLinking", [row])[0]
    ents = json.loads(s.messages[-1].content)
    assert {"text": "Móritani", "type": "LOC"} in ents
    assert s.task is TaskType.NER


def test_sib200_topic():
    row = {"category": "geography", "text": "Géej moo wër Turquie."}
    s = _convert("Davlan/sib200", [row])[0]
    assert s.messages[-1].content == "geography"
    assert s.task is TaskType.CLASSIFICATION


def test_sentiment():
    row = {"Wolof": "Bo gisee ma gën laa néew alal.", "sentiment": "Negative"}
    s = _convert("michsethowusu/wolof-sentiments-corpus", [row])[0]
    assert s.messages[-1].content == "Negative"


def test_afriqa_parses_stringified_list():
    row = {"question": "Lan moo waraloon?", "answers": "['Ngir xeex bi']"}
    s = _convert("masakhane/afriqa", [row])[0]
    assert s.messages[-1].content == "Ngir xeex bi"
    assert s.task is TaskType.QA


def test_wori_bilingual():
    row = {
        "text_wo": "Mitchell Gourley moo jël onzième palaas.",
        "instruction_wo": "Ci atlet yii, ban réew la muy teewal?",
        "instruction_fr": "Quel pays représente cet athlète ?",
    }
    samples = _convert("m-a-d-i/wori-wolof-instructions", [row])
    langs = {s.prompt_lang.value for s in samples}
    assert langs == {"wo", "fr"}
    for s in samples:
        assert s.messages[-1].content.startswith("Mitchell")


def test_code170k_sharegpt_multiturn():
    row = {
        "conversations": [
            {"from": "human", "value": "Naka lañuy tekki kod bii?"},
            {"from": "gpt", "value": "Man nga jëfandikoo ```python\nprint('a')\n```"},
        ]
    }
    s = _convert("michsethowusu/Code-170k-wolof", [row])[0]
    assert s.task is TaskType.TOOL_USE
    assert s.messages[0].role is Role.USER
    assert s.messages[-1].role is Role.ASSISTANT


def test_code170k_trailing_non_assistant_trimmed():
    row = {
        "conversations": [
            {"from": "human", "value": "Q1"},
            {"from": "gpt", "value": "A1"},
            {"from": "human", "value": "Q2 sans réponse"},
        ]
    }
    s = _convert("michsethowusu/Code-170k-wolof", [row])[0]
    assert s.messages[-1].role is Role.ASSISTANT
    assert s.messages[-1].content == "A1"
