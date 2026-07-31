"""Tests des validators (schéma, qualité, décontamination, statistiques)."""

from __future__ import annotations

from galsenai_sft.core.schema import Message, Role, Sample, TaskType, ToolCall
from galsenai_sft.validators import (
    compute_statistics,
    sample_fingerprint,
    validate_quality,
    validate_raw_chatml,
)
from galsenai_sft.validators.decontamination import decontaminate
from galsenai_sft.validators.quality_validator import filter_quality


def _s(user: str, assistant: str, **kw) -> Sample:
    return Sample(
        messages=[
            Message(role=Role.USER, content=user),
            Message(role=Role.ASSISTANT, content=assistant),
        ],
        task=kw.get("task", TaskType.TRANSLATION),
        source=kw.get("source", "test/ds"),
    )


# --- schema_validator ---
def test_raw_chatml_valid():
    row = {
        "messages": [
            {"role": "user", "content": "Naka?"},
            {"role": "assistant", "content": "Maa ngi fi."},
        ]
    }
    assert validate_raw_chatml(row) == []


def test_raw_chatml_no_final_assistant():
    row = {"messages": [{"role": "user", "content": "Naka?"}]}
    codes = {i.code for i in validate_raw_chatml(row)}
    assert "no_final_assistant" in codes


def test_raw_chatml_invalid_role_and_empty():
    row = {"messages": [{"role": "robot", "content": ""}, {"role": "assistant", "content": "ok"}]}
    codes = {i.code for i in validate_raw_chatml(row)}
    assert "invalid_role" in codes and "empty_content" in codes


# --- quality_validator ---
def test_duplicate_detection():
    rep = validate_quality([_s("Bonjour", "Salaam"), _s("Bonjour", "Salaam")])
    assert not rep.ok
    assert rep.counts_by_code().get("duplicate") == 1


def test_echo_answer_warning():
    rep = validate_quality([_s("Dakar", "Dakar")])
    assert rep.ok  # warning seulement
    assert "echo_answer" in rep.counts_by_code()


def test_filter_quality_removes_duplicates():
    kept = list(filter_quality([_s("a", "b"), _s("a", "b"), _s("c", "d")]))
    assert len(kept) == 2


def test_fingerprint_stable():
    assert sample_fingerprint(_s("X", "Y")) == sample_fingerprint(_s("x", "y "))


# --- decontamination ---
def test_decontaminate_removes_seen_text():
    import hashlib

    def h(t):
        return hashlib.sha1(" ".join(t.lower().split()).encode()).hexdigest()

    index = {h("Salaamaalekum")}
    kept = list(decontaminate([_s("Bonjour", "Salaamaalekum"), _s("Merci", "Jërëjëf")], index))
    assert len(kept) == 1
    assert kept[0].messages[-1].content == "Jërëjëf"


def test_decontaminate_noop_without_index():
    samples = [_s("a", "b"), _s("c", "d")]
    assert len(list(decontaminate(samples, set()))) == 2


# --- statistics ---
def test_statistics():
    tool_sample = Sample(
        messages=[
            Message(role=Role.USER, content="Météo à Dakar ?"),
            Message(
                role=Role.ASSISTANT,
                tool_calls=[ToolCall(name="weather", arguments={"city": "Dakar"})],
            ),
        ],
        task=TaskType.TOOL_USE,
        source="test/tools",
    )
    stats = compute_statistics([_s("a", "bb"), tool_sample])
    assert stats.total == 2
    assert stats.by_task["translation"] == 1
    assert stats.with_tool_calls == 1
    assert stats.by_source["test/ds"] == 1


def test_rapport_borne_le_detail_mais_pas_les_compteurs():
    """Mémoire bornée : compteurs exacts, détail limité (datasets massifs)."""
    from galsenai_sft.validators.report import Issue, Severity, ValidationReport

    rep = ValidationReport(max_issues=10)
    for i in range(1000):
        rep.add(Issue(code="duplicate", severity=Severity.ERROR, message="x", sample_index=i))

    assert rep.n_errors == 1000  # compteur exact
    assert len(rep.issues) == 10  # détail borné
    assert rep.truncated is True
    assert rep.counts_by_code()["duplicate"] == 1000
    assert rep.ok is False
    assert "1000 erreurs" in rep.summary()


# ════════════════════════════════════════════════════════════════════
#  Déduplication globale (v0.2) : partagée entre datasets
# ════════════════════════════════════════════════════════════════════
def test_dedup_partage_entre_deux_datasets():
    """Plusieurs corpus de traduction wolof agrègent les mêmes phrases :
    sans index partagé, le doublon inter-sources passerait."""
    from galsenai_sft.core.schema import Message, Role, Sample, TaskType
    from galsenai_sft.validators.quality_validator import filter_quality

    def make(source: str) -> Sample:
        return Sample(
            messages=[
                Message(role=Role.USER, content="Tekkil lii ci wolof: Bonjour"),
                Message(role=Role.ASSISTANT, content="Salaamaalekum"),
            ],
            task=TaskType.TRANSLATION,
            source=source,
        )

    shared: set[int] = set()
    first = list(filter_quality([make("corpus/a")], seen=shared))
    second = list(filter_quality([make("corpus/b")], seen=shared))
    assert len(first) == 1
    assert second == [], "le même contenu venu d'une autre source doit être écarté"


def test_dedup_reste_local_sans_index_partage():
    """Sans `seen`, chaque appel repart d'un index vide (comportement d'origine)."""
    from galsenai_sft.core.schema import Message, Role, Sample, TaskType
    from galsenai_sft.validators.quality_validator import filter_quality

    s = Sample(
        messages=[
            Message(role=Role.USER, content="q"),
            Message(role=Role.ASSISTANT, content="r"),
        ],
        task=TaskType.QA,
        source="x",
    )
    assert len(list(filter_quality([s]))) == 1
    assert len(list(filter_quality([s]))) == 1
