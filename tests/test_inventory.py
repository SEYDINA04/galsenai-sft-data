"""Tests de l'inventaire amont — aucun accès réseau (sonde injectée)."""

from __future__ import annotations

import pytest

from galsenai_sft.inventory import (
    CANDIDATE,
    EXCLUDED,
    INTEGRATED,
    Inventory,
    SourceVolume,
    SplitSize,
    build_inventory,
    measure,
    render_inventory,
    write_inventory,
)


class FakeProbe:
    """Sonde factice : sert des tailles fixes, ou lève pour simuler un dataset gated."""

    def __init__(self, table: dict[str, list[SplitSize]], failing: set[str] | None = None):
        self.table = table
        self.failing = failing or set()
        self.calls: list[str] = []

    def sizes(self, dataset_id: str) -> list[SplitSize]:
        self.calls.append(dataset_id)
        if dataset_id in self.failing:
            raise RuntimeError("HTTP Error 404: Not Found")
        return self.table.get(dataset_id, [])


TABLE = {
    "a/multi": [
        SplitSize(config="wol", split="train", n_rows=100),
        SplitSize(config="wol", split="test", n_rows=25),
        SplitSize(config="fra", split="train", n_rows=999),
    ],
    "a/single": [
        SplitSize(config="default", split="train", n_rows=40),
        SplitSize(config="default", split="validation", n_rows=10),
    ],
    "a/unnamed": [SplitSize(config="lonely", split="train", n_rows=7)],
}


# ════════════════════════════════════════════════════════════════════
#  measure() : ciblage config/split
# ════════════════════════════════════════════════════════════════════
def test_measure_cible_la_config_demandee():
    """Une config explicite ne doit jamais faire fuiter les autres langues."""
    n_t, n_all, err = measure("a/multi", FakeProbe(TABLE), config="wol", split="train")
    assert (n_t, n_all, err) == (100, 125, None)


def test_measure_sans_config_prend_default():
    n_t, n_all, err = measure("a/single", FakeProbe(TABLE), config=None, split="train")
    assert (n_t, n_all, err) == (40, 50, None)


def test_measure_sans_config_prend_l_unique_config():
    """Pas de 'default' mais une seule config : c'est celle que load_dataset utilise."""
    n_t, n_all, err = measure("a/unnamed", FakeProbe(TABLE), config=None, split="train")
    assert (n_t, n_all) == (7, 7)
    assert err is None


def test_measure_signale_une_config_ambigue():
    """Plusieurs configs et aucune demandée : on somme, mais on le dit."""
    n_t, n_all, err = measure("a/multi", FakeProbe(TABLE), config=None, split="train")
    assert (n_t, n_all) == (1099, 1124)
    assert err is not None and "ambigu" in err


def test_measure_capture_l_erreur_sans_lever():
    """Un dataset gated ne doit pas faire échouer tout l'inventaire."""
    n_t, n_all, err = measure("a/gated", FakeProbe(TABLE, failing={"a/gated"}))
    assert (n_t, n_all) == (None, None)
    assert err is not None and "404" in err


def test_measure_config_introuvable():
    _, _, err = measure("a/multi", FakeProbe(TABLE), config="zzz")
    assert err is not None and "introuvable" in err


# ════════════════════════════════════════════════════════════════════
#  build_inventory() : agrégation par tâche
# ════════════════════════════════════════════════════════════════════
def test_inventory_agrege_par_tache(monkeypatch):
    """Le cœur de la consigne : un volume disponible par tâche, pas par dataset."""
    import galsenai_sft.metadata as meta_mod

    # La tâche vient du registre (donc du converter) : on la stube pour rester hors ligne.
    monkeypatch.setattr(
        meta_mod,
        "load_registry",
        lambda: {
            "a/multi": type("M", (), {"task": "translation"})(),
            "a/single": type("M", (), {"task": "translation"})(),
        },
    )

    plan = [
        {"id": "a/multi", "config": "wol", "split": "train"},
        {"id": "a/single"},
    ]
    candidates = [
        {"id": "a/unnamed", "task": "ner", "status": CANDIDATE, "reason": "à écrire"},
        {"id": "a/multi", "task": "ner", "config": "fra", "status": EXCLUDED, "reason": "licence"},
    ]

    inv = build_inventory(plan=plan, probe=FakeProbe(TABLE), candidates=candidates)

    assert inv.by_task["translation"].n_integrated_rows == 140  # 100 + 40
    assert inv.by_task["ner"].n_candidate_rows == 7
    assert inv.by_task["ner"].n_excluded_rows == 999
    assert inv.by_task["ner"].n_reachable == 7  # l'écarté ne compte pas
    assert inv.total_integrated == 140
    assert inv.total_reachable == 147


def test_inventory_n_unused_compte_les_splits_non_lus():
    """Les splits validation/test sont disponibles mais jamais lus par le build."""
    s = SourceVolume(dataset_id="x", task="qa", n_targeted=503, n_all_splits=1341)
    assert s.n_unused == 838


def test_n_unused_vaut_zero_si_mesure_absente():
    assert SourceVolume(dataset_id="x", task="qa").n_unused == 0


# ════════════════════════════════════════════════════════════════════
#  Rendu & persistance
# ════════════════════════════════════════════════════════════════════
@pytest.fixture
def sample_inventory() -> Inventory:
    return build_inventory(
        plan=[{"id": "a/multi", "config": "wol", "split": "train"}],
        probe=FakeProbe(TABLE),
        candidates=[{"id": "a/unnamed", "task": "ner", "status": EXCLUDED, "reason": "licence NC"}],
    )


def test_render_expose_les_motifs_d_exclusion(sample_inventory):
    md = render_inventory(sample_inventory)
    assert "Par tâche" in md
    assert "licence NC" in md  # le motif doit être lisible, pas seulement le chiffre
    assert "écarté" in md


def test_render_signale_les_lignes_non_lues(sample_inventory):
    """25 lignes de test existent chez a/multi mais n'entrent pas dans le build."""
    assert "non lues" in render_inventory(sample_inventory)


def test_write_puis_load_est_idempotent(tmp_path, sample_inventory):
    from galsenai_sft.inventory import load_inventory

    jp, mp = write_inventory(
        sample_inventory, json_path=tmp_path / "inv.json", md_path=tmp_path / "inv.md"
    )
    assert jp.exists() and mp.exists()
    relu = load_inventory(jp)
    assert relu is not None
    assert relu.by_task.keys() == sample_inventory.by_task.keys()
    assert relu.total_integrated == sample_inventory.total_integrated


def test_load_inventory_absent_retourne_none(tmp_path):
    from galsenai_sft.inventory import load_inventory

    assert load_inventory(tmp_path / "nexiste_pas.json") is None


# ════════════════════════════════════════════════════════════════════
#  candidates.yaml : le fichier réel doit rester exploitable
# ════════════════════════════════════════════════════════════════════
def test_candidates_yaml_est_bien_forme():
    from galsenai_sft.inventory import load_candidates

    cands = load_candidates()
    assert cands, "metadata/candidates.yaml doit tracer les datasets ciblés non intégrés"
    for c in cands:
        assert "id" in c and "task" in c
        assert c.get("status") in {CANDIDATE, EXCLUDED, INTEGRATED}
        assert c.get("reason"), f"{c['id']} : un dataset écarté/candidat doit porter un motif"
