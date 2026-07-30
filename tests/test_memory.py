"""Tests du garde-fou mémoire — la protection qui empêche un build de figer la
machine (incident du 30/07/2026).
"""

from __future__ import annotations

import pytest

from galsenai_sft.core.memory import (
    MemoryGuard,
    MemoryPressure,
    available_mb,
    cgroup_limit_mb,
    describe_environment,
    process_rss_mb,
)


def test_lectures_proc_plausibles():
    assert available_mb() > 0
    assert process_rss_mb() > 0  # le processus pytest occupe forcément de la RAM
    limit = cgroup_limit_mb()
    assert limit is None or limit > 0
    assert "mémoire" in describe_environment()


def test_guard_inactif_ne_bloque_pas():
    guard = MemoryGuard(min_available_mb=0, interval_s=0.01)
    with guard:
        guard.evaluate()
        guard.check()  # ne doit rien lever
    assert not guard.triggered
    assert guard.peak_rss_mb > 0


def test_guard_declenche_sous_le_plancher():
    """Plancher irréaliste (1 To) -> pression détectée -> arrêt demandé."""
    guard = MemoryGuard(min_available_mb=1_000_000, interval_s=0.01)
    guard.evaluate()
    assert guard.triggered
    assert guard.reason and "plancher" in guard.reason
    with pytest.raises(MemoryPressure):
        guard.check()


def test_guard_declenche_sur_rss():
    guard = MemoryGuard(min_available_mb=0, max_rss_mb=0.001, interval_s=0.01)
    guard.evaluate()
    assert guard.triggered
    assert guard.reason and "RSS" in guard.reason


def test_guard_sans_levee_expose_le_drapeau():
    """``raise_on_pressure=False`` : le pipeline décide lui-même quoi faire."""
    guard = MemoryGuard(min_available_mb=1_000_000, raise_on_pressure=False)
    guard.evaluate()
    guard.check()  # ne lève pas
    assert guard.triggered


def test_guard_thread_detecte_en_arriere_plan():
    guard = MemoryGuard(min_available_mb=1_000_000, interval_s=0.01)
    with guard:
        deadline = 2.0
        import time

        start = time.monotonic()
        while not guard.triggered and time.monotonic() - start < deadline:
            time.sleep(0.01)
    assert guard.triggered


def test_from_settings():
    from galsenai_sft.core.config import MemoryConfig

    guard = MemoryGuard.from_settings(MemoryConfig(min_available_mb=42, max_rss_mb=99))
    assert guard.min_available_mb == 42
    assert guard.max_rss_mb == 99
