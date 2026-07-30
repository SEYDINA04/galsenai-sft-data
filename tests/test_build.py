"""Tests du builder end-to-end (loader factice, sans réseau) + data card."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from galsenai_sft.build import build, convert_entry
from galsenai_sft.publish import build_datacard


class FakeLoader:
    """Loader injectable : sert des lignes en mémoire par dataset_id."""

    def __init__(self, data: dict[str, list[dict]]) -> None:
        self.data = data

    def load(
        self,
        dataset_id: str,
        split: str = "train",
        config: str | None = None,
        columns: list[str] | None = None,
    ) -> Iterable[dict[str, Any]]:
        return list(self.data.get(dataset_id, []))


FAKE_DATA = {
    "galsenai/french-wolof-translation": [
        {"french": "Bonjour", "wolof": "Salaamaalekum"},
        {"french": "Merci", "wolof": "Jërëjëf"},
    ],
    "masakhane/InjongoIntent": [
        {"intent": "alarm", "text": "Defal alarm ci 8h", "target": "TIME: 8h"},
    ],
    "mbaye930/WolofEntityLinking": [
        {"text": "Móritani réew la", "entities": [{"text": "Móritani", "ner_type": "LOC"}]},
    ],
}


def test_convert_entry_with_fake_loader():
    loader = FakeLoader(FAKE_DATA)
    samples, report = convert_entry({"id": "masakhane/InjongoIntent", "config": "wol"}, loader)
    assert report.n_raw == 1
    assert report.task == "intent"
    assert len(samples) >= 1


def test_build_end_to_end(tmp_path, monkeypatch):
    # Rediriger les sorties dans tmp_path
    from galsenai_sft.core import config as cfg_mod

    settings = cfg_mod.get_settings()
    settings.paths.processed_chatml = tmp_path / "chatml"
    settings.paths.processed_alpaca = tmp_path / "alpaca"
    settings.paths.processed_sharegpt = tmp_path / "sharegpt"
    settings.paths.interim = tmp_path / "interim"

    plan = [
        {"id": "galsenai/french-wolof-translation", "split": "train"},
        {"id": "masakhane/InjongoIntent", "split": "train", "config": "wol"},
        {"id": "mbaye930/WolofEntityLinking", "split": "train"},
    ]
    manifest = build(plan=plan, loader=FakeLoader(FAKE_DATA), settings=settings, version="test")

    assert manifest.total_samples > 0
    assert "translation" in manifest.by_task
    assert "intent" in manifest.by_task
    assert "ner" in manifest.by_task
    # Fichiers générés + checksums
    assert (tmp_path / "chatml" / "all.jsonl").exists()
    assert (tmp_path / "chatml" / "translation.jsonl").exists()
    assert (tmp_path / "alpaca" / "all.jsonl").exists()
    assert manifest.outputs  # checksums présents


def test_build_robust_to_failing_dataset(tmp_path):
    from galsenai_sft.core import config as cfg_mod

    settings = cfg_mod.get_settings()
    settings.paths.processed_chatml = tmp_path / "c"
    settings.paths.processed_alpaca = tmp_path / "a"
    settings.paths.processed_sharegpt = tmp_path / "s"
    settings.paths.interim = tmp_path / "i"

    # Dataset absent du loader -> entrée en échec, mais build continue
    plan = [
        {"id": "galsenai/french-wolof-translation"},
        {"id": "masakhane/InjongoIntent", "config": "wol"},
    ]
    loader = FakeLoader(
        {"galsenai/french-wolof-translation": FAKE_DATA["galsenai/french-wolof-translation"]}
    )
    manifest = build(plan=plan, loader=loader, settings=settings, version="t")
    # InjongoIntent renvoie 0 lignes (pas d'erreur), translation ok
    assert manifest.total_samples > 0


def test_datacard_generation():
    loader = FakeLoader(FAKE_DATA)
    import tempfile
    from pathlib import Path

    from galsenai_sft.build import build as run_build

    with tempfile.TemporaryDirectory() as d:
        from galsenai_sft.core import config as cfg_mod

        settings = cfg_mod.get_settings()
        base = Path(d)
        settings.paths.processed_chatml = base / "c"
        settings.paths.processed_alpaca = base / "a"
        settings.paths.processed_sharegpt = base / "s"
        settings.paths.interim = base / "i"
        manifest = run_build(
            plan=[{"id": "galsenai/french-wolof-translation"}],
            loader=loader,
            settings=settings,
            version="1.0",
        )
    card = build_datacard(manifest, "galsenai/wolof_sft")
    assert "galsenai/wolof_sft" in card
    assert "Répartition par tâche" in card
    assert "translation" in card


# ════════════════════════════════════════════════════════════════════
#  Mémoire : le build doit rester paresseux et survivre à une coupure
# ════════════════════════════════════════════════════════════════════
class EndlessLoader:
    """Sert 100 000 lignes : un pipeline non paresseux les matérialiserait toutes."""

    def load(self, dataset_id, split="train", config=None, columns=None):
        return ({"french": f"Bonjour {i}", "wolof": f"Salaamaalekum {i}"} for i in range(100_000))


def test_pipeline_est_paresseux():
    """Consommer 3 Samples ne doit lire que ~3 lignes source (flux, pas de liste)."""
    from galsenai_sft.build import BuildEntryReport, iter_entry_samples

    report = BuildEntryReport(
        dataset_id="galsenai/french-wolof-translation", task="t", split="train"
    )
    stream = iter_entry_samples(
        {"id": "galsenai/french-wolof-translation"}, EndlessLoader(), report
    )
    for _ in range(3):
        next(stream)

    assert report.n_samples == 3
    assert report.n_raw <= 10, "le loader a été consommé en avance : pipeline non paresseux"


def test_build_interrompu_par_la_memoire_reste_exploitable(tmp_path):
    """Pression mémoire -> arrêt propre : fichiers fermés, manifest écrit, partial=True."""
    from galsenai_sft.build import build
    from galsenai_sft.core import config as cfg_mod
    from galsenai_sft.core.memory import MemoryGuard

    settings = cfg_mod.get_settings()
    settings.paths.processed_chatml = tmp_path / "chatml"
    settings.paths.processed_alpaca = tmp_path / "alpaca"
    settings.paths.processed_sharegpt = tmp_path / "sharegpt"
    settings.paths.interim = tmp_path / "interim"

    guard = MemoryGuard(min_available_mb=1_000_000, interval_s=0.01)  # plancher inatteignable
    manifest = build(
        plan=[{"id": "galsenai/french-wolof-translation"}],
        loader=FakeLoader(FAKE_DATA),
        settings=settings,
        version="partial",
        guard=guard,
    )

    assert manifest.partial is True
    assert manifest.stop_reason
    assert (tmp_path / "chatml" / "all.jsonl").exists()  # fermé proprement
    assert (tmp_path / "interim" / "build_manifest.json").exists()


def test_manifest_trace_le_pic_memoire(tmp_path):
    from galsenai_sft.build import build
    from galsenai_sft.core import config as cfg_mod

    settings = cfg_mod.get_settings()
    settings.paths.processed_chatml = tmp_path / "chatml"
    settings.paths.processed_alpaca = tmp_path / "alpaca"
    settings.paths.processed_sharegpt = tmp_path / "sharegpt"
    settings.paths.interim = tmp_path / "interim"

    manifest = build(
        plan=[{"id": "galsenai/french-wolof-translation"}],
        loader=FakeLoader(FAKE_DATA),
        settings=settings,
        version="mem",
    )
    assert manifest.peak_rss_mb > 0
    assert manifest.partial is False


def test_checksums_identiques_au_fichier_ecrit(tmp_path):
    """Le checksum calculé au vol doit égaler le sha256 du fichier final."""
    from galsenai_sft.build import build
    from galsenai_sft.core import config as cfg_mod
    from galsenai_sft.core.io import sha256_file

    settings = cfg_mod.get_settings()
    settings.paths.processed_chatml = tmp_path / "chatml"
    settings.paths.processed_alpaca = tmp_path / "alpaca"
    settings.paths.processed_sharegpt = tmp_path / "sharegpt"
    settings.paths.interim = tmp_path / "interim"

    manifest = build(
        plan=[{"id": "galsenai/french-wolof-translation"}],
        loader=FakeLoader(FAKE_DATA),
        settings=settings,
        version="sum",
    )
    for rel, checksum in manifest.outputs.items():
        path = Path(rel)
        if not path.is_absolute():
            from galsenai_sft.core.config import REPO_ROOT

            path = REPO_ROOT / rel
        assert sha256_file(path) == checksum, rel
