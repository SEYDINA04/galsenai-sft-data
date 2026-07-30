"""Tests du builder end-to-end (loader factice, sans réseau) + data card."""

from __future__ import annotations

from collections.abc import Iterable
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
