"""Tests du registre de métadonnées et de la génération du catalogue."""

from __future__ import annotations

from galsenai_sft.metadata import LicenseStatus, load_registry, render_catalog


def test_registry_covers_all_converters():
    from galsenai_sft.registry import available

    reg = load_registry()
    assert set(reg) == set(available())


def test_task_comes_from_converter():
    reg = load_registry()
    assert reg["masakhane/InjongoIntent"].task == "intent"
    assert reg["mbaye930/WolofEntityLinking"].task == "ner"
    assert reg["michsethowusu/Code-170k-wolof"].task == "tool_use"


def test_license_metadata_loaded():
    reg = load_registry()
    injongo = reg["masakhane/InjongoIntent"]
    assert injongo.license_status is LicenseStatus.PERMISSIVE
    assert injongo.commercial_ok is True
    afriqa = reg["masakhane/afriqa"]
    assert afriqa.license_status is LicenseStatus.NON_COMMERCIAL


def test_render_catalog_contains_datasets():
    reg = load_registry()
    md = render_catalog(reg)
    assert "Catalogue des datasets" in md
    assert "masakhane/InjongoIntent" in md
    assert "Conformité licences" in md
