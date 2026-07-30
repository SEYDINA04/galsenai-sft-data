"""Tests du HFLoader — fallback parquet quand un dataset utilise un script."""

from __future__ import annotations

import pytest

from galsenai_sft import loaders
from galsenai_sft.loaders import HFLoader


def test_hfloader_fallback_to_parquet(monkeypatch):
    """Sur erreur 'script no longer supported', bascule sur les URLs parquet."""
    calls = {"n": 0}

    def fake_load_dataset(*args, **kwargs):
        # 1er appel = chargement direct -> échoue (script)
        # 2e appel = load_dataset("parquet", data_files=...) -> réussit
        if args and args[0] == "parquet":
            assert kwargs["data_files"] == ["http://x/train.parquet"]
            return [{"question": "Q", "answers": "['R']"}]
        calls["n"] += 1
        raise RuntimeError("Dataset scripts are no longer supported, but found afriqa.py")

    monkeypatch.setattr(loaders, "_parquet_urls", lambda *a, **k: ["http://x/train.parquet"])

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    rows = list(HFLoader().load("masakhane/afriqa", split="train", config="wol"))
    assert rows == [{"question": "Q", "answers": "['R']"}]


def test_hfloader_reraises_other_errors(monkeypatch):
    """Une erreur non liée au script est propagée telle quelle."""
    import datasets

    def boom(*a, **k):
        raise ValueError("connexion perdue")

    monkeypatch.setattr(datasets, "load_dataset", boom)
    with pytest.raises(ValueError, match="connexion perdue"):
        HFLoader().load("x/y", split="train")


def test_hfloader_no_parquet_found(monkeypatch):
    import datasets

    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("scripts are no longer supported")),
    )
    monkeypatch.setattr(loaders, "_parquet_urls", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="aucun parquet"):
        HFLoader().load("x/y", split="train", config="wol")
