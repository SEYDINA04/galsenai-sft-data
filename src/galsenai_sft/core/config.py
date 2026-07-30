"""Configuration centrale de la plateforme.

Chargée depuis ``configs/settings.yaml`` (surchargée par variables d'env et
``.env``). Aucune valeur de chemin/seuil n'est codée en dur ailleurs.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Racine du dépôt (…/galsenai-sft-data).
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SETTINGS = REPO_ROOT / "configs" / "settings.yaml"


class Paths(BaseModel):
    raw_wolof: Path = REPO_ROOT / "data" / "raw" / "wolof"
    raw_external: Path = REPO_ROOT / "data" / "raw" / "external"
    interim: Path = REPO_ROOT / "data" / "interim"
    processed_chatml: Path = REPO_ROOT / "data" / "processed" / "chatml"
    processed_alpaca: Path = REPO_ROOT / "data" / "processed" / "alpaca"
    processed_sharegpt: Path = REPO_ROOT / "data" / "processed" / "sharegpt"
    datasets_registry: Path = REPO_ROOT / "metadata" / "datasets_registry.yaml"
    licenses: Path = REPO_ROOT / "metadata" / "licenses.yaml"
    dataset_configs: Path = REPO_ROOT / "configs" / "datasets"


class LIDConfig(BaseModel):
    """Détection de langue — GlotLID v3 épinglé (reproductibilité)."""

    repo_id: str = "cis-lmu/glotlid"
    filename: str = "model_v3.bin"  # épinglé : jamais 'model.bin' (pointeur mouvant)
    target_label: str = "wol_Latn"
    threshold: float = 0.5


class QualityConfig(BaseModel):
    min_chars: int = 1
    max_chars: int = 50_000
    drop_duplicates: bool = True


class HFConfig(BaseModel):
    repo: str = "galsenai/wolof_sft"
    repo_type: str = "dataset"
    token_env: str = "HF_TOKEN"


class MemoryConfig(BaseModel):
    """Garde-fou mémoire du build (voir :mod:`galsenai_sft.core.memory`)."""

    #: Plancher de mémoire système disponible (Mo). 0 = surveillance désactivée.
    min_available_mb: float = 1536.0
    #: Plafond de RSS du processus (Mo). None = pas de plafond propre.
    max_rss_mb: float | None = None
    #: Période d'échantillonnage du surveillant (s).
    interval_s: float = 2.0


class BuildConfig(BaseModel):
    """Comportement du builder."""

    #: Streaming HuggingFace : lit les datasets par morceaux au lieu de les
    #: télécharger intégralement (indispensable sur les gros datasets).
    streaming: bool = True
    #: Taille des lots lus dans les parquets distants.
    batch_size: int = 1000
    #: Fréquence des lignes de progression (en exemples écrits).
    log_every: int = 5000


class Settings(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    lid: LIDConfig = Field(default_factory=LIDConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    hf: HFConfig = Field(default_factory=HFConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    build: BuildConfig = Field(default_factory=BuildConfig)
    # Corpus de pré-entraînement pour la décontamination (chemins parquet/jsonl).
    pretraining_corpus_paths: list[Path] = Field(default_factory=list)

    @property
    def hf_token(self) -> str | None:
        return os.environ.get(self.hf.token_env)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=1)
def get_settings(path: Path | None = None) -> Settings:
    """Charge la configuration (fichier YAML optionnel + ``.env``)."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    settings_path = path or DEFAULT_SETTINGS
    data: dict[str, Any] = {}
    if settings_path.exists():
        data = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(data)
