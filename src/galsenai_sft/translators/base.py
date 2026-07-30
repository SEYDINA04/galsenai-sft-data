"""Interface de traduction (backend-agnostique) + adaptateurs.

Conçu comme un :class:`Protocol` pour brancher n'importe quel moteur (LLM
frontière via API, Google Translate, NLLB local…) sans toucher au pipeline —
principe Open/Closed. **Le moteur réel est différé** (décision projet) : seul un
``EchoTranslator`` (stub, pour tests/démo) est fourni pour l'instant.

Stratégie cible (cf. survey) : pivot **français**, localisation d'entités,
puis QE + revue humaine. Le pipeline (``pipeline.py``) est prêt à recevoir un
vrai backend quand une clé API/un budget seront décidés.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from galsenai_sft.core.logging import get_logger

log = get_logger(__name__)


class TranslationRequest(BaseModel):
    text: str
    source_lang: str  # ex. "fr", "en"
    target_lang: str = "wo"


class TranslationResult(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    translated: str
    engine: str


@runtime_checkable
class Translator(Protocol):
    """Contrat d'un backend de traduction."""

    name: str

    def translate(self, req: TranslationRequest) -> TranslationResult:
        """Traduit un texte. Doit être idempotent pour un même input."""
        ...


class EchoTranslator:
    """Backend factice : renvoie le texte inchangé (tests/démo, PAS de traduction).

    Sert à valider le pipeline (batch, cache, QE, review) sans dépendance réseau.
    Le vrai backend (LLM frontière pivot FR) sera ajouté au lot « étape 3 ».
    """

    name = "echo"

    def translate(self, req: TranslationRequest) -> TranslationResult:
        return TranslationResult(
            text=req.text,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            translated=req.text,
            engine=self.name,
        )


class CachingTranslator:
    """Décorateur : met en cache (disque JSONL) les traductions d'un backend.

    La traduction par LLM/API étant coûteuse, le cache garantit idempotence et
    reprise. Clé = hash(engine, source_lang, target_lang, text).
    """

    def __init__(self, backend: Translator, cache_path: str | Path) -> None:
        self.backend = backend
        self.name = backend.name
        self.cache_path = Path(cache_path)
        self._cache: dict[str, str] = {}
        self._load()

    def _key(self, req: TranslationRequest) -> str:
        raw = f"{self.backend.name}|{req.source_lang}|{req.target_lang}|{req.text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if self.cache_path.exists():
            with self.cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        self._cache[obj["key"]] = obj["translated"]
            log.info("cache traduction : %d entrées", len(self._cache))

    def _append(self, key: str, translated: str) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "translated": translated}, ensure_ascii=False) + "\n")

    def translate(self, req: TranslationRequest) -> TranslationResult:
        key = self._key(req)
        if key in self._cache:
            return TranslationResult(
                text=req.text,
                source_lang=req.source_lang,
                target_lang=req.target_lang,
                translated=self._cache[key],
                engine=self.backend.name + "+cache",
            )
        result = self.backend.translate(req)
        self._cache[key] = result.translated
        self._append(key, result.translated)
        return result
