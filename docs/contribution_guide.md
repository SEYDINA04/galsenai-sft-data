# 🤝 Guide de contribution — galsenai-sft-data

Comment ajouter un dataset, un format, ou un backend — sans casser l'existant.

## Ajouter un dataset (le cas courant)

Grâce au principe **Open/Closed**, ajouter un dataset = écrire **un converter**.
Aucune modification du pipeline.

### 1. Écrire le converter

Créer un fichier dans `src/galsenai_sft/converters/<tâche>/`.

**Cas traduction** (corpus parallèle) — hériter du patron :

```python
from galsenai_sft.converters.translation.base_translation import TranslationConverter
from galsenai_sft.registry import register

@register("mon-org/mon-corpus")
class MonCorpus(TranslationConverter):
    source_col, source_lang = "french", "fr"
    target_col, target_lang = "wolof", "wo"
    bidirectional = True
```

**Cas générique** (autre tâche) — hériter de `BaseConverter` :

```python
from typing import Any, ClassVar
from galsenai_sft.converters.base import BaseConverter
from galsenai_sft.core.schema import Message, Role, Sample, TaskType
from galsenai_sft.registry import register

@register("mon-org/mon-dataset")
class MonDataset(BaseConverter):
    task: ClassVar[TaskType] = TaskType.QA

    def convert_row(self, row: dict[str, Any]) -> list[Sample]:
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            return []
        lang = self.pick_lang()  # wo ou fr (déterministe)
        return [self.make_sample(
            [Message(role=Role.USER, content=f"…{question}"),
             Message(role=Role.ASSISTANT, content=answer)],
            prompt_lang=lang,
        )]
```

### 2. Déclarer les métadonnées

Ajouter une entrée dans `metadata/datasets_registry.yaml` (licence, URL, notes).

### 3. Ajouter au plan de build

Ajouter une ligne dans `configs/build.yaml` (`{ id, split, config }`).

### 4. Tester

Écrire un test dans `tests/` avec une ligne brute **fidèle au schéma réel**
(vérifier via l'API HF `datasets-server` : `/first-rows`).

```bash
uv run pytest -q
uv run galsenai-sft converters   # doit lister le nouveau dataset
```

## Vérifier le schéma réel d'un dataset

```bash
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://datasets-server.huggingface.co/first-rows?dataset=ORG/NAME&config=CONF&split=train"
```

## Règles de style

- **Consignes bilingues** wo/fr (le contenu cible reste wolof).
- **Schémas / clés JSON / types NER en anglais** ; seule la consigne est localisée.
- Types annotés partout ; docstrings en français.
- `ruff check` + `ruff format` doivent passer (hooks pre-commit).
- Ajouter des tests pour tout nouveau comportement.

## Ajouter un format d'export

Créer `exporters/<fmt>.py` avec une fonction `to_<fmt>(sample) -> dict`, puis
l'enregistrer dans `exporters/__init__.py` (`EXPORTERS`).

## Ajouter un backend LID ou de traduction

Implémenter le `Protocol` correspondant (`LanguageIdentifier` dans
`validators/lid.py`, `Translator` dans `translators/base.py`) et l'injecter —
aucune modification du pipeline.

## Workflow Git

- Hooks : `make hooks` (ruff au commit, pytest au push).
- Commits conventionnels (`feat:`, `fix:`, `docs:`…).
- Ne jamais committer de données (`data/` est ignoré) ni de secrets.
