# 🗂️ galsenai-sft-data

Plateforme de préparation de **datasets d'instruction (SFT) en wolof** pour le
projet **GalsenAI LLM**.

Elle collecte des datasets wolof existants, les convertit dans un **format
canonique unique (ChatML)**, et prépare la traduction de datasets d'instruction
externes vers le wolof — le tout de façon modulaire, testable et reproductible.

> Pipeline : `raw → validation → conversion → ChatML → quality checks →
> traduction (si besoin) → review → versioning → export → fine-tuning`

## Capacités ciblées (façon LFM 2.5)

- **Data extraction** : NER, Information Retrieval, Intent Classification, QA
- **Tool use** (agentique) : function calling

## Format canonique : ChatML

La représentation interne est un `Sample` typé (pydantic) — une conversation
ChatML. Les formats **Alpaca** et **ShareGPT** sont produits par des exporters
dédiés. On ne stocke jamais de ChatML brut : tout est validable et transformable.

## Architecture

```
src/galsenai_sft/
├── core/         schéma canonique (Sample/Message), config, logging, io
├── registry.py   système de plugins (Open/Closed)
├── converters/   1 converter par dataset/tâche (translation, intent, ner, qa, …)
├── validators/   schéma, qualité, LID (GlotLID v3), décontamination, stats
├── translators/  interface pluggable + QE + review (étape 3, différée)
├── exporters/    chatml · alpaca · sharegpt
├── metadata/     registre + catalogue auto
└── cli.py        `galsenai-sft`
```

## Ajouter un dataset (principe Open/Closed)

Écrire **un** converter, sans modifier le pipeline :

```python
from galsenai_sft.converters.translation.base_translation import TranslationConverter
from galsenai_sft.registry import register

@register("mon-org/mon-dataset")
class MonDataset(TranslationConverter):
    source_col, source_lang = "french", "fr"
    target_col, target_lang = "wolof", "wo"
    bidirectional = True
```

## Démarrage

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest -q          # tests
uv run galsenai-sft --help
```

## Détection de langue

**GlotLID v3 épinglé** (`model_v3.bin`) derrière une interface pluggable — jamais
le pointeur mouvant `model.bin`, pour garantir la reproductibilité.

## Licence

Code sous licence **MIT**. Chaque dataset source conserve sa licence d'origine
(voir `metadata/licenses.yaml`).

---

_Projet GalsenAI LLM — Data Team (Babacar Ndao)._
