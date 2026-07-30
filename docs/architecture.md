# 🏛️ Architecture — galsenai-sft-data

Ce document décrit l'architecture logicielle de la plateforme et **justifie**
les décisions de conception.

## 1. Principes directeurs

| Principe | Mise en œuvre |
|---|---|
| **Modulaire** | Chaque responsabilité dans un sous-package (`core`, `converters`, `validators`, `translators`, `exporters`, `metadata`). |
| **Extensible (Open/Closed)** | Ajouter un dataset = 1 converter décoré `@register`, **sans** modifier le pipeline. Idem pour un backend LID ou de traduction (Protocols). |
| **Reproductible** | GlotLID **v3 épinglé** ; build **manifest** avec checksums SHA-256 ; seeds déterministes dans les converters. |
| **Testable** | Loader et backends injectables → tests sans réseau ni GPU. 49 tests unitaires. |
| **Typé / validable** | Tout passe par des modèles **pydantic** (`Sample`, `DatasetMeta`, `BuildManifest`…). |
| **Documenté** | Docstrings systématiques + `docs/` + catalogue auto-généré. |

## 2. Représentation canonique : pourquoi ChatML typé

On aurait pu stocker directement du texte ChatML, ou du Alpaca. Choix retenu :
un **modèle objet** `Sample` (liste de `Message` avec rôles + `ToolCall`).

- **Une seule vérité** : NER, traduction, intent, QA, multi-tours, tool use →
  tous représentés par le même objet. Les formats de sortie (ChatML, Alpaca,
  ShareGPT) sont des **sérialisations** produites par les exporters.
- **Validable** : l'ordre des rôles, les tours vides, l'UTF-8 sont vérifiés à la
  construction (pydantic) — impossible de produire un exemple structurellement
  invalide.
- **Transformable** : ajouter un format de sortie = 1 exporter, sans toucher aux
  converters.

Alternatives écartées : ChatML brut (non validable, difficile à transformer) ;
Alpaca comme format interne (incapable de représenter le multi-tours et le tool
use).

## 3. Vue des composants

```
                         ┌──────────────┐
   dataset HF (brut) ──▶ │   Loader     │  (HFLoader | injectable)
                         └──────┬───────┘
                                ▼
                         ┌──────────────┐   @register (plugin)
                         │  Converter   │◀── registry (découverte auto)
                         └──────┬───────┘
                                ▼  list[Sample]  (schéma canonique)
                         ┌──────────────┐
                         │  Validators  │  schéma · qualité · LID · décontam · stats
                         └──────┬───────┘
                                ▼
                  ┌─────────────┴──────────────┐
                  ▼                             ▼
          ┌──────────────┐            ┌──────────────────┐
          │  Exporters   │            │   Translators    │ (étape 3, différée)
          │ chatml/alpaca│            │ backend + QE +   │
          │  /sharegpt   │            │ review (pluggable)│
          └──────┬───────┘            └──────────────────┘
                 ▼
          ┌──────────────┐      ┌──────────────┐
          │   build.py   │────▶ │  publish.py  │──▶ HF galsenai/wolof_sft
          │ manifest+stats│     │ data card    │
          └──────────────┘      └──────────────┘
```

## 4. Rôle de chaque package

| Package | Responsabilité | Entrées | Sorties | Dépendances |
|---|---|---|---|---|
| `core` | Schéma canonique, config, logging, I/O | — | `Sample`, `Settings` | pydantic, pyyaml |
| `registry` | Découverte/enregistrement des converters | — | classes converter | — |
| `converters` | Ligne brute → `Sample` (1 par dataset/tâche) | dict | `list[Sample]` | core |
| `validators` | Contrôles + LID + décontamination + stats | `Sample` | `ValidationReport`, `Statistics` | fasttext*, pyarrow* |
| `exporters` | `Sample` → dict (chatml/alpaca/sharegpt) | `Sample` | dict | core |
| `translators` | Traduction pluggable + QE + review | texte | `ReviewItem` | (backend*) |
| `metadata` | Registre + catalogue | registry.yaml | `DatasetMeta`, catalogue.md | core |
| `loaders` | Chargement des datasets (injectable) | dataset_id | `Iterable[dict]` | datasets* |
| `build` | Orchestration end-to-end | plan | manifest + artefacts | tout |
| `publish` | Data card + upload HF | manifest | dataset HF | huggingface_hub* |

\* dépendances lourdes importées **paresseusement** (le package reste léger).

## 5. Gestion des erreurs & logging

- Le **build est robuste par dataset** : un dataset qui échoue est capturé dans
  le manifest (`error`) sans interrompre les autres.
- Logging structuré via **Rich** (`core.logging`), configuré une seule fois.
- Les validators renvoient des `Issue` (erreur vs avertissement) plutôt que de
  lever — on peut valider tout un lot et rapporter, façon quality gates.

## 6. Reproductibilité & versioning

- **Code + métadonnées** : Git.
- **Données** : ignorées par Git (structure conservée via `.gitkeep`) ;
  destinées à DVC ou aux révisions HF (plus tard).
- **Build manifest** (`data/interim/build_manifest.json`) : version, datasets,
  compteurs, checksums SHA-256 de chaque sortie → build rejouable et auditable.
- **GlotLID v3 épinglé** (`model_v3.bin`) : jamais le pointeur mouvant.
