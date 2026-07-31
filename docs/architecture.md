# 🏛️ Architecture — galsenai-sft-data

Ce document décrit l'architecture logicielle de la plateforme et **justifie**
les décisions de conception.

## 1. Principes directeurs

| Principe | Mise en œuvre |
|---|---|
| **Modulaire** | Chaque responsabilité dans un sous-package (`core`, `converters`, `validators`, `translators`, `exporters`, `metadata`). |
| **Extensible (Open/Closed)** | Ajouter un dataset = 1 converter décoré `@register`, **sans** modifier le pipeline. Idem pour un backend LID ou de traduction (Protocols). |
| **Reproductible** | GlotLID **v3 épinglé** ; build **manifest** avec checksums SHA-256 ; seeds déterministes dans les converters. |
| **Testable** | Loader et backends injectables → tests sans réseau ni GPU. 101 tests unitaires. |
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
   │ inventory.py │  API HF /size — volume disponible par tâche, AVANT build
   └──────┬───────┘  (configs/build.yaml + metadata/candidates.yaml)
          ▼
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
| `inventory` | Volume disponible par tâche **avant** build (sonde injectable) | plan + candidats | `Inventory`, inventory.md | — |
| `build` | Orchestration end-to-end | plan | manifest + artefacts | tout |
| `publish` | Data card + upload HF | manifest | dataset HF | huggingface_hub* |

\* dépendances lourdes importées **paresseusement** (le package reste léger).

## 5. Gestion des erreurs & logging

- Le **build est robuste par dataset** : un dataset qui échoue est capturé dans
  le manifest (`error`) sans interrompre les autres.
- Logging structuré via **Rich** (`core.logging`), configuré une seule fois.
- Les validators renvoient des `Issue` (erreur vs avertissement) plutôt que de
  lever — on peut valider tout un lot et rapporter, façon quality gates.

## 5 bis. Mémoire : une contrainte d'architecture, pas un réglage

Un build traite des centaines de milliers d'exemples sur un poste de 15 Go. La
mémoire est donc un **invariant de conception** (garde-fou :
[`core/memory.py`](../src/galsenai_sft/core/memory.py), plafond dur :
[`scripts/build_guarded.sh`](../scripts/build_guarded.sh)) :

| Décision | Effet |
|---|---|
| Pipeline **générateur** de bout en bout (`iter_entry_samples` → `SampleSink.write`) | Rien n'est accumulé : RSS **constant ≈ 180 Mo** quel que soit le volume. |
| **Checksums calculés à l'écriture** | Pas de seconde passe de lecture sur des fichiers multi-Go. |
| **Statistiques incrémentales** (`StatisticsAccumulator`) | Agrégation en O(1) mémoire. |
| **Streaming HF par défaut** + projection de colonnes parquet | Aucun téléchargement intégral ; colonnes audio jamais lues. |
| **Libération du LID** après le dernier dataset filtré | 2 010 Mo → 215 Mo, mesuré. |
| `MemoryGuard` (`core/memory.py`) | Arrêt **propre** sous pression : artefacts fermés, manifest `partial`. |
| `scripts/build_guarded.sh` (cgroup `MemoryMax`, swap 0) | Plafond **dur** : le noyau tue le build, jamais la session. |

Conséquence pour les contributeurs : **ne jamais matérialiser un flux de
`Sample`** (`list(...)`) dans le chemin de build. `convert_entry()` reste
disponible, mais réservé au debug et aux petits lots.

## 6. Reproductibilité & versioning

- **Code + métadonnées** : Git.
- **Données** : ignorées par Git (structure conservée via `.gitkeep`) ;
  destinées à DVC ou aux révisions HF (plus tard).
- **Build manifest** (`data/interim/build_manifest.json`) : version, datasets,
  compteurs, checksums SHA-256 de chaque sortie → build rejouable et auditable.
- **GlotLID v3 épinglé** (`model_v3.bin`) : jamais le pointeur mouvant.
