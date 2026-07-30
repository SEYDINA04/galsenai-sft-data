# 🔄 Flux de données — galsenai-sft-data

Du dataset brut jusqu'au fine-tuning, étape par étape.

```
 Dataset brut (HF)
      │   Loader.load(dataset_id, split, config)
      ▼
 Lignes brutes (dict)                         data/raw/
      │   Converter.convert_row()  ── @register (plugin)
      ▼
 Sample[] (ChatML canonique typé)             data/interim/
      │   filter_quality()  (doublons, vides, longueurs)
      ▼
 Sample[] nettoyés
      │   decontaminate()  (anti-fuite vs corpus de pré-entraînement) [optionnel]
      ▼
 Sample[] validés
      │
      ├── [datasets externes] Translator → QE (LID) → file de revue humaine → approuvés
      │                                                        (étape 3, moteur différé)
      │
      │   Exporters (chatml / alpaca / sharegpt)
      ▼
 data/processed/chatml/{task}.jsonl + all.jsonl
 data/processed/alpaca/all.jsonl
 data/processed/sharegpt/all.jsonl
      │   build.py → statistics + manifest (checksums)
      ▼
 Build manifest + stats                       data/interim/
      │   publish.py → data card + upload (dry-run par défaut)
      ▼
 HF: galsenai/wolof_sft
      │
      ▼
 Équipe Fine-Tuning (Marième, Sophie, Mohamed)
```

## Détail des étapes

1. **Chargement** — `Loader` (HFLoader réel, ou injecté en test). Sépare l'accès
   réseau du reste → pipeline testable hors ligne.
2. **Conversion** — le `Converter` du dataset (trouvé via le registry) transforme
   chaque ligne en `Sample` ChatML. Consignes **bilingues wo/fr** (seed
   déterministe).
3. **Validation qualité** — retire doublons exacts (empreinte normalisée),
   réponses vides, échos question→réponse ; signale les longueurs anormales.
4. **Décontamination** *(optionnelle)* — retire les exemples déjà présents dans
   le corpus de **pré-entraînement** (fuite d'évaluation). Index de hash chargé
   en streaming depuis parquet/jsonl.
5. **Traduction** *(datasets externes, étape 3 différée)* — backend pluggable →
   estimation de qualité (LID GlotLID v3) → file de revue humaine → seuls les
   items **approuvés** entrent dans le dataset.
6. **Export** — sérialisation vers ChatML (défaut), Alpaca, ShareGPT.
7. **Build** — agrège, écrit un fichier par tâche + global, calcule stats et
   **manifest** (checksums, versions).
8. **Publication** — génère la data card, pousse sur HF (dry-run par défaut,
   `--execute` pour publier réellement).

## Commandes correspondantes

```bash
galsenai-sft converters                 # datasets disponibles
galsenai-sft convert <id> -o out.jsonl  # 1 dataset → Samples
galsenai-sft validate out.jsonl         # contrôle qualité
galsenai-sft stats out.jsonl            # statistiques
galsenai-sft export out.jsonl --to alpaca -o alpaca.jsonl
galsenai-sft build                      # build complet (tous les datasets)
galsenai-sft publish                    # dry-run (ajouter --execute pour publier)
galsenai-sft catalog                    # régénère docs/dataset_catalog.md
```
