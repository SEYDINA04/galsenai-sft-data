# 🎯 Ciblage des jeux de données SFT

> Pourquoi **ces** sources et pas d'autres. Document de décision : le détail
> exhaustif (90+ datasets recensés, papiers, tables comparatives) reste dans
> `wolof_scraper/docs/SFT_DATASETS_SURVEY.md` et `SFT_TARGETING.md`.
>
> Les chiffres cités ici sont **mesurés**, pas déclarés : ils viennent de
> [`inventory.md`](inventory.md), régénérable par `galsenai-sft inventory`.

---

## 1. Méthodologie

Un modèle de 1 à 7 milliards de paramètres n'a pas la capacité de connaissance
d'un grand modèle sur le QA général ouvert — **mais il excelle sur les tâches
cadrées**. Le ciblage suit donc deux familles de capacités plutôt qu'une
recherche de volume :

| Famille | Tâches visées | État dans le dataset |
|---|---|---|
| **A — Data Extraction** | NER, intent, classification, QA | couvert, sauf retrieval |
| **B — Tool Use** (agentique) | function calling, décision d'appel/abstention | **couvert depuis la v0.2** (184 541 exemples avec appels réels) |

Hors périmètre v1 : QA général ouvert, mathématiques, benchmarks de code.

Trois constats de la littérature structurent les arbitrages :

- **Une pincée de wolof suffit.** 10³–10⁴ exemples wolof de qualité, mélangés à
  une base EN/FR forte, transfèrent l'instruction-following. Le volume brut
  n'est pas l'objectif — d'où le refus d'intégrer des sources bruitées
  simplement parce qu'elles sont grosses.
- **Vérifié bat volumineux** pour le tool use : 60 k exemples exécutés-vérifiés
  battent des jeux dix fois plus gros non vérifiés.
- **L'extraction est une tranche minoritaire** : un ratio ~20 % extraction /
  80 % général préserve les capacités générales.

---

## 2. Critères d'inclusion

Une source entre dans le build si elle satisfait les quatre conditions :

| # | Critère | Vérification |
|---|---|---|
| 1 | **Wolof natif ou aligné wolof** | manuelle, à la lecture du dataset |
| 2 | **Format structuré exploitable** (entrée → sortie étiquetée) | l'écriture d'un converter sert de preuve |
| 3 | **Qualité de la cible** | filtre LID `wol_Latn` quand la source est bruitée (`lid_filter` dans `configs/build.yaml`) |
| 4 | **Apport réel de volume** | la déduplication globale du build tranche : une source entièrement recouverte ne produit rien |

Une exception assumée au critère 1 : les jeux de *function calling* sont
**anglophones**. Ils n'apportent pas de wolof mais la capacité d'appeler un
outil, que le transfert multilingue propage. Ils sont étiquetés
`prompt_lang=en` et ne sont jamais comptés comme de la donnée wolof.

Le critère 2 explique un point qui surprend à la lecture du dépôt : la liste des
converters, celle du registre et celle du plan de build sont **identiques** (30).
Ce n'est pas un défaut de conception mais la conséquence du critère — écrire un
converter *est* l'acte d'inclusion. Le ciblage réel se lit donc dans ce que le
dépôt **n'a pas** intégré, d'où `metadata/candidates.yaml`.

> La licence n'est **pas** un critère d'inclusion : le projet est open source.
> `metadata/datasets_registry.yaml` continue de la tracer à titre informatif
> (`license_status`, `commercial_ok`), mais aucune décision de périmètre ne s'y
> appuie.

---

## 3. Ce qui est intégré (30 sources)

Volumes de **lignes source** mesurés par `galsenai-sft inventory` ; le détail
par licence est dans [`dataset_catalog.md`](dataset_catalog.md).

| Tâche | Sources | Lignes source | Rôle |
|---|---:|---:|---|
| translation | 9 | 305 366 | capacité pivot fr/en ↔ wo ; corpus partiellement recouvrants |
| tool_use | 5 | 324 479 | **function calling réel** (Toucan, Hermes, ToolACE, When2Call) + Q/R de code |
| classification | 4 | 642 521 | sentiment, émotion, thématique, inférence (NLI) — **plafonné au build** |
| instruction | 4 | 116 304 | instructions wolof (WORI natif, Alpaca traduit, Aya) |
| intent | 2 | 14 072 | ancre wolof gold + domaine bancaire |
| ner | 2 | 5 638 | MasakhaNER 2.0 (gold) + entity linking |
| qa | 4 | 2 153 | QA ouvert, compréhension de lecture, QCM, maths |
| retrieval | 0 | 0 | **capacité absente** |

---

## 4. Ce qui est écarté, et pourquoi

Chaque exclusion est tracée dans `metadata/candidates.yaml` et mesurée par
l'inventaire. Les motifs sont désormais techniques, non juridiques : le projet
est open source, la licence d'une source n'est plus un critère de blocage.

### 4.1 Écarté sur décision

| Dataset | Tâche | Lignes | Motif |
|---|---|---:|---|
| `vonewman/alpaca-dataset-wolof` | instruction | 47 463 | Reconditionnement d'`alpaca-data-in-wolof` (intégré) au format Alpaca — même compte exact. |
| `vonewman/alpaca-sharegpt-wolof` | instruction | 47 463 | Même contenu, format ShareGPT. |
| `vonewman/wolof-instruction-dataset-alpaca` | instruction | 3 | Dépôt quasi vide (parquet de 9 Ko). |

Aucune de ces exclusions ne perd de contenu : ce sont des rediffusions de
sources déjà intégrées. La déduplication globale les écarterait de toute façon,
mais autant ne pas payer le téléchargement.

### 4.2 Ciblé, converter non écrit

| Dataset | Tâche | Lignes | Blocage |
|---|---|---:|---|
| `Salesforce/xlam-function-calling-60k` | tool_use | ~60 000 | dépôt *gated* : accepter les conditions à la main |
| `Salesforce/APIGen-MT-5k` | tool_use | 5 000 | intégrable sans travail de schéma (multi-tours supporté) |
| `miracl/miracl` | retrieval | — | script de chargement (API taille en 501) |
| `masakhane/masakhaner` / `-x` | ner | ~3 300 | recouvrement quasi total avec MasakhaNER 2.0 intégré |
| `AmazonScience/massive` | intent | 11 514 | aucune variante wolof — exige une localisation, pas une intégration |

### 4.3 Ce qui reste à zéro

**Retrieval.** `TaskType.RETRIEVAL` existe dans le schéma, aucun converter ne
l'alimente. Il n'existe pas de corpus IR wolof : la voie est la synthèse sur le
corpus de pré-entraînement (protocole SWIM-IR), pas l'intégration.

---

## 5. Plafonds réels par tâche

L'objectif « amener chaque tâche au niveau de la classification » (640 964
exemples en v0.1) se heurte à une contrainte de disponibilité mondiale, pas
d'ingénierie. Volumes atteignables en intégrant **tout** le wolof existant :

| Tâche | Atteint en v0.2 | Plafond mondial estimé | Verdict |
|---|---:|---:|---|
| translation | 573 832 | ~640 000 | **cible quasi atteinte** |
| tool_use | 206 557 | ~500 000 (sources EN) | atteignable |
| instruction | 60 644 | ~120 000 | ×5 sous la cible |
| intent | 15 012 | ~15 000 | **plafond atteint** |
| ner | 5 638 | ~6 500 | **plafond atteint** — ×100 sous la cible |
| qa | 2 152 | ~3 000 | **plafond atteint** — ×200 sous la cible |
| retrieval | 0 | 0 | inexistant |

Pour NER, QA, intent et retrieval, **aucune recherche de sources ne comblera
l'écart** : les données n'existent pas. Les seules voies sont la synthèse sur
le corpus wolof (SWIM-IR pour l'IR, Pile-NER distillé pour le NER) et la
localisation de jeux EN/FR (protocole INJONGO pour l'intent) — c'est de la
production de données, pas du sourcing.

C'est pourquoi l'équilibrage de la v0.2 passe aussi par le **plafonnement de la
classification** (`max_samples: 60000` sur les deux corpus dominants) : rapprocher
les tâches par le haut est impossible, on les rapproche donc aussi par le bas.

---

## 6. Rejouer le ciblage

```bash
galsenai-sft inventory   # mesure les volumes à la source (aucun téléchargement)
galsenai-sft catalog     # régénère docs/dataset_catalog.md avec ces volumes
```

Pour modifier le périmètre : ajouter/retirer une ligne dans
`configs/build.yaml` (inclusion) ou dans `metadata/candidates.yaml`
(exclusion motivée). Toute exclusion doit porter un motif — un test le
vérifie (`tests/test_inventory.py`).
