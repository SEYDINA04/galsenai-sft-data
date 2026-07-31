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
| **A — Data Extraction** | NER, intent, classification, QA/retrieval | partiellement couvert |
| **B — Tool Use** (agentique) | function calling, décision d'appel/abstention | **non couvert** (voir §4) |

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
| 3 | **Licence tracée** | `metadata/datasets_registry.yaml` (`license_status`, `commercial_ok`) |
| 4 | **Qualité de la cible** | filtre LID `wol_Latn` quand la source est bruitée (`lid_filter` dans `configs/build.yaml`) |

Le critère 2 explique un point qui surprend à la lecture du dépôt : la liste des
converters, celle du registre et celle du plan de build sont **identiques** (11).
Ce n'est pas un défaut de conception mais la conséquence du critère — écrire un
converter *est* l'acte d'inclusion. Le ciblage réel se lit donc dans ce que le
dépôt **n'a pas** intégré, d'où `metadata/candidates.yaml`.

> ⚠️ Ces critères sont appliqués **par une personne**, pas par le code. Aucun
> garde-fou n'empêche aujourd'hui d'ajouter au plan de build une source en
> licence non commerciale : `commercial_ok` est descriptif, jamais lu par
> `build.py`. C'est une limite connue, pas un oubli.

---

## 3. Ce qui est intégré (11 sources)

Volumes mesurés à la source ; le détail par licence est dans
[`dataset_catalog.md`](dataset_catalog.md).

| Tâche | Sources | Lignes source | Rôle |
|---|---|---:|---|
| classification | `michsethowusu/wolof-sentiments-corpus`, `michsethowusu/wolof-emotions-corpus`, `Davlan/sib200` | 641 921 | volume ; **surreprésenté** (voir §5) |
| tool_use | `michsethowusu/Code-170k-wolof` | 176 999 | format conversationnel, **pas** du tool calling |
| translation | `galsenai/french-wolof-translation`, `bilalfaye/english-wolof-french-dataset` | 32 447 | capacité pivot fr↔wo |
| intent | `karim155/WolBanking77`, `masakhane/InjongoIntent` | 14 072 | ancre wolof gold, domaine bancaire |
| instruction | `m-a-d-i/wori-wolof-instructions` | 3 724 | instructions wolof natives (reverse-instructions) |
| ner | `mbaye930/WolofEntityLinking` | 1 045 | seule source NER retenue |
| qa | `masakhane/afriqa` | 503 | QA wolof à pivot français |

---

## 4. Ce qui est écarté, et pourquoi

C'est la partie que le dépôt ne portait pas jusqu'ici. Chaque exclusion est
tracée dans `metadata/candidates.yaml` et mesurée par l'inventaire.

### 4.1 Écarté sur décision

| Dataset | Tâche | Lignes | Motif |
|---|---|---:|---|
| `masakhane/masakhaner2` (wol) | ner | 4 593 | **CC BY-NC.** Seul NER wolof gold annoté humainement — 4,4 fois notre source NER actuelle. Incompatible avec un jeu commercial ; réservé à un futur track recherche. |
| `ngia/alpaca-data-in-wolof` | instruction | 47 463 | **Traduction automatique non auditée.** 12,7 fois notre volume d'instructions. Le ciblage interdit la MT non vérifiée comme donnée gold. Réintégrable après audit LID + qualité. |
| `Salesforce/APIGen-MT-5k` | tool_use | 5 000 | CC BY-NC, même règle que MasakhaNER 2.0. |

Ces trois exclusions représentent **57 056 lignes**, dont 52 056 en wolof — soit
plus que l'ensemble de nos tâches instruction + NER + QA + intent réunies. Ce
n'est pas un détail : c'est le prix payé pour rester licence-propre et
qualité-propre.

### 4.2 Ciblé, converter non écrit

| Dataset | Tâche | Lignes | Priorité |
|---|---|---:|---|
| `Agent-Ark/Toucan-1.5M` (SFT) | tool_use | 119 287 | P0 — Apache-2.0, multi-tours réellement exécutés |
| `nvidia/When2Call` (train_sft) | tool_use | 15 000 | P1 — décision d'appel / abstention |
| `Team-ACE/ToolACE` | tool_use | 11 300 | P1 — diversité de schémas d'API |
| `NousResearch/hermes-function-calling-v1` | tool_use | 1 893 | P0 — verrouille le format `<tool_call>` |
| `Salesforce/xlam-function-calling-60k` | tool_use | ~60 000 | P0 — CC BY 4.0 mais dépôt *gated* |
| `miracl/miracl` | retrieval | — | P1 — l'API taille ne répond pas (501) |

### 4.3 Deux trous à assumer

**La famille B n'existe pas dans le dataset.** La tâche `tool_use` pèse 19,5 %
des exemples produits, mais aucun n'est un appel d'outil : `Code-170k-wolof` est
du Q/R de code traduit, tous les exemples font exactement deux messages et aucun
ne porte de `tool_calls`. Le nom de la tâche décrit une intention, pas un
contenu. Les sources du §4.2 sont la trajectoire de rattrapage.

**La capacité retrieval est à zéro.** `TaskType.RETRIEVAL` existe dans le schéma
(`core/schema.py`) sans aucun converter pour l'alimenter.

---

## 5. Conséquence connue : le déséquilibre

La classification représente 641 921 lignes source sur 870 711, soit **70,8 %
des exemples produits**. Ce n'est pas le résultat d'un choix de ciblage mais de
la disponibilité : les deux corpus de sentiments et d'émotions sont, de loin,
les plus gros jeux wolof étiquetés disponibles.

Deux leviers existent et **ne sont pas activés** :

- `max_samples` par entrée dans `configs/build.yaml` — plafonner ces deux
  sources rééquilibrerait le mélange à la construction ;
- pondération des tâches à l'entraînement — décision qui appartient à l'équipe
  fine-tuning, pas à ce dépôt.

Le choix actuel est de **livrer le volume brut et de documenter le déséquilibre**
plutôt que de trancher à la place du consommateur. Ce choix est signalé dans la
data card du dataset publié.

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
