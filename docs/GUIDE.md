# 📘 Guide clair — Plateforme de datasets SFT wolof

> Document d'explication destiné au chef de projet et à l'équipe fine-tuning
> (Marième, Sophie, Mohamed). Aucun prérequis technique poussé.

---

## 1. C'est quoi, en une phrase ?

Une **usine à datasets d'instruction en wolof** : elle prend des jeux de données
wolof éparpillés (chacun avec son propre format), les transforme tous en **un
seul format standard (ChatML)**, et prépare des données prêtes à **entraîner
(fine-tuner) le LLM wolof**.

---

## 2. Le problème qu'elle résout

Aujourd'hui les données wolof existent mais sont **incompatibles entre elles** :
- un dataset de traduction a des colonnes `french` / `wolof` ;
- un dataset d'intention a des colonnes `text` / `intent` ;
- un dataset NER a des `entities` imbriquées ;
- un dataset de code a des `conversations`…

Impossible de tout donner tel quel à l'entraînement. **Il faut un format unique.**
C'est exactement ce que fait la plateforme.

---

## 3. Le format unique : ChatML

Chaque exemple devient une petite **conversation** :

```json
{
  "messages": [
    {"role": "user", "content": "Traduis en wolof : Bonjour, comment vas-tu ?"},
    {"role": "assistant", "content": "Salaamaalekum, naka nga def ?"}
  ],
  "task": "translation",
  "source": "galsenai/french-wolof-translation"
}
```

- `user` = la **consigne** (bilingue : wolof ou français).
- `assistant` = la **réponse attendue** (en wolof).
- Ce format marche pour **toutes** les tâches : traduction, NER, intention, QA,
  code, et même l'usage d'outils (tool use).

Si l'équipe préfère un autre format, la plateforme **exporte aussi** en Alpaca
et en ShareGPT — automatiquement.

---

## 4. Ce qu'elle sait déjà faire (11 datasets, 7 tâches)

| Tâche | Exemple de ce qui est produit |
|---|---|
| **Traduction** | « Traduis en wolof : … » → phrase wolof |
| **Intention** | « Quelle est l'intention de : … » → `alarm` (+ extraction des détails) |
| **NER** | « Extrais les entités : … » → `[{"text":"Dakar","type":"LOC"}]` |
| **Classification** | « Quel sentiment : … » → `Negative` |
| **Question-réponse** | « Réponds à : … » → réponse wolof |
| **Instruction** | consigne wolof → texte wolof |
| **Code / outils** | conversation assistant de programmation |

La liste complète (avec licences) est dans `docs/dataset_catalog.md`.

---

## 5. Comment on l'utilise (3 commandes)

```bash
# 1. Voir les datasets disponibles
galsenai-sft converters

# 2. Construire le dataset SFT complet (sous plafond mémoire : voir §6 bis)
make build

# 3. Le préparer pour HuggingFace (test d'abord, puis réel)
galsenai-sft publish              # aperçu (ne publie rien)
galsenai-sft publish --execute    # publie vraiment (supervisé)
```

Le résultat est écrit dans `data/processed/` : un fichier par tâche + un global,
dans les 3 formats.

---

## 6. Les garde-fous qualité (automatiques)

Avant qu'un exemple entre dans le dataset, il passe des contrôles :
- **doublons** exacts supprimés ;
- **réponses vides** rejetées ;
- **langue vérifiée** (GlotLID : c'est bien du wolof ?) ;
- **anti-triche** (décontamination) : on retire ce que le modèle a déjà vu en
  pré-entraînement, pour que l'évaluation reste honnête.

---

## 6 bis. Le garde-fou mémoire (pour ne plus jamais bloquer la machine)

Le 30/07/2026, un build a saturé la RAM et **la machine a gelé**. Corrigé, à
trois niveaux :

1. le build **écrit au fil de l'eau** et ne garde rien en mémoire — il consomme
   environ **180 Mo**, que l'on traite 1 000 ou 1 000 000 d'exemples ;
2. un **surveillant** arrête proprement le build si la mémoire libre s'effondre
   (les données déjà produites restent utilisables) ;
3. `make build` lance le build dans un **enclos mémoire** : en cas de problème,
   c'est le build qui s'arrête, **jamais l'ordinateur**.

```bash
make doctor        # état de la mémoire avant de lancer un gros build
make build-smoke   # essai rapide (100 lignes par dataset)
make build         # build complet, protégé
```

Explication complète : `docs/architecture.md` (section mémoire).

---

## 7. Et la traduction de datasets anglais vers le wolof ?

C'est le **3ᵉ objectif** du projet. L'ossature est **prête** (traduction par lots,
vérification qualité, file de **relecture humaine**), mais le **moteur de
traduction n'est pas encore branché** : il faut d'abord choisir l'outil (un grand
modèle type GPT/Claude via le français) et le budget. C'est une décision à
prendre ensemble.

---

## 8. Pourquoi c'est bien construit (pour durer 5 ans)

- **Ajouter un nouveau dataset = écrire ~15 lignes** (un « converter »), sans
  rien casser d'autre (principe Open/Closed).
- **Tout est testé** (68 tests automatiques) et vérifié à chaque modification (CI).
- **Reproductible** : chaque build produit un « manifest » avec des empreintes
  (checksums) — on peut refaire exactement le même dataset plus tard.
- **Documenté** : architecture, flux de données, guide de contribution.

---

## 9. Où en est-on ?

**Plateforme terminée** (7 lots sur 7). Il reste à :
1. lancer le build complet et le publier sur HuggingFace (avec ta validation) ;
2. brancher le moteur de traduction (étape 3) ;
3. livrer à l'équipe fine-tuning.

Détails dans `ETAT_AVANCEMENT.md` et `PLAN.md`.
