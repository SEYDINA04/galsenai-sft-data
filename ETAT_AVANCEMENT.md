# 📊 État d'avancement — Plateforme SFT wolof (`galsenai-sft-data`)

> **But de ce fichier** : reprendre le travail rapidement. Lire ce fichier, puis
> aller à la section **« PROCHAINE ÉTAPE »**.

_Dernière mise à jour : 2026-07-31 (nuit) · Auteur : Babacar Ndao_
_Dépôt : https://github.com/SEYDINA04/galsenai-sft-data (public, branche `main`)_

---

## 0. ⚠️ Incident mémoire du 30/07 — corrigé (lot 8)

La machine a **gelé à 17:15** pendant un build (thrashing mémoire, aucun
processus tué par l'OOM-killer, redémarrage forcé). Cause : le builder gardait
**tous les exemples en RAM** (344 000 objets pydantic pour Code-170k) pendant que
`datasets` téléchargeait les corpus en entier (cache HF : 14 Go).

**Corrigé à trois niveaux** :

1. build **100 % en flux** — écriture au fil de l'eau, statistiques
   incrémentales, checksums calculés à l'écriture, LID libéré après usage
   (2 010 Mo → 215 Mo mesurés) → **RSS constant ≈ 200 Mo** ;
2. **garde-fou** `MemoryGuard` : sous 1,5 Go de RAM libre, arrêt **propre**
   (fichiers fermés, manifest `partial: true`, données exploitables) ;
3. **plafond dur cgroup** : `make build` isole le build (`MemoryMax`, swap
   interdit) → le noyau tue le build, **jamais la session graphique**.

Streaming HuggingFace activé par défaut : plus aucun téléchargement intégral.

---

## 1. Résumé en 30 secondes

La **plateforme complète de préparation de datasets SFT wolof** est construite,
testée et poussée sur GitHub. Elle fait les 3 objectifs du brief :

1. **Collecter** des datasets wolof existants (NER, intent, traduction, QA,
   classification, tool use).
2. **Convertir** chacun en un **format canonique unique (ChatML)**.
3. **Traduire** des datasets externes vers le wolof (interface prête, moteur
   réel volontairement différé).

**Les 7 lots (0 → 7) sont terminés**, plus un **lot 8 « mémoire »** après
l'incident du 30/07. 133 tests passent. Le build a été vérifié sur données
réelles.

---

## 2. Ce qui est fait (lot par lot)

| Lot | Contenu | Statut |
|---|---|---|
| 0 | Squelette : pyproject (uv), ruff, pytest, pre-commit, CI, `core/` (schéma ChatML typé, config, logging, io) | ✅ |
| 1 | Registry plugin (Open/Closed) + `BaseConverter` + exporters chatml/alpaca/sharegpt + CLI | ✅ |
| 2 | Validators : schéma, qualité (doublons/vides/longueurs), **LID GlotLID v3 épinglé**, décontamination, stats | ✅ |
| 3 | **converters wolof** (7 tâches ; 11 en v0.1, **30** en v0.2) : translation, intent(+slots), ner, classification, qa, instruction, tool_use | ✅ |
| 4 | Metadata : registre + **catalogue auto** (licences, statuts commercial/recherche) | ✅ |
| 5 | Translators : interface pluggable + QE (LID) + file de revue humaine (moteur réel **différé**) | ✅ |
| 6 | Builder end-to-end + **manifest** (checksums) + `publish` HF (dry-run) | ✅ |
| 7 | Docs : architecture, data_flow, contribution_guide, catalogue | ✅ |
| 8 | **Mémoire** : build en flux, `MemoryGuard`, plafond cgroup, streaming HF, libération du LID | ✅ |

**Datasets intégrés (30 en v0.2, 11 en v0.1)** : voir `docs/dataset_catalog.md`
et `docs/targeting.md`. Couverture : traduction (9), tool_use (5), classification (4),
instruction (4), qa (4), intent (2), ner (2).

**Commits poussés** : lots 0→7 (voir `git log`). Correctif annexe dans
`wolof_scraper` : GlotLID `model.bin` → `model_v3.bin` (déjà poussé).

---

## 2 bis. ✅ BUILD COMPLET RÉUSSI (30/07, 18:02 → 18:28)

Premier build réel des 11 datasets, sous plafond mémoire de 6 Go :

| Indicateur | Valeur |
|---|---|
| Exemples produits | **905 362** |
| Durée | 26 min |
| **Pic mémoire** | **2 050 Mo** (transitoire, dû au LID) — RSS de croisière ≈ 300 Mo |
| Datasets en échec | **0** |
| Sorties | `chatml/` 1,1 Go · `alpaca/` 483 Mo · `sharegpt/` 513 Mo |

Répartition par tâche :

| Tâche | Exemples | Part |
|---|---:|---:|
| classification | 640 964 | 70,8 % |
| tool_use | 176 723 | 19,5 % |
| translation | 64 832 | 7,2 % |
| intent | 15 012 | 1,7 % |
| instruction | 6 284 | 0,7 % |
| ner | 1 045 | 0,1 % |
| qa | 502 | 0,1 % |

La machine est restée pleinement utilisable pendant toute la durée du build.

⚠️ **Point à décider — déséquilibre.** Les corpus `wolof-sentiments-corpus` et
`wolof-emotions-corpus` pèsent à eux seuls **71 %** du dataset. Contrôle qualité
effectué sur 400 exemples de chacun : le texte est **bien du wolof** (GlotLID :
95 %) et peu bruité (2,2 %) — le problème n'est donc pas la qualité mais la
**proportion** : le modèle apprendrait surtout à produire des étiquettes courtes
de sentiment. Levier prêt : clé `max_samples` par dataset dans
`configs/build.yaml` (ex. `max_samples: 50000` sur ces deux entrées → dataset
d'environ 415 k exemples, bien plus équilibré). **À trancher avant publication.**

---

- **Format canonique = ChatML typé** (pydantic `Sample`), pas de texte brut →
  validable, transformable. Exporters Alpaca/ShareGPT dédiés.
- **Consignes bilingues wo/fr** ; contenu cible wolof ; **schémas/JSON/types NER
  en anglais** (seule la consigne est localisée).
- **GlotLID v3 conservé et épinglé** (`model_v3.bin`) après recherche comparative
  (AfroLID/ConLID/OpenLID-v3 écartés ou différés ; interface LID pluggable).
- **Traduction (étape 3) : moteur réel NON branché.** Interface + QE + review
  prêts ; `EchoTranslator` de test seulement. Raison : pas de clé API/budget
  décidé.
- **Publication HF** : exécutée le 30/07 en **privé** (`galsenai/wolof_sft`).
  Data card : licence `other` + tableau des licences par source — annoncer une
  licence unique aurait été faux (8 sources sur 11 non commerciales/non vérifiées).

---

## 3 bis. Décisions prises (suite)

- **Mémoire = contrainte d'architecture** : aucun flux de `Sample` ne doit être
  matérialisé dans le chemin de build (`src/galsenai_sft/build.py`).
- **Streaming HuggingFace par défaut** : plus aucun téléchargement intégral
  (le cache HF était monté à 14 Go).

---

## 4. 🎯 OBJECTIF v0.2 — rééquilibrer les tâches (fait, partiellement atteint)

**Objectif fixé** : amener chaque tâche au niveau de la classification
(640 964 exemples en v0.1), en intégrant de nouvelles sources.

### 4.1 Le constat qui change l'objectif

Atteindre 640 964 exemples pour chacune des six autres tâches demanderait
**3 581 386 exemples supplémentaires**. C'est impossible par sourcing : le
stock mondial de NER wolof gold est d'environ **6 500 lignes**, celui de QA
wolof d'environ **3 000**. Aucune recherche ne trouvera 640 000 exemples de QA
wolof — ils n'existent pas.

L'objectif a donc été tenu **par les deux bouts** : monter les tâches basses
autant que les sources le permettent, et **plafonner la classification**
(`max_samples: 60000` sur les deux corpus dominants). Rapprocher les tâches
uniquement par le haut était mathématiquement exclu.

### 4.2 Sources découvertes (recherche du 31/07)

Recensement programmatique du Hub HuggingFace (685 dépôts scannés) + recherche
documentaire. **19 nouvelles sources** intégrées, 11 → **30 converters** :

| Tâche | Sources ajoutées |
|---|---|
| translation (+7) | `galsenai/centralized_wolof_french_translation_data` (98 345), `sudoping01/english-wolof-translation` (84 709), `MaroneAI/*` (2 × 30 002), `dofbi/jolof` (12 084), `Alwaly/french-wolof-translation-gs` (10 372), `galsenai/english-wolof-smol-translation` (7 405) |
| instruction (+3) | `bilalfaye/wolof-sft` (61 971), `ngia/alpaca-data-in-wolof` (47 463), `CohereLabs/aya_collection_language_split` wolof (3 146) |
| tool_use (+4) | `Agent-Ark/Toucan-1.5M` (119 287), `nvidia/When2Call` (15 000), `Team-ACE/ToolACE` (11 300), `NousResearch/hermes-function-calling-v1` (1 893) |
| ner (+1) | `masakhane/masakhaner2` (4 593) — seul NER wolof gold |
| qa (+3) | `facebook/belebele` (900), `masakhane/afrimmlu` (500), `masakhane/afrimgsm` (250) |
| classification (+1) | `masakhane/afrixnli` (600) — inférence textuelle |

### 4.3 ✅ BUILD v0.2 (31/07, 01:04 → 02:03)

| Indicateur | v0.1 | v0.2 |
|---|---:|---:|
| Exemples | 905 362 | **985 136** |
| Datasets | 11 | **30** |
| Datasets en échec | 0 | **0** |
| Durée | 26 min | 59 min |
| Pic mémoire | 2 050 Mo | 2 227 Mo |
| Sortie ChatML | 1,1 Go | 2,4 Go |

| Tâche | v0.1 | v0.2 | Évolution | Part v0.2 |
|---|---:|---:|---:|---:|
| translation | 64 832 | **573 832** | ×8,9 | 58,2 % |
| tool_use | 176 723 | **206 557** | ×1,2 | 21,0 % |
| classification | 640 964 | 121 301 | plafonné | 12,3 % |
| instruction | 6 284 | **60 644** | ×9,6 | 6,2 % |
| intent | 15 012 | 15 012 | — | 1,5 % |
| ner | 1 045 | **5 638** | ×5,4 | 0,6 % |
| qa | 502 | **2 152** | ×4,3 | 0,2 % |

**Trois capacités qui n'existaient pas** apparaissent dans la v0.2 :

| Indicateur | v0.1 | v0.2 |
|---|---:|---:|
| Exemples avec `tool_calls` réels | **0** | **184 541** |
| Conversations multi-tours | **0** | **35 752** |
| Messages `system` (outils déclarés) | **0** | présents |

La tâche `tool_use` de la v0.1 ne contenait aucun appel d'outil — c'était du
questions/réponses de code. Elle en contient désormais de vrais, dans quatre
formats normalisés vers le schéma `ToolCall`.

### 4.4 Ce que la déduplication globale a révélé

La déduplication est passée de **locale à chaque dataset** à **globale au
build** (index partagé). Effet immédiat et mesurable :

- `ngia/alpaca-data-in-wolof` : 47 463 lignes → **12 284** exemples. Environ
  34 000 étaient déjà présents via `bilalfaye/wolof-sft`.
- `CohereLabs/aya_collection_language_split` : 3 146 → **266**. Le sous-ensemble
  wolof d'Aya est presque entièrement composé d'AfriQA, déjà intégré.
- Le filtre LID a écarté **20 037 lignes** de `bilalfaye/wolof-sft` (32 %) :
  ce corpus n'est pas majoritairement wolof.

Sans ces deux garde-fous, la v0.2 aurait annoncé ~55 000 exemples de plus,
tous faux.

### 4.5 ⚠️ Le déséquilibre a changé de camp

La classification est passée de 70,8 % à 12,3 %. Mais la **traduction occupe
maintenant 58,2 %** du dataset. On a remplacé un déséquilibre par un autre,
moins sévère (rapport max/min : 1277:1 → 266:1) mais réel.

Deux raisons de ne pas le corriger tout de suite :

1. l'objectif fixé était d'amener les tâches au niveau de la classification
   d'origine (640 964) — la traduction y est presque (573 832), c'est le
   résultat demandé ;
2. plafonner la traduction jetterait ~400 000 exemples qui viennent d'être
   acquis, sans que personne n'ait encore dit qu'ils gênent.

**Levier prêt, une ligne** : ajouter `max_samples` aux entrées de traduction
dans `configs/build.yaml` (à 60 000 chacune, la traduction retomberait à
~150 000 et le dataset serait équilibré à ±3 % près entre les cinq tâches
principales).

### 4.6 Reste à faire

1. **Splits train/validation/test** — toujours absents ; c'est le seul point
   qui empêche toute évaluation interprétable. Aggravé en v0.2 : `belebele`,
   `afrimmlu`, `afrimgsm` et `afrixnli` étaient des **jeux d'évaluation** et
   ont été versés dans `train` pour le volume. Il faut soit les en retirer,
   soit construire un jeu de test wolof natif indépendant.
2. **Retrieval** : capacité toujours à zéro. Aucune source ne l'apportera —
   passer par la synthèse sur le corpus de pré-entraînement (SWIM-IR).
3. **Intent** : plafond mondial atteint (~15 000). Seule voie : localiser
   MASSIVE (protocole INJONGO) — production de données, pas sourcing.
4. **`Salesforce/xlam-function-calling-60k`** : dépôt *gated*, accepter les
   conditions à la main puis relancer (+60 000 appels exécutés-vérifiés).
5. **Durcissement machine (demande `sudo`)** :
   ```bash
   sudo apt install earlyoom && sudo systemctl enable --now earlyoom
   ```
   `systemd-oomd` ne surveille pas les processus lancés depuis un terminal —
   c'est pourquoi rien n'a tué le build avant le gel du 30/07.
6. **Étape 3 (traduction externe)** : choisir un moteur + budget, puis brancher
   un backend `Translator`.

---

## 5. Commandes mémo

```bash
cd galsenai-sft-data
make setup                    # venv + install
make test                     # 133 tests
make doctor                   # état mémoire avant un gros build
make build-smoke              # build de test (100 lignes/dataset, plafond 4 Go)
make build                    # build complet SOUS PLAFOND MÉMOIRE  ← à utiliser
uv run galsenai-sft converters        # datasets disponibles
uv run galsenai-sft publish           # dry-run (data card)
uv run galsenai-sft publish --execute # publier sur HF (supervisé)
uv run galsenai-sft catalog           # régénère le catalogue
```

⚠️ Toujours passer par `make build` (ou `scripts/build_guarded.sh`) pour un gros
build : `uv run galsenai-sft build` fonctionne aussi mais sans plafond cgroup.

## 6. Fichiers utiles

| Élément | Chemin |
|---|---|
| Schéma canonique | `src/galsenai_sft/core/schema.py` |
| Ajouter un dataset | `docs/contribution_guide.md` |
| **Ciblage (retenus / écartés)** | `docs/targeting.md`, `metadata/candidates.yaml` |
| **Volume disponible par tâche** | `docs/inventory.md` (`galsenai-sft inventory`) |
| **Mémoire (protections)** | `docs/architecture.md` |
| Garde-fou mémoire | `src/galsenai_sft/core/memory.py` |
| Plan de build | `configs/build.yaml` |
| Registre licences | `metadata/datasets_registry.yaml` |
| Catalogue | `docs/dataset_catalog.md` |
| Architecture / flux | `docs/architecture.md`, `docs/data_flow.md` |
| Plan détaillé | `PLAN.md` |
