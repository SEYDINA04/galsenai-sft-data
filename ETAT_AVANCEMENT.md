# 📊 État d'avancement — Plateforme SFT wolof (`galsenai-sft-data`)

> **But de ce fichier** : reprendre le travail rapidement. Lire ce fichier, puis
> aller à la section **« PROCHAINE ÉTAPE »**.

_Dernière mise à jour : 2026-07-30 (soir) · Auteur : Babacar Ndao_
_Dépôt : https://github.com/SEYDINA04/galsenai-sft-data (public, branche `main`)_

---

## 0. ⚠️ Incident mémoire du 30/07 — corrigé (lot 8)

La machine a **gelé à 17:15** pendant un build (thrashing mémoire, aucun
processus tué par l'OOM-killer, redémarrage forcé). Cause : le builder gardait
**tous les exemples en RAM** (344 000 objets pydantic pour Code-170k) pendant que
`datasets` téléchargeait les corpus en entier (cache HF : 14 Go).

**Corrigé à trois niveaux** (détails : `docs/memoire.md`) :

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
l'incident du 30/07. 68 tests passent. Le build a été vérifié sur données
réelles.

---

## 2. Ce qui est fait (lot par lot)

| Lot | Contenu | Statut |
|---|---|---|
| 0 | Squelette : pyproject (uv), ruff, pytest, pre-commit, CI, `core/` (schéma ChatML typé, config, logging, io) | ✅ |
| 1 | Registry plugin (Open/Closed) + `BaseConverter` + exporters chatml/alpaca/sharegpt + CLI | ✅ |
| 2 | Validators : schéma, qualité (doublons/vides/longueurs), **LID GlotLID v3 épinglé**, décontamination, stats | ✅ |
| 3 | **11 converters wolof** (7 tâches) : translation, intent(+slots), ner, classification, qa, instruction, tool_use | ✅ |
| 4 | Metadata : registre + **catalogue auto** (licences, statuts commercial/recherche) | ✅ |
| 5 | Translators : interface pluggable + QE (LID) + file de revue humaine (moteur réel **différé**) | ✅ |
| 6 | Builder end-to-end + **manifest** (checksums) + `publish` HF (dry-run) | ✅ |
| 7 | Docs : architecture, data_flow, contribution_guide, catalogue | ✅ |
| 8 | **Mémoire** : build en flux, `MemoryGuard`, plafond cgroup, streaming HF, libération du LID | ✅ |

**Datasets intégrés (11)** : voir `docs/dataset_catalog.md`. Couverture :
traduction (2), intent (2 : INJONGO+slots, WolBanking77), NER (WolofEntityLinking),
classification (sentiments, emotions, sib200), QA (AfriQA), instruction (WORI),
tool_use (Code-170k ShareGPT).

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
- **Publication HF NON exécutée cette nuit** (dry-run). Créer un repo sous l'org
  `galsenai` et pousser plusieurs Go méritent ta supervision.

---

## 3 bis. Décisions prises (suite)

- **Mémoire = contrainte d'architecture** : aucun flux de `Sample` ne doit être
  matérialisé dans le chemin de build (`docs/memoire.md`).
- **Streaming HuggingFace par défaut** : plus aucun téléchargement intégral
  (le cache HF était monté à 14 Go).

---

## 4. ⚠️ PROCHAINE ÉTAPE (à décider avec toi)

1. **Rééquilibrer ou non le dataset** (cf. §2 bis) : `max_samples: 50000` sur
   les corpus sentiments/émotions ? Sans cela, 71 % du SFT est de la
   classification. **Décision requise avant publication.**
2. **Publier sur HF** : `galsenai-sft publish --execute`. Décider :
   - nom du repo (`galsenai/wolof_sft` par défaut) ;
   - **privé ou public** ;
   - track **commercial** (permissifs seulement) vs **recherche** (inclut
     INJONGO, AfriQA…). Le catalogue distingue déjà les deux.
3. **Étape 3 (traduction externe)** : choisir un moteur (LLM frontière pivot FR /
   Google Translate / NLLB) + budget, puis brancher un backend `Translator`.
4. **Décontamination** : renseigner `pretraining_corpus_paths` dans
   `configs/settings.yaml` (chemin du parquet 1M du corpus) pour activer
   l'anti-fuite au build.
5. **Livraison à l'équipe fine-tuning** (Marième, Sophie, Mohamed) : format ChatML
   (défaut) + Alpaca/ShareGPT générés en parallèle.

---

## 5. Commandes mémo

```bash
cd galsenai-sft-data
make setup                    # venv + install
make test                     # 68 tests
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
| **Mémoire (incident + protections)** | `docs/memoire.md` |
| Garde-fou mémoire | `src/galsenai_sft/core/memory.py` |
| Plan de build | `configs/build.yaml` |
| Registre licences | `metadata/datasets_registry.yaml` |
| Catalogue | `docs/dataset_catalog.md` |
| Architecture / flux | `docs/architecture.md`, `docs/data_flow.md` |
| Plan détaillé | `PLAN.md` |
