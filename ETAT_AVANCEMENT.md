# 📊 État d'avancement — Plateforme SFT wolof (`galsenai-sft-data`)

> **But de ce fichier** : reprendre le travail rapidement. Lire ce fichier, puis
> aller à la section **« PROCHAINE ÉTAPE »**.

_Dernière mise à jour : 2026-07-30 (nuit) · Auteur : Babacar Ndao_
_Dépôt : https://github.com/SEYDINA04/galsenai-sft-data (public, branche `main`)_

---

## 1. Résumé en 30 secondes

La **plateforme complète de préparation de datasets SFT wolof** est construite,
testée et poussée sur GitHub. Elle fait les 3 objectifs du brief :

1. **Collecter** des datasets wolof existants (NER, intent, traduction, QA,
   classification, tool use).
2. **Convertir** chacun en un **format canonique unique (ChatML)**.
3. **Traduire** des datasets externes vers le wolof (interface prête, moteur
   réel volontairement différé).

**Les 7 lots (0 → 7) sont terminés.** 49 tests passent. Le build end-to-end a été
vérifié sur données réelles (30 lignes → 60 exemples ChatML).

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

**Datasets intégrés (11)** : voir `docs/dataset_catalog.md`. Couverture :
traduction (2), intent (2 : INJONGO+slots, WolBanking77), NER (WolofEntityLinking),
classification (sentiments, emotions, sib200), QA (AfriQA), instruction (WORI),
tool_use (Code-170k ShareGPT).

**Commits poussés** : lots 0→7 (voir `git log`). Correctif annexe dans
`wolof_scraper` : GlotLID `model.bin` → `model_v3.bin` (déjà poussé).

---

## 3. Décisions prises (et pourquoi)

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

## 4. ⚠️ PROCHAINE ÉTAPE (à décider avec toi)

1. **Lancer le build réel complet** : `galsenai-sft build` télécharge les 11
   datasets et produit le dataset SFT ChatML. ⚠️ Volumineux (Code-170k = 344k
   lignes). À lancer quand tu valides. Option de test : `--limit 100`.
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
make test                     # 49 tests
uv run galsenai-sft converters        # datasets disponibles
uv run galsenai-sft build --limit 100 # build de test (rapide)
uv run galsenai-sft build             # build complet
uv run galsenai-sft publish           # dry-run (data card)
uv run galsenai-sft publish --execute # publier sur HF (supervisé)
uv run galsenai-sft catalog           # régénère le catalogue
```

## 6. Fichiers utiles

| Élément | Chemin |
|---|---|
| Schéma canonique | `src/galsenai_sft/core/schema.py` |
| Ajouter un dataset | `docs/contribution_guide.md` |
| Plan de build | `configs/build.yaml` |
| Registre licences | `metadata/datasets_registry.yaml` |
| Catalogue | `docs/dataset_catalog.md` |
| Architecture / flux | `docs/architecture.md`, `docs/data_flow.md` |
| Plan détaillé | `PLAN.md` |
