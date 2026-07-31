# 🗺️ Plan — Plateforme SFT wolof (`galsenai-sft-data`)

Plan de référence du projet : objectifs, architecture, avancement, suite.

---

## 1. Objectif

Construire la **plateforme centrale de données SFT** du projet GalsenAI LLM :
collecter les datasets wolof existants, les convertir dans un **format canonique
unique (ChatML)**, et intégrer la traduction de datasets externes vers le wolof.
Livrable : des datasets d'instruction prêts pour l'équipe fine-tuning.

Capacités ciblées (façon LFM 2.5) : **Data Extraction** (NER, IR, Intent, QA) +
**Tool Use** (agentique).

---

## 2. Architecture (résumé)

```
Loader → Converter (@register) → Validators → Exporters → build → publish → HF
             ▲ plugin              (schéma/LID/    (chatml/    (manifest)
                                   décontam)      alpaca/sharegpt)
                                        └→ Translators (QE + review) [étape 3, différée]
```

Détails : `docs/architecture.md` et `docs/data_flow.md`.

---

## 3. Lots (0 → 7) — TOUS TERMINÉS ✅

| Lot | Objet | État |
|---|---|---|
| 0 | Squelette + `core` (schéma ChatML, config, logging, io) | ✅ |
| 1 | Registry plugin + BaseConverter + exporters + CLI | ✅ |
| 2 | Validators (schéma, qualité, LID v3, décontamination, stats) | ✅ |
| 3 | 11 converters wolof (7 tâches) | ✅ |
| 4 | Metadata + catalogue auto | ✅ |
| 5 | Translators (interface + QE + review, moteur différé) | ✅ |
| 6 | Builder end-to-end + publish HF (dry-run) | ✅ |
| 7 | Documentation | ✅ |
| 8 | Mémoire : build en flux, `MemoryGuard`, plafond cgroup, streaming HF | ✅ |

Qualité : **101 tests**, ruff lint+format, CI GitHub Actions, hooks pre-commit.

---

## 4. Suite (post-lot 7) — à valider avec le chef

### Phase A — Livraison v1
- [x] **Lot 8 mémoire** : build en flux, garde-fou, plafond cgroup.
- [x] `make build` complet (11 datasets) → **905 362 exemples**, pic 2 Go, 0 échec.
- [x] Revue des stats → déséquilibre identifié (71 % classification), assumé.
- [x] `galsenai-sft publish --execute` → `galsenai/wolof_sft` (**privé**, v0.1.0).
- [ ] Durcissement machine : `sudo apt install earlyoom` (à faire par le chef).
- [ ] Remettre ChatML + Alpaca + ShareGPT à l'équipe fine-tuning (+ consigne de
      pondération des tâches à l'entraînement).
- [ ] Décider de la publication publique + du track (commercial vs recherche).

### Phase B — Étape 3 : traduction de datasets externes
- [ ] Choisir le moteur (LLM frontière pivot FR recommandé) + budget/clé.
- [ ] Implémenter un backend `Translator` (ex. `LLMTranslator`).
- [ ] Registre d'entités Sénégal (localisation, pas traduction).
- [ ] Sélectionner les datasets externes (INJONGO-en, MASSIVE, xlam-FC…).
- [ ] Boucle QE (LID + SSA-COMET) → revue communautaire → intégration.

### Phase C — Extensions
- [ ] Nouveaux converters (MasakhaNER, autres traductions, tool use wolof).
- [ ] Versioning données : DVC ou révisions HF.
- [ ] Splits train/val/test + décontamination croisée.
- [ ] Transférer le repo vers l'org `galsenai` (`gh repo transfer`).

---

## 5. Risques & conformité

- **Licences** : le catalogue sépare permissif (commercial) / non-commercial
  (recherche) / non vérifié. Ne pas mélanger les tracks à la publication.
- **Fuite d'évaluation** : décontaminer vs corpus de pré-entraînement (plusieurs
  datasets sources y sont déjà : InjongoIntent, WolofEntityLinking, sib200…).
- **Qualité traduction** : ne jamais publier de traduction non relue comme gold.
- **Reproductibilité** : GlotLID v3 épinglé + manifest à checksums.

---

## 6. Références

- Survey datasets SFT : `wolof_scraper/docs/SFT_DATASETS_SURVEY.md`
- Ciblage datasets : `wolof_scraper/docs/SFT_TARGETING.md`
- État d'avancement : `ETAT_AVANCEMENT.md`
