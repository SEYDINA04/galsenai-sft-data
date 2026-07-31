# 📊 Inventaire — volume disponible par tâche

> Généré automatiquement (`galsenai-sft inventory`). Ne pas éditer à la main.
> Mesuré **à la source** via l'API `datasets-server` de HuggingFace,
> sans rien télécharger. Ce sont des lignes **brutes** : un converter peut
> en produire plusieurs exemples SFT (traduction bidirectionnelle) ou moins
> (déduplication, filtre LID).

- **Généré le** : 2026-07-31T02:05:28.994891+00:00
- **Lignes intégrées au build** : 1,410,533
- **Lignes atteignables** (intégré + candidats) : 1,427,047

## Par tâche

| tâche | sources | intégré | candidat | écarté | atteignable |
|---|---:|---:|---:|---:|---:|
| classification | 4 | 642,521 | 0 | 0 | 642,521 |
| instruction | 7 | 116,304 | 0 | 94,929 | 116,304 |
| intent | 3 | 14,072 | 11,514 | 0 | 25,586 |
| ner | 4 | 5,638 | 0 | 0 | 5,638 |
| qa | 4 | 2,153 | 0 | 0 | 2,153 |
| retrieval | 1 | 0 | 0 | 0 | 0 |
| tool_use | 7 | 324,479 | 5,000 | 0 | 329,479 |
| translation | 9 | 305,366 | 0 | 0 | 305,366 |

## Par source

| dataset | tâche | statut | config/split | ciblé | tous splits | note |
|---|---|---|---|---:|---:|---|
| `Davlan/sib200` | classification | ✅ intégré | wol_Latn/train | 701 | 1,004 |  |
| `masakhane/afrixnli` | classification | ✅ intégré | wol/test | 600 | 1,050 |  |
| `michsethowusu/wolof-emotions-corpus` | classification | ✅ intégré | —/train | 320,611 | 320,611 |  |
| `michsethowusu/wolof-sentiments-corpus` | classification | ✅ intégré | —/train | 320,609 | 320,609 |  |
| `vonewman/alpaca-dataset-wolof` | instruction | ⛔ écarté | —/train | 47,463 | 47,463 | Reconditionnement d'alpaca-data-in-wolof au format Alpaca (même compte exact : 47 463). Déjà couvert par la source d'origine, intégrée. |
| `vonewman/alpaca-sharegpt-wolof` | instruction | ⛔ écarté | —/train | 47,463 | 47,463 | Même contenu que ci-dessus, au format ShareGPT. Aucun apport de volume. |
| `vonewman/wolof-instruction-dataset-alpaca` | instruction | ⛔ écarté | —/train | 3 | 3 | Dépôt quasi vide (parquet de 9 Ko) : volume négligeable. |
| `CohereLabs/aya_collection_language_split` | instruction | ✅ intégré | wolof/train | 3,146 | 3,699 |  |
| `bilalfaye/wolof-sft` | instruction | ✅ intégré | —/train | 61,971 | 61,971 |  |
| `m-a-d-i/wori-wolof-instructions` | instruction | ✅ intégré | —/train | 3,724 | 3,724 |  |
| `ngia/alpaca-data-in-wolof` | instruction | ✅ intégré | —/train | 47,463 | 47,463 |  |
| `AmazonScience/massive` | intent | 🕐 candidat | fr-FR/train | 11,514 | 16,521 | Aucune variante wolof n'existe. Utilisable seulement par localisation (protocole INJONGO) — c'est un travail de production de données, pas d'intégration. L'intent plafonne à ~15 000 exemples sans cela. |
| `karim155/WolBanking77` | intent | ✅ intégré | —/train | 11,832 | 14,791 |  |
| `masakhane/InjongoIntent` | intent | ✅ intégré | wol/train | 2,240 | 3,198 |  |
| `masakhane/masakhaner` | ner | 🕐 candidat | wol/train | — | — | MasakhaNER 1.0 (1 871 lignes wolof). Largement inclus dans MasakhaNER 2.0 désormais intégré ; l'écart résiduel ne justifie pas encore un converter. L'API taille ne répond pas (501) : le dataset repose sur un script. |
| `masakhane/masakhaner-x` | ner | 🕐 candidat | —/train | — | — | Agrégat MasakhaNER 1.0 + 2.0 en annotations de *spans* (format plus proche du SFT que le BIO). Recouvrement quasi total avec l'intégré. |
| `masakhane/masakhaner2` | ner | ✅ intégré | wol/train | 4,593 | 6,561 |  |
| `mbaye930/WolofEntityLinking` | ner | ✅ intégré | —/train | 1,045 | 1,045 |  |
| `facebook/belebele` | qa | ✅ intégré | wol_Latn/test | 900 | 900 |  |
| `masakhane/afrimgsm` | qa | ✅ intégré | wol/test | 250 | 258 |  |
| `masakhane/afrimmlu` | qa | ✅ intégré | wol/test | 500 | 608 |  |
| `masakhane/afriqa` | qa | ✅ intégré | wol/train | 503 | 1,341 |  |
| `miracl/miracl` | retrieval | 🕐 candidat | fr/train | — | — | `TaskType.RETRIEVAL` existe dans le schéma mais aucun converter ne l'alimente : la capacité IR reste à zéro après la v0.2. L'API taille répond 501 (script de chargement) — passer par la branche parquet. |
| `Salesforce/APIGen-MT-5k` | tool_use | 🕐 candidat | —/train | 5,000 | 5,000 | 5 000 trajectoires multi-tours de référence. Le format multi-tours est désormais supporté (Toucan) : intégrable sans nouveau travail de schéma. |
| `Salesforce/xlam-function-calling-60k` | tool_use | 🕐 candidat | —/train | — | — | Dépôt *gated* : l'API taille répond 404 tant que les conditions ne sont pas acceptées. Volume annoncé par la source : 60 000 appels exécutés-vérifiés. À débloquer manuellement puis intégrer. |
| `Agent-Ark/Toucan-1.5M` | tool_use | ✅ intégré | SFT/train | 119,287 | 119,287 |  |
| `NousResearch/hermes-function-calling-v1` | tool_use | ✅ intégré | func_calling/train | 1,893 | 1,893 |  |
| `Team-ACE/ToolACE` | tool_use | ✅ intégré | —/train | 11,300 | 11,300 |  |
| `michsethowusu/Code-170k-wolof` | tool_use | ✅ intégré | —/train | 176,999 | 176,999 |  |
| `nvidia/When2Call` | tool_use | ✅ intégré | train_sft/train | 15,000 | 15,000 |  |
| `Alwaly/french-wolof-translation-gs` | translation | ✅ intégré | —/train | 10,372 | 10,372 |  |
| `MaroneAI/French-Wolof_Translation-Dataset` | translation | ✅ intégré | —/train | 30,002 | 30,002 |  |
| `MaroneAI/Wolof-to-French_Translation-Dataset` | translation | ✅ intégré | —/train | 30,002 | 30,002 |  |
| `bilalfaye/english-wolof-french-dataset` | translation | ✅ intégré | —/train | 14,670 | 14,670 |  |
| `dofbi/jolof` | translation | ✅ intégré | —/train | 12,084 | 12,084 |  |
| `galsenai/centralized_wolof_french_translation_data` | translation | ✅ intégré | —/train | 98,345 | 98,345 |  |
| `galsenai/english-wolof-smol-translation` | translation | ✅ intégré | —/train | 7,405 | 7,405 |  |
| `galsenai/french-wolof-translation` | translation | ✅ intégré | —/train | 17,777 | 17,777 |  |
| `sudoping01/english-wolof-translation` | translation | ✅ intégré | —/train | 84,709 | 84,709 |  |

> **8,145 lignes disponibles mais non lues** par le build : ce sont
> les splits `validation`/`test` des sources intégrées. Elles constituent
> la réserve naturelle pour un futur jeu d'évaluation.
