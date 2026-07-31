# 📊 Inventaire — volume disponible par tâche

> Généré automatiquement (`galsenai-sft inventory`). Ne pas éditer à la main.
> Mesuré **à la source** via l'API `datasets-server` de HuggingFace,
> sans rien télécharger. Ce sont des lignes **brutes** : un converter peut
> en produire plusieurs exemples SFT (traduction bidirectionnelle) ou moins
> (déduplication, filtre LID).

- **Généré le** : 2026-07-31T00:18:52.838294+00:00
- **Lignes intégrées au build** : 870,711
- **Lignes atteignables** (intégré + candidats) : 1,018,191

## Par tâche

| tâche | sources | intégré | candidat | écarté | atteignable |
|---|---:|---:|---:|---:|---:|
| classification | 3 | 641,921 | 0 | 0 | 641,921 |
| instruction | 2 | 3,724 | 0 | 47,463 | 3,724 |
| intent | 2 | 14,072 | 0 | 0 | 14,072 |
| ner | 2 | 1,045 | 0 | 4,593 | 1,045 |
| qa | 1 | 503 | 0 | 0 | 503 |
| retrieval | 1 | 0 | 0 | 0 | 0 |
| tool_use | 7 | 176,999 | 147,480 | 5,000 | 324,479 |
| translation | 2 | 32,447 | 0 | 0 | 32,447 |

## Par source

| dataset | tâche | statut | config/split | ciblé | tous splits | note |
|---|---|---|---|---:|---:|---|
| `Davlan/sib200` | classification | ✅ intégré | wol_Latn/train | 701 | 1,004 |  |
| `michsethowusu/wolof-emotions-corpus` | classification | ✅ intégré | —/train | 320,611 | 320,611 |  |
| `michsethowusu/wolof-sentiments-corpus` | classification | ✅ intégré | —/train | 320,609 | 320,609 |  |
| `ngia/alpaca-data-in-wolof` | instruction | ⛔ écarté | —/train | 47,463 | 47,463 | Traduction automatique non auditée. Le ciblage impose de ne jamais utiliser de MT non vérifiée comme donnée gold ; à réintégrer seulement après audit LID + qualité (cf. le filtre appliqué à WORI). |
| `m-a-d-i/wori-wolof-instructions` | instruction | ✅ intégré | —/train | 3,724 | 3,724 |  |
| `karim155/WolBanking77` | intent | ✅ intégré | —/train | 11,832 | 14,791 |  |
| `masakhane/InjongoIntent` | intent | ✅ intégré | wol/train | 2,240 | 3,198 |  |
| `masakhane/masakhaner2` | ner | ⛔ écarté | wol/train | 4,593 | 6,561 | Seul NER wolof gold annoté humainement, mais CC BY-NC : incompatible avec un jeu utilisable en commercial. Réservé à un futur track recherche. |
| `mbaye930/WolofEntityLinking` | ner | ✅ intégré | —/train | 1,045 | 1,045 |  |
| `masakhane/afriqa` | qa | ✅ intégré | wol/train | 503 | 1,341 |  |
| `miracl/miracl` | retrieval | 🕐 candidat | fr/train | — | — | P1. Apache-2.0. `TaskType.RETRIEVAL` existe dans le schéma mais aucun converter ne l'alimente : la capacité IR est à zéro aujourd'hui. |
| `Agent-Ark/Toucan-1.5M` | tool_use | 🕐 candidat | SFT/train | 119,287 | 119,287 | P0. Apache-2.0, trajectoires multi-tours réellement exécutées (MCP). Choix par défaut pour introduire du vrai function calling. |
| `NousResearch/hermes-function-calling-v1` | tool_use | 🕐 candidat | func_calling/train | 1,893 | 1,893 | P0. Apache-2.0. Verrouille le format <tool_call> de l'écosystème. |
| `Salesforce/xlam-function-calling-60k` | tool_use | 🕐 candidat | —/train | — | — | P0. CC BY 4.0 mais dépôt *gated* : l'API taille répond 404 tant que les conditions ne sont pas acceptées. Volume annoncé par la source : 60 000. |
| `Team-ACE/ToolACE` | tool_use | 🕐 candidat | —/train | 11,300 | 11,300 | P1. Apache-2.0, 26k+ API : diversité de schémas. |
| `nvidia/When2Call` | tool_use | 🕐 candidat | train_sft/train | 15,000 | 15,000 | P1. Décision d'appel / abstention — indispensable à un petit modèle, qui sur-appelle les outils sans contre-exemples. |
| `Salesforce/APIGen-MT-5k` | tool_use | ⛔ écarté | —/train | 5,000 | 5,000 | CC BY-NC : exclu du jeu commercial, comme MasakhaNER 2.0. |
| `michsethowusu/Code-170k-wolof` | tool_use | ✅ intégré | —/train | 176,999 | 176,999 |  |
| `bilalfaye/english-wolof-french-dataset` | translation | ✅ intégré | —/train | 14,670 | 14,670 |  |
| `galsenai/french-wolof-translation` | translation | ✅ intégré | —/train | 17,777 | 17,777 |  |

> **5,058 lignes disponibles mais non lues** par le build : ce sont
> les splits `validation`/`test` des sources intégrées. Elles constituent
> la réserve naturelle pour un futur jeu d'évaluation.
