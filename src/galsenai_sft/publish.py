"""Publication du dataset SFT sur HuggingFace + génération de la data card.

Implémenté mais **non exécuté automatiquement** : la publication réelle vers
``galsenai/wolof_sft`` est une action manuelle supervisée (``galsenai-sft
publish``), qui exige un ``HF_TOKEN`` en écriture. Un garde-fou refuse toute
publication sans token.
"""

from __future__ import annotations

import json
from pathlib import Path

from galsenai_sft.build import BuildManifest
from galsenai_sft.core.config import get_settings
from galsenai_sft.core.logging import get_logger

log = get_logger(__name__)

#: Description de chaque tâche, telle qu'elle apparaît dans la data card.
#: Une part par tâche sans dire ce qu'elle contient n'explique rien.
TASK_DOC: dict[str, str] = {
    "classification": (
        "Classer un texte wolof : sentiment, émotion ou thématique. "
        "Réponse attendue = un label unique (en anglais)."
    ),
    "tool_use": (
        "Appel d'outils (*function calling*) : l'assistant décide d'appeler une "
        "fonction, avec ses arguments, ou de répondre directement. Contient aussi "
        "des questions/réponses de code en wolof."
    ),
    "translation": (
        "Traduction dans les deux sens entre le wolof et le français ou l'anglais. "
        "Chaque paire source produit deux exemples (aller et retour)."
    ),
    "intent": (
        "Identifier l'intention d'un énoncé (domaine bancaire et domaines "
        "généraux), et en extraire les slots au format JSON."
    ),
    "instruction": (
        "Instructions générales en wolof avec réponse wolof native "
        "(reverse-instructions : la réponse est du texte authentique)."
    ),
    "ner": (
        "Extraire les entités nommées d'une phrase wolof, au format JSON "
        '`[{"text": ..., "type": ...}]`. Les types restent en anglais.'
    ),
    "qa": "Répondre à une question wolof en wolof (questions à pivot français).",
    "retrieval": "Recherche d'information — non couvert dans cette version.",
}


def _load_stats() -> dict:
    """Relit ``build_stats.json`` s'il existe (statistiques fines du build).

    Séparé du manifest : ces indicateurs (langue de consigne, longueurs, tours)
    sont produits par l'accumulateur de statistiques, pas par le builder. Leur
    absence ne doit pas empêcher de générer une carte.
    """
    path = get_settings().paths.interim / "build_stats.json"
    if not path.exists():
        log.warning("build_stats.json absent : data card sans statistiques fines")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:  # pragma: no cover - fichier corrompu
        log.warning("build_stats.json illisible (%s)", e)
        return {}


def _eval_manifest() -> BuildManifest | None:
    """Manifest du jeu d'évaluation, s'il a été construit."""
    path = get_settings().paths.interim / "build_manifest_test.json"
    if not path.exists():
        return None
    try:
        return BuildManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):  # pragma: no cover - manifest corrompu
        return None


def build_datacard(manifest: BuildManifest, repo: str) -> str:
    """Génère la data card (README) du dataset SFT à partir du manifest.

    La carte doit permettre de **décider d'utiliser ou non** le dataset : d'où
    les statistiques (répartition, langue de consigne, longueurs), l'explication
    de chaque tâche, et une section limites qui énonce les défauts connus plutôt
    que de les laisser découvrir à l'entraînement.

    Aucune licence n'est déclarée : le dataset agrège des sources aux licences
    différentes, annoncer une licence unique serait faux. Le détail par source
    est publié dans le catalogue du dépôt.
    """
    sources = [e for e in manifest.entries if e.n_samples]
    stats = _load_stats()
    total = manifest.total_samples or 1

    # Les langues déclarées suivent la réalité mesurée : les jeux de function
    # calling sont anglophones, l'omettre rendrait l'en-tête faux.
    languages = sorted(stats.get("by_prompt_lang", {}) or {"wo": 1}, key=lambda k: k != "wo")
    evaluation = _eval_manifest()

    lines = [
        "---",
        "language:",
        *(f"- {code}" for code in languages),
        "task_categories:",
        "- text-generation",
        "tags:",
        "- wolof",
        "- sft",
        "- instruction-tuning",
        "- chatml",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: train",
        "    path: data/train.jsonl",
        *(
            ["  - split: test", "    path: data/test.jsonl"]
            if evaluation and evaluation.total_samples
            else []
        ),
        "---",
        "",
        f"# {repo}",
        "",
        "Dataset d'instruction (SFT) **wolof** au format **ChatML**, produit par la",
        "plateforme [`galsenai-sft-data`](https://github.com/SEYDINA04/galsenai-sft-data).",
        "",
        "## Aperçu",
        "",
        f"- **Exemples** : {manifest.total_samples:,}",
        f"- **Version** : {manifest.version}",
        f"- **Généré le** : {manifest.created_at}",
        f"- **Sources** : {len(sources)}",
    ]
    if evaluation and evaluation.total_samples:
        lines.append(
            f"- **Splits** : `train` ({manifest.total_samples:,}) · "
            f"`test` ({evaluation.total_samples:,}, benchmarks tenus à l'écart)"
        )
    else:
        lines.append("- **Split** : `train` uniquement — **aucun jeu de test**")
    if manifest.partial:
        lines.append(f"- ⚠️ **Build partiel** : {manifest.stop_reason}")
    lines += _tasks_section(manifest, total)
    lines += _stats_section(stats, total)
    lines += _sources_section(sources)
    lines += _format_section()
    lines += _limits_section(manifest, stats, total)
    return "\n".join(lines)


def _tasks_section(manifest: BuildManifest, total: int) -> list[str]:
    """Répartition par tâche, chaque tâche accompagnée de ce qu'elle contient."""
    lines = [
        "",
        "## Tâches",
        "",
        "| tâche | exemples | part | contenu |",
        "|---|---:|---:|---|",
    ]
    for task, n in sorted(manifest.by_task.items(), key=lambda kv: -kv[1]):
        doc = TASK_DOC.get(task, "")
        lines.append(f"| `{task}` | {n:,} | {n / total:.1%} | {doc} |")
    return lines


def _stats_section(stats: dict, total: int) -> list[str]:
    """Statistiques transverses : langue de consigne, longueurs, structure."""
    if not stats:
        return []
    lines = ["", "## Statistiques", ""]

    if by_lang := stats.get("by_prompt_lang"):
        lines += [
            "**Langue de la consigne** — seule l'instruction est localisée ; les",
            "schémas JSON, types d'entités et labels restent en anglais.",
            "",
            "| langue | exemples | part |",
            "|---|---:|---:|",
        ]
        for lang, n in sorted(by_lang.items(), key=lambda kv: -kv[1]):
            label = {"wo": "wolof", "fr": "français"}.get(lang, lang)
            lines.append(f"| {label} (`{lang}`) | {n:,} | {n / total:.1%} |")
        lines.append("")

    user_chars = stats.get("total_user_chars", 0)
    asst_chars = stats.get("total_assistant_chars", 0)
    lines += [
        "**Volume et structure**",
        "",
        "| indicateur | valeur |",
        "|---|---:|",
        f"| caractères (consignes) | {user_chars:,} |",
        f"| caractères (réponses) | {asst_chars:,} |",
        f"| longueur moyenne d'une consigne | {user_chars // total:,} car. |",
        f"| longueur moyenne d'une réponse | {asst_chars // total:,} car. |",
        f"| conversations multi-tours | {stats.get('multi_turn', 0):,} |",
        f"| exemples avec appel d'outil | {stats.get('with_tool_calls', 0):,} |",
        f"| appels d'outil au total | {stats.get('total_tool_calls', 0):,} |",
    ]
    return lines


def _sources_section(sources: list) -> list[str]:
    """Sources avec le taux de conservation : lignes lues -> exemples produits."""
    lines = [
        "",
        "## Sources",
        "",
        "`lignes lues` = lignes du dataset d'origine ; `exemples` = exemples SFT",
        "produits. Le rapport peut dépasser 100 % (traduction bidirectionnelle) ou",
        "descendre en dessous (déduplication, filtre de langue).",
        "",
        "| dataset | tâche | lignes lues | exemples | rapport |",
        "|---|---|---:|---:|---:|",
    ]
    for e in sources:
        ratio = f"{e.n_samples / e.n_raw:.0%}" if e.n_raw else "—"
        lines.append(f"| `{e.dataset_id}` | {e.task} | {e.n_raw:,} | {e.n_samples:,} | {ratio} |")
    lines += [
        "",
        "Licences détaillées par source : "
        "[`docs/dataset_catalog.md`](https://github.com/SEYDINA04/galsenai-sft-data/blob/main/docs/dataset_catalog.md). "
        "Les sources ont des licences hétérogènes ; le dataset n'en déclare donc aucune.",
        "Sources écartées et motifs : "
        "[`docs/targeting.md`](https://github.com/SEYDINA04/galsenai-sft-data/blob/main/docs/targeting.md).",
    ]
    return lines


def _format_section() -> list[str]:
    return [
        "",
        "## Format",
        "",
        "Un objet JSON par ligne :",
        "",
        "```json",
        '{"messages": [{"role": "user", "content": "Tekkil lii ci wolof: Bonjour"},',
        '              {"role": "assistant", "content": "Salaamaalekum"}],',
        ' "task": "translation", "source": "galsenai/french-wolof-translation",',
        ' "prompt_lang": "wo", "meta": {"direction": "fr->wo"}}',
        "```",
        "",
        "| champ | rôle |",
        "|---|---|",
        "| `messages` | conversation ChatML (`user` puis `assistant`) |",
        "| `task` | tâche SFT (voir le tableau ci-dessus) |",
        "| `source` | dataset HuggingFace d'origine |",
        "| `prompt_lang` | langue de la consigne (`wo` ou `fr`) |",
        "| `meta` | informations propres à la tâche (direction de traduction, sous-tâche…) |",
        "",
        "```python",
        "from datasets import load_dataset",
        f'train = load_dataset("{get_settings().hf.repo}", split="train")',
        f'test = load_dataset("{get_settings().hf.repo}", split="test")   # benchmarks',
        "```",
    ]


def _limits_section(manifest: BuildManifest, stats: dict, total: int) -> list[str]:
    """Défauts connus. Les taire reviendrait à les faire découvrir à l'entraînement."""
    lines = ["", "## Limites connues", ""]

    if manifest.by_task:
        top_task, top_n = max(manifest.by_task.items(), key=lambda kv: kv[1])
        lines += [
            f"**Déséquilibre des tâches.** `{top_task}` représente {top_n / total:.1%} du",
            "dataset. Un entraînement sans pondération apprendra surtout cette tâche.",
            "Pondérer les tâches, ou sous-échantillonner, est laissé au consommateur.",
            "",
        ]

    if stats.get("with_tool_calls", 0) == 0 and manifest.by_task.get("tool_use"):
        n = manifest.by_task["tool_use"]
        lines += [
            f"**La tâche `tool_use` ne contient aucun appel d'outil.** Ses {n:,} exemples",
            "sont des questions/réponses de code au format conversationnel. Le nom de la",
            "tâche décrit une intention de couverture, pas son contenu actuel.",
            "",
        ]

    if stats.get("multi_turn", 0) == 0:
        lines += [
            "**Aucune conversation multi-tours.** Tous les exemples font exactement un",
            "tour (une consigne, une réponse). Le dataset n'entraîne pas le suivi de",
            "contexte sur plusieurs échanges.",
            "",
        ]

    evaluation = _eval_manifest()
    if evaluation and evaluation.total_samples:
        benchmarks = ", ".join(f"`{e.dataset_id}`" for e in evaluation.entries if e.n_samples)
        lines += [
            f"**Le split `test` reste optimiste.** Il réunit {benchmarks} "
            f"({evaluation.total_samples:,} exemples), tenus hors du `train` et vérifiés",
            "sans recouvrement. Mais les phrases sources de plusieurs de ces benchmarks",
            "figurent déjà, en texte brut, dans le corpus de pré-entraînement du projet :",
            "un modèle les a donc pu voir **avant** le SFT. Un score honnête demande un",
            "jeu de test **wolof natif**, écrit pour l'occasion et jamais publié.",
            "",
            "**Pas de split `validation`.** À découper dans `train` (~2 %, stratifié par",
            "tâche et par source) pour régler les hyperparamètres.",
            "",
        ]
    else:
        lines += [
            "**Pas de jeu d'évaluation.** L'intégralité des exemples est dans `train`.",
            "Aucune décontamination n'a été appliquée : plusieurs sources sont déjà",
            "présentes dans le corpus de pré-entraînement du projet. Toute métrique",
            "calculée sur ce dataset serait optimiste — construire un jeu de test",
            "**wolof natif et indépendant** avant d'évaluer.",
            "",
        ]

    lines += [
        "**Vérification de langue partielle.** Le filtre GlotLID (`wol_Latn`) n'est",
        "appliqué qu'aux sources issues de traduction automatique, les plus bruitées.",
        "Les autres sont reprises telles quelles : leur qualité est celle de leur",
        "producteur. Les jeux de *function calling* sont anglophones par nature.",
        "",
        "**Déduplication exacte uniquement.** Les doublons sont retirés globalement",
        "(index partagé par toutes les sources), mais sur une empreinte **exacte** :",
        "deux formulations proches d'une même phrase subsistent. Aucune",
        "déduplication approchée (MinHash) n'est appliquée.",
        "",
    ]
    return lines


def publish(
    manifest: BuildManifest,
    chatml_file: str | Path,
    repo: str | None = None,
    private: bool = True,
    dry_run: bool = True,
    card_only: bool = False,
    eval_file: str | Path | None = None,
) -> str:
    """Publie le dataset SFT sur HF. ``dry_run`` par défaut (n'envoie rien).

    ``card_only`` n'envoie que la data card : utile quand seul le README change
    et que le gigaoctet de ``train.jsonl`` est déjà en ligne.

    ``eval_file`` publie le split ``test`` (benchmarks) à côté du ``train``.

    Retourne l'URL du dataset (ou un message dry-run).
    """
    settings = get_settings()
    repo = repo or settings.hf.repo
    token = settings.hf_token

    card = build_datacard(manifest, repo)
    card_path = settings.paths.interim / "README_sft.md"
    card_path.write_text(card, encoding="utf-8")

    if dry_run:
        log.info("[dry-run] data card générée -> %s (aucun envoi HF)", card_path)
        return f"[dry-run] prêt à publier {manifest.total_samples} exemples vers {repo}"

    if not token:
        # Repli sur le token de la session `huggingface-cli login` : le vrai
        # garde-fou est le drapeau --execute, pas la variable d'environnement.
        from huggingface_hub import get_token

        token = get_token()
    if not token:
        raise RuntimeError(
            "aucun token HF (ni HF_TOKEN, ni session `huggingface-cli login`) : "
            "publication refusée (garde-fou)."
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"SFT wolof v{manifest.version} — {manifest.total_samples} exemples",
    )
    if not card_only:
        api.upload_file(
            path_or_fileobj=str(chatml_file),
            path_in_repo="data/train.jsonl",
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"SFT wolof v{manifest.version}",
        )
        if eval_file is not None and Path(eval_file).exists():
            api.upload_file(
                path_or_fileobj=str(eval_file),
                path_in_repo="data/test.jsonl",
                repo_id=repo,
                repo_type="dataset",
                commit_message=f"jeu d'évaluation v{manifest.version}",
            )
    else:
        log.info("card-only : les fichiers de données restent inchangés")
    url = f"https://huggingface.co/datasets/{repo}"
    log.info("publié -> %s", url)
    return url
