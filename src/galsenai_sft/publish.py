"""Publication du dataset SFT sur HuggingFace + génération de la data card.

Implémenté mais **non exécuté automatiquement** : la publication réelle vers
``galsenai/wolof_sft`` est une action manuelle supervisée (``galsenai-sft
publish``), qui exige un ``HF_TOKEN`` en écriture. Un garde-fou refuse toute
publication sans token.
"""

from __future__ import annotations

from pathlib import Path

from galsenai_sft.build import BuildManifest
from galsenai_sft.core.config import get_settings
from galsenai_sft.core.logging import get_logger

log = get_logger(__name__)


def build_datacard(manifest: BuildManifest, repo: str) -> str:
    """Génère la data card (README) du dataset SFT à partir du manifest.

    Aucune licence n'est déclarée : le dataset **agrège des sources aux
    licences différentes**, annoncer une licence unique serait faux. La carte
    se contente de lister les sources et leur volume ; la vérification des
    conditions d'usage de chaque source relève de l'utilisateur (le catalogue
    du dépôt documente les licences connues).
    """
    sources = [e for e in manifest.entries if e.n_samples]

    lines = [
        "---",
        "language:",
        "- wo",
        "task_categories:",
        "- text-generation",
        "tags:",
        "- wolof",
        "- sft",
        "- instruction-tuning",
        "- chatml",
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
    ]
    if manifest.partial:
        lines.append(f"- ⚠️ **Build partiel** : {manifest.stop_reason}")
    lines += [
        "",
        "## Répartition par tâche",
        "",
        "| tâche | exemples | part |",
        "|---|---:|---:|",
    ]
    total = manifest.total_samples or 1
    for task, n in manifest.by_task.items():
        lines.append(f"| {task} | {n:,} | {n / total:.1%} |")
    lines += [
        "",
        "## Sources",
        "",
        "| dataset | tâche | exemples |",
        "|---|---|---:|",
    ]
    for e in sources:
        lines.append(f"| `{e.dataset_id}` | {e.task} | {e.n_samples:,} |")
    lines += [
        "",
        "## Format",
        "",
        "Chaque exemple est une conversation ChatML : "
        '`{"messages": [{"role": "user", "content": ...}, '
        '{"role": "assistant", "content": ...}]}`.',
        "",
    ]
    return "\n".join(lines)


def publish(
    manifest: BuildManifest,
    chatml_file: str | Path,
    repo: str | None = None,
    private: bool = True,
    dry_run: bool = True,
    card_only: bool = False,
) -> str:
    """Publie le dataset SFT sur HF. ``dry_run`` par défaut (n'envoie rien).

    ``card_only`` n'envoie que la data card : utile quand seul le README change
    et que les 550 Mo de ``train.jsonl`` sont déjà en ligne.

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
    else:
        log.info("card-only : data/train.jsonl inchangé (non renvoyé)")
    url = f"https://huggingface.co/datasets/{repo}"
    log.info("publié -> %s", url)
    return url
