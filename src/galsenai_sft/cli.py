"""Interface en ligne de commande : ``galsenai-sft``.

Sous-commandes (lot 1) :
  - ``converters``  : liste les datasets disposant d'un converter
  - ``convert``     : ingère un dataset HF -> Samples ChatML (JSONL)
  - ``export``      : convertit un JSONL de Samples vers alpaca/sharegpt/chatml
  - ``version``
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from galsenai_sft import __version__
from galsenai_sft.core.io import read_samples_jsonl, write_jsonl, write_samples_jsonl
from galsenai_sft.core.logging import get_logger
from galsenai_sft.exporters import get_exporter
from galsenai_sft.registry import available, get_converter

app = typer.Typer(add_completion=False, help="Plateforme de datasets SFT wolof (GalsenAI).")
console = Console()
log = get_logger("cli")


@app.command()
def version() -> None:
    """Affiche la version."""
    console.print(f"galsenai-sft {__version__}")


@app.command()
def converters() -> None:
    """Liste les datasets disposant d'un converter enregistré."""
    table = Table("dataset_id", "tâche", title="Converters disponibles")
    for ds in available():
        cls = get_converter(ds)
        table.add_row(ds, cls.task.value)
    console.print(table)


@app.command()
def catalog(
    out: Path = typer.Option(
        None, "--out", "-o", help="chemin de sortie (défaut: docs/dataset_catalog.md)"
    ),
) -> None:
    """Génère le catalogue Markdown des datasets (métadonnées + licences)."""
    from galsenai_sft.metadata import write_catalog

    path = write_catalog(out_path=out)
    console.print(f"[green]✓[/green] catalogue généré -> {path}")


@app.command()
def convert(
    dataset_id: str = typer.Argument(..., help="repo_id HF (doit avoir un converter)"),
    split: str = typer.Option("train", help="split HF à charger"),
    config: str | None = typer.Option(None, help="config/sous-ensemble HF"),
    limit: int | None = typer.Option(None, help="limiter le nombre de lignes (debug)"),
    out: Path = typer.Option(..., "--out", "-o", help="fichier JSONL de sortie (Samples)"),
    seed: int = typer.Option(42, help="seed déterministe"),
) -> None:
    """Ingère un dataset HF et écrit les Samples canoniques en JSONL."""
    from itertools import islice

    from galsenai_sft.loaders import HFLoader

    cls = get_converter(dataset_id)
    conv = cls(seed=seed)
    log.info("Chargement %s (split=%s, config=%s)…", dataset_id, split, config)
    rows = HFLoader().load(dataset_id, split=split, config=config)  # streaming
    if limit:
        rows = islice(rows, limit)

    n = write_samples_jsonl(conv.convert(iter(rows)), out)
    console.print(f"[green]✓[/green] {n:,} Samples écrits -> {out}")


@app.command()
def validate(
    inp: Path = typer.Argument(..., help="JSONL de Samples à valider"),
) -> None:
    """Valide un JSONL de Samples (qualité + doublons) et affiche un rapport."""
    from galsenai_sft.validators import validate_quality

    report = validate_quality(read_samples_jsonl(inp))
    console.print(report.summary())
    if not report.ok:
        console.print(f"[red]✗ {report.n_errors} erreurs bloquantes[/red]")
        raise typer.Exit(code=1)
    console.print("[green]✓ aucune erreur bloquante[/green]")


@app.command()
def stats(
    inp: Path = typer.Argument(..., help="JSONL de Samples"),
) -> None:
    """Affiche les statistiques d'un JSONL de Samples."""
    from galsenai_sft.validators import compute_statistics

    st = compute_statistics(read_samples_jsonl(inp))
    table = Table("indicateur", "valeur", title="Statistiques")
    table.add_row("total", f"{st.total:,}")
    table.add_row("multi-tours", f"{st.multi_turn:,}")
    table.add_row("avec tool_calls", f"{st.with_tool_calls:,}")
    table.add_row("car. assistant moy.", f"{st.avg_assistant_chars}")
    for task, n in st.by_task.items():
        table.add_row(f"tâche · {task}", f"{n:,}")
    for lang, n in st.by_prompt_lang.items():
        table.add_row(f"consigne · {lang}", f"{n:,}")
    console.print(table)


@app.command()
def build(
    limit: int | None = typer.Option(None, help="limiter les lignes par dataset (debug/smoke)"),
    version: str = typer.Option("0.1.0", help="version du build"),
    decontaminate: bool = typer.Option(False, help="décontaminer vs corpus de pré-entraînement"),
    streaming: bool = typer.Option(
        True, help="lire les datasets en flux (défaut) au lieu de tout télécharger"
    ),
    min_available_mb: float | None = typer.Option(
        None,
        help="plancher de RAM système libre (Mo) sous lequel le build s'arrête proprement",
    ),
    max_rss_mb: float | None = typer.Option(
        None, help="plafond de mémoire du processus de build (Mo)"
    ),
) -> None:
    """Construit le dataset SFT complet (charge, convertit, valide, exporte).

    Mémoire bornée : écriture au fil de l'eau + garde-fou. Pour un plafond
    **dur** (cgroup), préférer ``make build`` / ``scripts/build_guarded.sh``.
    """
    from galsenai_sft.build import build as run_build
    from galsenai_sft.core.config import get_settings
    from galsenai_sft.core.memory import MemoryGuard
    from galsenai_sft.loaders import HFLoader

    settings = get_settings()
    if min_available_mb is not None:
        settings.memory.min_available_mb = min_available_mb
    if max_rss_mb is not None:
        settings.memory.max_rss_mb = max_rss_mb

    manifest = run_build(
        version=version,
        limit=limit,
        decontaminate_corpus=decontaminate,
        settings=settings,
        loader=HFLoader(streaming=streaming, batch_size=settings.build.batch_size),
        guard=MemoryGuard.from_settings(settings.memory),
    )
    table = Table(
        "tâche", "exemples", title=f"Build v{version} — {manifest.total_samples:,} exemples"
    )
    for task, n in manifest.by_task.items():
        table.add_row(task, f"{n:,}")
    console.print(table)
    console.print(f"pic mémoire du build : [cyan]{manifest.peak_rss_mb:,.0f} Mo[/cyan]")
    if manifest.partial:
        console.print(f"[yellow]⚠ build PARTIEL — {manifest.stop_reason}[/yellow]")
    errors = [e for e in manifest.entries if e.error]
    if errors:
        console.print(f"[yellow]⚠ {len(errors)} datasets en échec :[/yellow]")
        for e in errors:
            console.print(f"  - {e.dataset_id} : {e.error}")
    if manifest.partial:
        raise typer.Exit(code=2)


@app.command()
def doctor() -> None:
    """Vérifie les moyens mémoire avant un gros build (RAM, swap, cgroup)."""
    from galsenai_sft.core.config import get_settings
    from galsenai_sft.core.memory import available_mb, cgroup_limit_mb, process_rss_mb

    settings = get_settings()
    limit = cgroup_limit_mb()
    table = Table("contrôle", "valeur", title="Diagnostic mémoire")
    table.add_row("RAM disponible", f"{available_mb() / 1024:,.1f} Go")
    table.add_row("RSS du processus", f"{process_rss_mb():,.0f} Mo")
    table.add_row(
        "plafond cgroup", f"{limit / 1024:,.1f} Go" if limit else "[yellow]aucun[/yellow]"
    )
    table.add_row("plancher du garde-fou", f"{settings.memory.min_available_mb:,.0f} Mo")
    table.add_row("streaming HF", "oui" if settings.build.streaming else "[yellow]non[/yellow]")
    console.print(table)
    if not limit:
        console.print(
            "[yellow]⚠ pas de plafond cgroup : lance les gros builds via "
            "[bold]make build[/bold] (isolation mémoire).[/yellow]"
        )


@app.command()
def publish(
    version: str = typer.Option("0.1.0", help="version publiée"),
    repo: str | None = typer.Option(None, help="repo HF cible (défaut: config)"),
    execute: bool = typer.Option(False, "--execute", help="publier réellement (sinon dry-run)"),
    card_only: bool = typer.Option(
        False, "--card-only", help="n'envoyer que la data card (README), pas le JSONL"
    ),
) -> None:
    """Publie le dernier build sur HuggingFace (dry-run par défaut)."""
    import json

    from galsenai_sft.build import BuildManifest
    from galsenai_sft.core.config import get_settings
    from galsenai_sft.publish import publish as run_publish

    settings = get_settings()
    manifest_path = settings.paths.interim / "build_manifest.json"
    if not manifest_path.exists():
        console.print("[red]✗ aucun build — lance d'abord 'galsenai-sft build'[/red]")
        raise typer.Exit(code=1)
    manifest = BuildManifest.model_validate(json.loads(manifest_path.read_text()))
    chatml_all = settings.paths.processed_chatml / "all.jsonl"

    result = run_publish(
        manifest, chatml_all, repo=repo, dry_run=not execute, card_only=card_only
    )
    console.print(result)


@app.command()
def export(
    inp: Path = typer.Argument(..., help="JSONL de Samples (sortie de 'convert')"),
    fmt: str = typer.Option("chatml", "--to", help="chatml | alpaca | sharegpt"),
    out: Path = typer.Option(..., "--out", "-o", help="fichier JSONL de sortie"),
    skip_errors: bool = typer.Option(
        True, help="ignorer les Samples non exportables (ex. Alpaca multi-tours)"
    ),
) -> None:
    """Convertit un JSONL de Samples vers un format de sortie."""
    exporter = get_exporter(fmt)

    def _rows():
        for s in read_samples_jsonl(inp):
            try:
                yield exporter(s)
            except ValueError:
                if not skip_errors:
                    raise

    n = write_jsonl(_rows(), out)
    console.print(f"[green]✓[/green] {n:,} lignes exportées ({fmt}) -> {out}")


if __name__ == "__main__":
    app()
