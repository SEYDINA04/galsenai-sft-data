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
    from datasets import load_dataset

    cls = get_converter(dataset_id)
    conv = cls(seed=seed)
    log.info("Chargement %s (split=%s, config=%s)…", dataset_id, split, config)
    ds = (
        load_dataset(dataset_id, config, split=split)
        if config
        else load_dataset(dataset_id, split=split)
    )
    if limit:
        ds = ds.select(range(min(limit, ds.num_rows)))

    n = write_samples_jsonl(conv.convert(iter(ds)), out)
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
        console.print(f"[red]✗ {len(report.errors)} erreurs bloquantes[/red]")
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
