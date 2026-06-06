"""FoodScholar CLI.

Every command is a thin wrapper around `FoodScholar` so the CLI and Python API
exercise the exact same code path. Phase commands whose facade methods raise
`NotImplementedError` print a friendly deferred message and exit non-zero.
"""

from __future__ import annotations

from pathlib import Path

import typer

from foodscholar import FoodScholar, __version__

app = typer.Typer(
    name="foodscholar",
    help="FoodScholar — hierarchical knowledge graph over nutrition literature.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = typer.Option(
    ...,
    "--config",
    "-c",
    exists=True,
    dir_okay=False,
    readable=True,
    help="Path to the YAML config file.",
)


def _build(config: Path) -> FoodScholar:
    try:
        return FoodScholar.from_config(config)
    except NotImplementedError as e:
        typer.echo(f"[foodscholar] {e}", err=True)
        raise typer.Exit(code=1) from None


def _run_phase(fs: FoodScholar, phase_name: str, method: str) -> None:
    try:
        getattr(fs, method)()
    except NotImplementedError as e:
        typer.echo(f"[foodscholar] {e}", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def init(config: Path = ConfigOption) -> None:
    """Provision backing stores (ES index + Neo4j constraints) declared in the config."""
    fs = _build(config)
    fs.init()


@app.command()
def info(config: Path = ConfigOption) -> None:
    """Show package version, config hash, and resolved backend identities."""
    fs = _build(config)
    for k, v in fs.info().items():
        typer.echo(f"{k:14s} {v}")


@app.command()
def annotate(config: Path = ConfigOption) -> None:
    """Run NER + entity linking + embeddings over the loaded chunks."""
    _run_phase(_build(config), "annotate", "annotate")


@app.command("build-layer-a")
def build_layer_a(config: Path = ConfigOption) -> None:
    """Build Layer A — the curated, multi-facet backbone from FoodOn."""
    _run_phase(_build(config), "build-layer-a", "build_layer_a")


@app.command()
def attach(config: Path = ConfigOption) -> None:
    """Write chunk→shelf attachments and denormalize shelf_ids onto chunks."""
    _run_phase(_build(config), "attach", "attach")


@app.command("build-layer-b")
def build_layer_b(config: Path = ConfigOption) -> None:
    """Build Layer B — theme communities per shelf."""
    _run_phase(_build(config), "build-layer-b", "build_layer_b")


@app.command("report-layer-b")
def report_layer_b(
    config: Path = ConfigOption,
    facet: str = typer.Option("foods", "--facet", help="Facet to report on."),
) -> None:
    """Print the Layer B WARN-level quality report (metrics + warnings)."""
    fs = _build(config)
    typer.echo(str(fs.build_quality_report(facet=facet)))


@app.command("sweep-layer-b")
def sweep_layer_b(
    config: Path = ConfigOption,
    facet: str = typer.Option("foods", "--facet", help="Facet to sweep."),
) -> None:
    """Run the Layer B tuning sweep (non-mutating) and print a ranked table.

    Runs the full 160-config Cartesian grid as dry-run builds — slow.
    """
    fs = _build(config)
    typer.echo(str(fs.sweep_layer_b(facet=facet)))


@app.command("build-layer-c")
def build_layer_c(
    config: Path = ConfigOption,
    facet: str = typer.Option("foods", "--facet", help="Facet to summarize."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run without persisting."),
) -> None:
    """Build Layer C — one summary card per Layer B theme."""
    fs = _build(config)
    report = fs.build_layer_c(facet=facet, dry_run=dry_run)
    typer.echo(str(report))


@app.command("bench-layer-c")
def bench_layer_c(
    config: Path = ConfigOption,
    facet: str = typer.Option("foods", "--facet", help="Facet to benchmark."),
    themes: int = typer.Option(5, "--themes", help="How many largest themes."),
    out: str = typer.Option(None, "--out", help="Output dir for the JSON."),
) -> None:
    """Benchmark all extractive methods over the largest themes (read-only)."""
    fs = _build(config)
    results = fs.benchmark_layer_c(facet=facet, themes=themes, out=out)
    for tid, rows in results.items():
        typer.echo(f"\n# {tid}")
        for r in rows:
            typer.echo(
                f"  {r.method:10} {r.summary_length_chars:>6} chars "
                f"{r.execution_time_ms:>5} ms"
            )


@app.command("build-all")
def build_all(config: Path = ConfigOption) -> None:
    """Run annotate → build-layer-a → attach → build-layer-b → build-layer-c."""
    _run_phase(_build(config), "build-all", "build")


@app.command()
def query(
    text: str = typer.Argument(..., help="Free-text question to ask the graph."),
    config: Path = ConfigOption,
) -> None:
    """Interactive retrieval against the built graph."""
    fs = _build(config)
    try:
        answer = fs.query(text)
    except NotImplementedError as e:
        typer.echo(f"[foodscholar] {e}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(answer.model_dump_json(indent=2))


@app.command()
def version() -> None:
    """Print the installed foodscholar version and exit."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
