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


@app.command("build-relations")
def build_relations(
    config: Path = ConfigOption,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Extract and report without writing to the stores."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-extract chunks already covered and rebuild."
    ),
) -> None:
    """Build Layer 0 — typed relations between entities, grounded in FoodOn."""
    fs = _build(config)
    try:
        meta = fs.build_relations(dry_run=dry_run, force=force)
    except RuntimeError as e:
        typer.echo(f"[foodscholar] {e}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"relations={meta.record_count} artifact={meta.artifact_id}")


# Typer options are module-level singletons so they are constructed once, the
# same pattern as ConfigOption above (and what B008 asks for).
PdfDirOption = typer.Option(..., "--pdf-dir", help="Directory of source PDFs.")
OutDirOption = typer.Option(..., "--out-dir", help="Where to write corpus CSVs.")
MetadataCsvOption = typer.Option(
    None, "--metadata-csv", help="Per-document metadata (required for guides)."
)
ExcludedPagesOption = typer.Option(
    None, "--excluded-pages", help="removed_pages manifest."
)


@app.command("chunk-corpus")
def chunk_corpus(
    config: Path = ConfigOption,
    pdf_dir: Path = PdfDirOption,
    out_dir: Path = OutDirOption,
    source_type: str = typer.Option("guide", "--source-type", help="guide|textbook."),
    metadata_csv: Path = MetadataCsvOption,
    excluded_pages: Path = ExcludedPagesOption,
    force: bool = typer.Option(
        False, "--force", help="Overwrite an existing corpus. See the warning below."
    ),
) -> None:
    """Chunk PDFs into the corpus CSVs `ingest` reads.

    Re-chunking an ingested corpus assigns new chunk ids under the default
    strategy, orphaning existing relations and attachments — hence --force.
    """
    fs = _build(config)
    try:
        written = fs.chunk_documents(
            pdf_dir,
            out_dir=out_dir,
            source_type=source_type,
            metadata_csv=metadata_csv,
            excluded_pages=excluded_pages,
            force=force,
        )
    except Exception as e:
        typer.echo(f"[foodscholar] {e}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"wrote {len(written)} corpus CSV(s) to {out_dir}")


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
    k: int = typer.Option(5, "--top-k", "-k", help="Passages to return."),
) -> None:
    """Retrieve passages for a question — ranked evidence, not an answer.

    The library retrieves and stops there; formulating an answer is the
    consuming QA pipeline's job. Each row carries the three branch scores so a
    ranking can be inspected rather than taken on trust.
    """
    import json

    fs = _build(config)
    hits, trace = fs.retrieve(text, k=k)
    if trace.degraded:
        typer.echo(f"[foodscholar] degraded: {'; '.join(trace.degraded)}", err=True)
    typer.echo(
        json.dumps(
            {
                "query": text,
                "trace": {
                    "candidates": trace.candidates,
                    "relations": trace.relations,
                    "subgraph_nodes": trace.subgraph_nodes,
                    "branches": trace.branches_used,
                },
                "hits": [
                    {
                        "chunk_id": h.chunk_id,
                        "score": round(h.score, 4),
                        "text_sim": round(h.text_sim, 4),
                        "triplet_sim": round(h.triplet_sim, 4),
                        "ppr_score": round(h.ppr_score, 4),
                        "text": h.text[:300],
                    }
                    for h in hits
                ],
            },
            indent=2,
        )
    )


@app.command()
def version() -> None:
    """Print the installed foodscholar version and exit."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
