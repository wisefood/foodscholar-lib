"""``foodscholar chunk`` — the chunker's command group, also ``foodscholar-chunk``.

It never builds the FoodScholar facade and never touches a store, so it runs
on a box that has only the ``[chunking]`` extra.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer

app = typer.Typer(
    name="chunk",
    help="Extract, assemble and evaluate chunks. Needs no stores or services.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOpt = typer.Option(None, "--config", "-c", help="Chunking YAML (standalone or a full FoodScholar config).")
TokenizerOpt = typer.Option(
    "hf", "--tokenizer", help="'hf' counts with chunking.embedder_tokenizer; 'whitespace' is a model-free estimate."
)

SourcesArg = typer.Argument(None, help="Source files, for the v2 heads (none implemented yet).")
IrOpt = typer.Option(None, "--ir", help="An IR JSON file, or a directory of them, to assemble.")
OutDirOpt = typer.Option(..., "--out", "-o", help="Output directory.")
PreviousOpt = typer.Option(None, "--previous", help="An earlier run's chunks.jsonl, to keep chunk ids.")
SourceTypeOpt = typer.Option(None, "--source-type", help="Source type of the given files, when no registry knows them.")
RunRegistryOpt = typer.Option(None, "--registry", exists=True, help="Source catalogue; overrides chunking.registry.")
QueriesOpt = typer.Option(None, "--queries", exists=True, help="Query JSONL for the retrieval stage (query_id, text, evidence, evidence_docs).")
InspectArg = typer.Argument(..., exists=True, help="An IR JSON or a chunk file (JSONL, CSV, parquet).")
ChunksOpt = typer.Option(..., "--chunks", exists=True, help="Chunks: JSONL, corpus CSV(s) or a parquet snapshot.")
GoldOpt = typer.Option(..., "--gold", exists=True, help="Gold statements: an FSKG workbook (.xlsx) or JSONL.")
RegistryOpt = typer.Option(None, "--registry", exists=True, help="Source catalogue, to map files and statements to sources.")
ReportOutOpt = typer.Option(None, "--out", "-o", help="Write the full report as JSON here.")
PageOffsetOpt = typer.Option(
    None,
    "--gold-page-offset",
    help="SOURCE_ID=N: add N to that source's gold pages, for a gold set known to be shifted. Repeatable; reported.",
)
EvalTokenizerOpt = typer.Option(
    "estimate", "--tokenizer", help="Token counts for hygiene: 'estimate' (chars/4), 'hf' (exact) or 'whitespace'."
)


def _config(path: Path | None):  # type: ignore[no-untyped-def]
    from foodscholar.chunking.config import ChunkingConfig

    return ChunkingConfig.from_yaml(path) if path else ChunkingConfig()


def _counter(kind: str, cfg):  # type: ignore[no-untyped-def]
    from foodscholar.chunking.text import WhitespaceTokenCounter, hf_token_counter

    if kind == "whitespace":
        return WhitespaceTokenCounter()
    if kind == "hf":
        return hf_token_counter(cfg.embedder_tokenizer)
    raise typer.BadParameter("--tokenizer must be 'hf' or 'whitespace'")


def _ir_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.json")) if path.is_dir() else [path]


@app.command()
def run(
    sources: list[Path] = SourcesArg,
    ir: Path = IrOpt,
    out: Path = OutDirOpt,
    previous: Path = PreviousOpt,
    source_type: str = SourceTypeOpt,
    registry: Path = RunRegistryOpt,
    config: Path = ConfigOpt,
    tokenizer: str = TokenizerOpt,
) -> None:
    """Extract and assemble documents: corpus CSVs for fs.ingest plus chunks.jsonl."""
    from foodscholar.chunking.chunker import Chunker, load_previous
    from foodscholar.chunking.ir import DocumentIR

    cfg = _config(config)
    if registry is not None:
        cfg = cfg.model_copy(update={"registry": registry})
    chunker = Chunker(cfg, token_counter=_counter(tokenizer, cfg))
    prior = load_previous(previous) if previous else None
    try:
        if ir is not None:
            documents = [DocumentIR.load(p) for p in _ir_files(ir)]
        elif sources:
            documents = [doc for src in sources for doc in chunker.extract(src, source_type=source_type)]
        else:
            raise typer.BadParameter("give source files or --ir")
        result = chunker.assemble(documents, previous=prior)
    except (NotImplementedError, ValueError, KeyError) as e:
        typer.echo(f"[foodscholar chunk] {e}", err=True)
        raise typer.Exit(code=2) from None
    written = result.write(out)
    s = result.identity.summary()
    typer.echo(
        f"{len(result.documents)} document(s), {len(result.chunks)} chunk(s) → {out} "
        f"(ids: {s['reused']} reused, {s['new']} new, {s['superseding']} superseding, {s['retired']} retired)"
    )
    for key, path in written.items():
        typer.echo(f"  {key:<14} {path}")


@app.command()
def inspect(path: Path = InspectArg) -> None:
    """Summarise an IR or a chunk file."""
    if path.suffix == ".json":
        from foodscholar.chunking.ir import DocumentIR

        doc = DocumentIR.load(path)
        kinds = Counter(el.kind for el in doc.elements)
        pages = sorted({r.page for el in doc.elements for r in el.regions})
        typer.echo(f"{doc.meta.source_id} ({doc.meta.source_type}, {doc.meta.genre}): {len(doc.elements)} elements")
        typer.echo("  " + ", ".join(f"{k}={n}" for k, n in sorted(kinds.items())))
        if pages:
            typer.echo(f"  pages {pages[0]}-{pages[-1]} ({len(pages)} with content), {len(doc.assets)} asset(s)")
        return
    from foodscholar.chunking.evaluation import hygiene, load_chunk_records

    typer.echo(json.dumps(hygiene(load_chunk_records(path)), indent=2))


@app.command("eval")
def evaluate_cmd(
    chunks: Path = ChunksOpt,
    gold: Path = GoldOpt,
    registry: Path = RegistryOpt,
    label: str = typer.Option("pipeline", "--label", help="Name shown in the report."),
    threshold: float = typer.Option(0.8, "--threshold", help="Content-token support needed to count a statement."),
    retrieval: bool = typer.Option(False, "--retrieval", help="Also run the BM25 statement-as-query smoke test (leaky)."),
    queries: Path = QueriesOpt,
    out: Path = ReportOutOpt,
    gold_page_offset: list[str] = PageOffsetOpt,
    tokenizer: str = EvalTokenizerOpt,
    config: Path = ConfigOpt,
) -> None:
    """Score chunks against gold statements: assembly coverage, pages, sections, hygiene."""
    from foodscholar.chunking.evaluation import (
        evaluate,
        evaluate_retrieval,
        load_chunk_records,
        load_fskg_statements,
        load_gold_jsonl,
        load_queries_jsonl,
        statement_queries,
    )
    from foodscholar.chunking.registry import SourceRegistry

    reg = SourceRegistry.load(registry) if registry else None
    statements = load_fskg_statements(gold, reg) if gold.suffix == ".xlsx" else load_gold_jsonl(gold)
    offsets: dict[str, int] = {}
    for item in gold_page_offset or []:
        sid, _, n = item.partition("=")
        try:
            offsets[sid.strip()] = int(n)
        except ValueError:
            raise typer.BadParameter(f"--gold-page-offset wants SOURCE_ID=N, got {item!r}") from None
    for s in statements:
        if s.source_id in offsets and s.page is not None:
            s.page += offsets[s.source_id]
    # Score only the documents the gold set covers, so hygiene describes them.
    sources = {s.source_id for s in statements if s.source_id}
    records = load_chunk_records(chunks, registry=reg)
    records = [r for r in records if r.doc_id in sources] or records
    if tokenizer != "estimate":
        counter = _counter(tokenizer, _config(config))
        for r in records:
            r.tokens = counter.special_tokens + counter.count(r.text)
    report = evaluate(statements, records, label=label, threshold=threshold)
    if offsets:
        report.totals["gold_page_offsets"] = offsets
    if queries is not None:
        report.retrieval = evaluate_retrieval(load_queries_jsonl(queries), statements, records, threshold=threshold)
    elif retrieval:
        report.retrieval = evaluate_retrieval(statement_queries(statements), statements, records, threshold=threshold)
    typer.echo(report.to_markdown())
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(with_outcomes=True), indent=2, default=str), encoding="utf-8")
        typer.echo(f"\nfull report → {out}")


if __name__ == "__main__":  # pragma: no cover
    app()
