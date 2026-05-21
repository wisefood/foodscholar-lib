"""Load chunks from parquet, jsonl, or legacy corpus CSV files."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from foodscholar.corpus.csv_reader import iter_csv_chunks
from foodscholar.io.chunk import Chunk


def iter_chunks(path: str | Path, *, strict: bool = True) -> Iterator[Chunk]:
    """Yield chunks from a file or directory.

    Directories are treated as legacy corpus directories and all `*.csv` files
    are read in sorted order for deterministic builds.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"chunks file not found: {p}")

    if p.is_dir():
        csvs = sorted(p.glob("*.csv"))
        if not csvs:
            raise ValueError(f"no .csv files found in directory: {p}")
        for csv_path in csvs:
            yield from iter_csv_chunks(csv_path, strict=strict)
        return

    if p.suffix == ".csv":
        yield from iter_csv_chunks(p, strict=strict)
        return

    if p.suffix in {".jsonl", ".ndjson"}:
        for line in p.read_text().splitlines():
            if line.strip():
                yield Chunk.model_validate(json.loads(line))
        return

    if p.suffix == ".parquet":
        table = pq.read_table(p)
        for row in table.to_pylist():
            yield _chunk_from_row(row)
        return

    raise ValueError(f"unsupported chunks file extension: {p.suffix}")


def load_chunks(path: str | Path, *, strict: bool = True) -> list[Chunk]:
    return list(iter_chunks(path, strict=strict))


def write_chunks_parquet(chunks: list[Chunk], path: str | Path) -> int:
    """Write normalized chunks to Parquet and return the number of records."""
    rows = []
    for chunk in chunks:
        row = chunk.model_dump(mode="json")
        # Source metadata is intentionally free-form and can contain mixed
        # types for the same key across sources (e.g. year int vs string).
        # Store it as JSON so Arrow doesn't infer a brittle struct schema.
        row["source_metadata"] = json.dumps(row["source_metadata"], sort_keys=True)
        rows.append(row)
    table = pa.Table.from_pylist(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    return len(rows)


def _chunk_from_row(row: dict[str, object]) -> Chunk:
    metadata = row.get("source_metadata")
    if isinstance(metadata, str):
        row = dict(row)
        row["source_metadata"] = json.loads(metadata) if metadata else {}
    return Chunk.model_validate(row)
