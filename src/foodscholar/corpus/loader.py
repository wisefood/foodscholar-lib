"""Load chunks from a parquet (or jsonl) file into list[Chunk]."""

from __future__ import annotations

import json
from pathlib import Path

from foodscholar.io.chunk import Chunk


def load_chunks(path: str | Path) -> list[Chunk]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"chunks file not found: {p}")

    if p.suffix in {".jsonl", ".ndjson"}:
        return [Chunk.model_validate(json.loads(line)) for line in p.read_text().splitlines() if line.strip()]

    if p.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(p)
        return [Chunk.model_validate(row) for row in table.to_pylist()]

    raise ValueError(f"unsupported chunks file extension: {p.suffix}")
