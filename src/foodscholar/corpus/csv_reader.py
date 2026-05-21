"""Readers for the current FoodScholar corpus CSV format.

The legacy corpus files use one uniform shape:

    chunk_id, chunk_text, type, chunk_metadata

`chunk_metadata` is a Python literal dict string. We preserve it as
`Chunk.source_metadata` and derive only the core fields needed by FoodScholar.
"""

from __future__ import annotations

import ast
import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from foodscholar.io.chunk import Chunk, SectionType, SourceType

REQUIRED_COLUMNS = {"chunk_id", "chunk_text", "type", "chunk_metadata"}

# Allow up to 10MB per CSV field. Large abstracts and full-document
# chunks routinely exceed the stdlib default and would otherwise raise
# `_csv.Error: field larger than field limit`.
csv.field_size_limit(10 * 1024 * 1024)


def iter_csv_chunks(path: str | Path, *, strict: bool = True) -> Iterator[Chunk]:
    """Yield normalized `Chunk` objects from one legacy corpus CSV file."""
    p = Path(path)
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"{p} is missing required columns: {sorted(missing)}")

        for row in reader:
            try:
                yield _row_to_chunk(row)
            except Exception:
                if strict:
                    raise
                continue


def _row_to_chunk(row: dict[str, str | None]) -> Chunk:
    chunk_id = _required(row, "chunk_id")
    text = _required(row, "chunk_text")
    source_type = _source_type(_required(row, "type"))
    metadata = _parse_metadata(row.get("chunk_metadata") or "")
    section_type = _section_type(source_type)

    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=_source_doc_id(source_type, metadata, chunk_id),
        source_type=source_type,
        section_type=section_type,
        year=_parse_year(metadata.get("year")),
        source_metadata=metadata,
    )


def _required(row: dict[str, str | None], key: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"missing required value: {key}")
    return value


def _parse_metadata(raw: str) -> dict[str, object]:
    if not raw.strip():
        return {}
    value = ast.literal_eval(raw)
    if not isinstance(value, dict):
        raise ValueError("chunk_metadata must parse to a dict")
    return {str(k): _jsonable(v) for k, v in value.items()}


def _jsonable(value: Any) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _source_type(raw: str) -> SourceType:
    normalized = raw.strip().lower()
    if normalized not in {"abstract", "textbook", "guide"}:
        raise ValueError(f"unsupported chunk type: {raw!r}")
    return normalized  # type: ignore[return-value]


def _section_type(source_type: SourceType) -> SectionType:
    if source_type == "abstract":
        return "abstract"
    if source_type == "guide":
        return "guideline"
    return "textbook"


def _source_doc_id(
    source_type: SourceType, metadata: dict[str, object], chunk_id: str
) -> str:
    if source_type == "abstract":
        for key in ("DOI", "doi", "title"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return chunk_id

    value = metadata.get("file")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return chunk_id


def _parse_year(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None
