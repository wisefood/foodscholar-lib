"""Outputs: corpus CSV rows for ``fs.ingest``, neutral JSONL for evaluation.

The corpus CSV keeps the four-column contract the library already reads
(``chunk_id, chunk_text, type, chunk_metadata``). Everything v2 adds —
section path, page span, per-page regions, adjacency, identity, derivation —
travels in ``chunk_metadata``, which Elasticsearch maps as ``flattened``, so
no index mapping change is needed to filter on it. ``chunk_text`` is the
faithful passage; the context header is stored once in the metadata and the
retrieval text is derived from the two.

This is the only module in the package that knows the corpus contract, and it
still imports nothing from the rest of the library.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from foodscholar.chunking.assembler import AssembledChunk
from foodscholar.chunking.ir import DocumentMeta

CHUNKER_VERSION = "2.0.0.dev0"
CORPUS_COLUMNS = ["chunk_id", "chunk_text", "type", "chunk_metadata"]


def chunk_metadata(chunk: AssembledChunk, meta: DocumentMeta) -> dict[str, Any]:
    """The ``chunk_metadata`` dict for one chunk.

    Domain extras come first so the core keys always win. The v1 keys
    (``file``, ``heading``, ``page_number``) are still written, so readers that
    predate v2 keep working: ``page_number`` is the first page, as before.
    """
    out: dict[str, Any] = dict(meta.extra)
    labels = meta.page_labels
    core: dict[str, Any] = {
        "source_id": meta.source_id,
        "file": meta.file or meta.source_id,
        "title": meta.title,
        "short_name": meta.short_name,
        "heading": chunk.section_path[-1] if chunk.section_path else None,
        "section_path": list(chunk.section_path),
        "page_number": chunk.page_start,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "pages": list(chunk.pages),
        "printed_page_start": labels.get(chunk.page_start) if chunk.page_start else None,
        "printed_page_end": labels.get(chunk.page_end) if chunk.page_end else None,
        "regions": [r.model_dump(mode="json", exclude_none=True) for r in chunk.regions],
        "language": chunk.language or meta.language,
        "licence": meta.licence,
        "edition_pair_id": meta.edition_pair_id,
        "genre": meta.genre,
        "context_header": chunk.context_header,
        "section_labels": list(chunk.section_labels),
        # One labelled section → that section's type; a whole or mixed record
        # leaves it out and the reader falls back to the source type's default.
        "section_type": chunk.section_types[0] if len(chunk.section_types) == 1 else None,
        "figure_ids": list(chunk.figure_ids),
        "table_ids": list(chunk.table_ids),
        "element_ids": list(chunk.element_ids),
        "ordinal": chunk.ordinal,
        "prev_chunk_id": chunk.prev_chunk_id,
        "next_chunk_id": chunk.next_chunk_id,
        "parent_id": chunk.parent_id,
        "chunk_version_id": chunk.chunk_version_id,
        "supersedes": list(chunk.supersedes),
        "document_version_id": meta.document_version_id or None,
        "extraction_run_id": meta.extraction_run_id or None,
        "chunker_version": CHUNKER_VERSION,
        "tokens": chunk.total_tokens,
    }
    out.update(core)
    return {k: v for k, v in out.items() if v not in (None, "", [])}


def to_corpus_rows(chunks: Sequence[AssembledChunk], meta: DocumentMeta) -> list[dict[str, str]]:
    """Corpus CSV rows. Requires identity to have run (`chunk_id` set)."""
    rows: list[dict[str, str]] = []
    for c in chunks:
        if not c.chunk_id:
            raise ValueError("chunk_id is empty — run foodscholar.chunking.identity.assign_identity first")
        rows.append(
            {
                "chunk_id": c.chunk_id,
                "chunk_text": c.source_text,
                "type": meta.source_type,
                "chunk_metadata": repr(chunk_metadata(c, meta)),
            }
        )
    return rows


def write_corpus_csv(rows: Iterable[dict[str, str]], path: str | Path) -> Path:
    """Write rows in the shape ``corpus/csv_reader.py`` reads. Atomic."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CORPUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out)
    return out


def to_records(chunks: Sequence[AssembledChunk], meta: DocumentMeta) -> list[dict[str, Any]]:
    """The neutral chunk format the evaluation harness reads (one dict per chunk).

    Required keys are ``doc_id`` and ``text``; the rest are optional so any
    pipeline's output can be scored — see `foodscholar.chunking.evaluation`.
    """
    out: list[dict[str, Any]] = []
    for c in chunks:
        record = {
            "chunk_id": c.chunk_id,
            "doc_id": meta.source_id,
            "text": c.source_text,
            "context_header": c.context_header,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "section": c.section_path[-1] if c.section_path else None,
            "section_path": list(c.section_path),
            "section_type": c.section_types[0] if len(c.section_types) == 1 else None,
            "collection_id": meta.extra.get("collection_id"),
            "parent_id": c.parent_id,
            "ordinal": c.ordinal,
            "tokens": c.total_tokens,
            "chunk_version_id": c.chunk_version_id,
            "supersedes": list(c.supersedes),
        }
        out.append(record)
    return out


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)
    return out
