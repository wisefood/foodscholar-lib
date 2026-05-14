"""Source-type aware adapters. Stub — populated when concrete sources land."""

from __future__ import annotations

from foodscholar.io.chunk import Chunk, SourceType


def filter_by_source(chunks: list[Chunk], source_type: SourceType) -> list[Chunk]:
    return [c for c in chunks if c.source_type == source_type]
