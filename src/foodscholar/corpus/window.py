"""Moved to :mod:`foodscholar.chunking.window`; this path is kept for compatibility."""

from foodscholar.chunking.window import (
    DEFAULT_FINE_MAX_TOKENS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP,
    FineUnit,
    MergedChunk,
    build_overlapping_chunks,
    has_meaningful_text,
)

__all__ = [
    "DEFAULT_FINE_MAX_TOKENS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_OVERLAP",
    "FineUnit",
    "MergedChunk",
    "build_overlapping_chunks",
    "has_meaningful_text",
]
