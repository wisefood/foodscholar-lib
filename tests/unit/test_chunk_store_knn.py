"""Tests for ChunkStore.knn_search_chunks across implementations."""
from __future__ import annotations

import pytest

from foodscholar.storage.protocols import ChunkStore


def test_chunk_store_protocol_has_knn_search_chunks():
    """ChunkStore protocol must expose knn_search_chunks for Layer B global pass."""
    assert hasattr(ChunkStore, "knn_search_chunks")
