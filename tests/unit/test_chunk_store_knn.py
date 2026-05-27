"""Tests for ChunkStore.knn_search_chunks across implementations."""
from __future__ import annotations

import pytest

from foodscholar.storage.protocols import ChunkStore


def test_chunk_store_protocol_has_knn_search_chunks():
    """ChunkStore protocol must expose knn_search_chunks for Layer B global pass."""
    assert hasattr(ChunkStore, "knn_search_chunks")


np = pytest.importorskip("numpy")

from foodscholar.io.chunk import Chunk
from foodscholar.storage.memory import InMemoryChunkStore


def _make_chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=f"text for {cid}",
        source_doc_id="doc1",
        source_type="textbook",
        section_type="other",
        embedding=vec,
        embedding_model="test-model",
    )


def test_in_memory_knn_search_returns_top_k_by_cosine():
    """Three chunks: query is closest to A, then B, then C. k=2 returns A,B."""
    store = InMemoryChunkStore()
    store.upsert([
        _make_chunk("A", [1.0, 0.0, 0.0]),
        _make_chunk("B", [0.8, 0.6, 0.0]),
        _make_chunk("C", [0.0, 0.0, 1.0]),
    ])
    result = store.knn_search_chunks(
        query_vector=[1.0, 0.0, 0.0], k=2, exclude_ids=None, candidate_ids=None,
    )
    assert [cid for cid, _ in result] == ["A", "B"]
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_in_memory_knn_search_excludes_ids():
    """exclude_ids drops the query chunk from results."""
    store = InMemoryChunkStore()
    store.upsert([
        _make_chunk("A", [1.0, 0.0, 0.0]),
        _make_chunk("B", [0.8, 0.6, 0.0]),
    ])
    result = store.knn_search_chunks(
        query_vector=[1.0, 0.0, 0.0], k=2, exclude_ids=["A"], candidate_ids=None,
    )
    assert [cid for cid, _ in result] == ["B"]


def test_in_memory_knn_search_restricts_to_candidate_ids():
    """candidate_ids restricts the search universe."""
    store = InMemoryChunkStore()
    store.upsert([
        _make_chunk("A", [1.0, 0.0, 0.0]),
        _make_chunk("B", [0.8, 0.6, 0.0]),
        _make_chunk("C", [0.0, 0.0, 1.0]),
    ])
    result = store.knn_search_chunks(
        query_vector=[1.0, 0.0, 0.0], k=10,
        exclude_ids=None, candidate_ids=["B", "C"],
    )
    ids = [cid for cid, _ in result]
    assert "A" not in ids
    assert set(ids) == {"B", "C"}


def test_in_memory_knn_search_handles_chunks_without_embeddings():
    """Chunks with embedding=None are silently skipped."""
    store = InMemoryChunkStore()
    store.upsert([
        _make_chunk("A", [1.0, 0.0, 0.0]),
        Chunk(
            chunk_id="B", text="no vec", source_doc_id="doc1",
            source_type="textbook", section_type="other",
            embedding=None, embedding_model=None,
        ),
    ])
    result = store.knn_search_chunks(
        query_vector=[1.0, 0.0, 0.0], k=10, exclude_ids=None, candidate_ids=None,
    )
    assert [cid for cid, _ in result] == ["A"]
