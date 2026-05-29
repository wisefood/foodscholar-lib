"""Tests for the storage `init()` contract.

Local stores (in-memory) implement `init()` as a no-op so the facade's
`fs.init()` works uniformly. Remote stores (`ElasticChunkStore`,
`Neo4jGraphStore`) provision their indexes/constraints — those paths are
exercised in the integration suite.
"""

from __future__ import annotations

import pytest

from foodscholar import FoodScholar
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore
from foodscholar.storage.protocols import ChunkStore, GraphStore


def test_in_memory_chunk_store_init_is_noop() -> None:
    store = InMemoryChunkStore()
    store.init()  # no exception
    store.init()  # repeated call also fine


def test_in_memory_graph_store_init_is_noop() -> None:
    store = InMemoryGraphStore()
    store.init()
    store.init()


def test_protocol_runtime_check_accepts_in_memory_stores() -> None:
    """Both InMemory stores must satisfy the updated Protocol (init() added)."""
    assert isinstance(InMemoryChunkStore(), ChunkStore)
    assert isinstance(InMemoryGraphStore(), GraphStore)


def test_facade_init_calls_both_stores() -> None:
    """fs.init() must call init() on both stores, no matter the backend."""
    calls: list[str] = []

    class _RecChunk:
        def init(self) -> None:
            calls.append("chunk")

        # All other ChunkStore methods are unused by the test.
        def upsert(self, chunks):  # type: ignore[no-untyped-def]
            pass

    class _RecGraph:
        def init(self) -> None:
            calls.append("graph")

        # Pass-through stubs so the GraphView attached to fs doesn't trip.
        def upsert_shelves(self, shelves):  # type: ignore[no-untyped-def]
            pass

    fs = FoodScholar.in_memory()
    fs.chunk_store = _RecChunk()  # type: ignore[assignment]
    fs.graph_store = _RecGraph()  # type: ignore[assignment]
    fs.init()
    assert calls == ["chunk", "graph"]


def test_neo4j_resolve_password_prefers_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit password beats the env-var fallback."""
    from foodscholar.storage.neo4j import _resolve_password

    monkeypatch.setenv("NEO4J_PASSWORD", "from-env")
    assert _resolve_password("from-config") == "from-config"


def test_neo4j_resolve_password_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from foodscholar.storage.neo4j import _resolve_password

    monkeypatch.setenv("NEO4J_PASSWORD", "from-env")
    assert _resolve_password(None) == "from-env"
    assert _resolve_password("") == "from-env"


def test_neo4j_resolve_password_raises_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from foodscholar.storage.neo4j import _resolve_password

    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="Neo4j password is not configured"):
        _resolve_password(None)
