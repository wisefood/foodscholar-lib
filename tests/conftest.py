"""Shared pytest fixtures.

Provides:
  - mini_chunks: a tiny synthetic corpus of 5 chunks across nutrition topics
  - mock_embedder: deterministic toy embedder (no torch dependency)
  - mock_llm: scripted LLM that echoes a template with citation markers
  - chunk_store / graph_store: fresh in-memory stores per test
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import pytest

from foodscholar.io.chunk import Chunk
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore


class MockEmbedder:
    model_id = "mock-embedder-v0"

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [(h[i % len(h)] / 255.0) for i in range(self._dim)]
            vectors.append(vec)
        return vectors


class MockLLM:
    model_id = "mock-llm-v0"

    def __init__(self, response: str = "Mock answer citing [CHUNK].") -> None:
        self._response = response

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return self._response


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def mini_chunks(mock_embedder: MockEmbedder) -> list[Chunk]:
    raw = [
        ("c1", "Mediterranean diet rich in olive oil reduces cardiovascular risk.", "abstract", "abstract"),
        ("c2", "Whole grain consumption is associated with lower mortality.", "abstract", "results"),
        ("c3", "Peanut allergy management guidelines for paediatric populations.", "guide", "guideline"),
        ("c4", "Iron-rich foods include legumes, red meat, and fortified cereals.", "textbook", "textbook"),
        ("c5", "Plant-based dietary patterns are linked to improved metabolic markers.", "abstract", "discussion"),
    ]
    embeddings = mock_embedder.embed([r[1] for r in raw])
    chunks: list[Chunk] = []
    for (cid, text, source, section), emb in zip(raw, embeddings):
        chunks.append(
            Chunk(
                chunk_id=cid,
                text=text,
                source_doc_id=f"doc-{cid}",
                source_type=source,  # type: ignore[arg-type]
                section_type=section,  # type: ignore[arg-type]
                year=2024,
                embedding=emb,
                embedding_model=mock_embedder.model_id,
            )
        )
    return chunks


@pytest.fixture
def chunk_store() -> Iterator[InMemoryChunkStore]:
    yield InMemoryChunkStore()


@pytest.fixture
def graph_store() -> Iterator[InMemoryGraphStore]:
    yield InMemoryGraphStore()
