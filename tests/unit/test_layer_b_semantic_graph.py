"""Pass 1 — similarity graph (mutual-kNN over chunk embeddings).

Requires numpy + python-igraph (the `[clustering]` extra). Skipped on
environments without working numeric stack (e.g., the local broken-openblas
env); runs normally in Colab / CI / any env with the extra installed.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("igraph")

from foodscholar.config import SimilarityConfig  # noqa: E402
from foodscholar.io.chunk import Chunk  # noqa: E402
from foodscholar.layer_b.semantic_graph import build_similarity_graph  # noqa: E402


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=f"text {cid}",
        source_doc_id="d",
        source_type="abstract",
        section_type="abstract",
    )


def test_similarity_graph_has_one_node_per_chunk() -> None:
    """Every chunk gets a vertex regardless of whether it has edges — the
    builder downstream filters disconnected vertices via Leiden's
    min_community_size, not at graph construction."""
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    embeddings = {
        "a": np.array([1.0, 0.0, 0.0]),
        "b": np.array([0.95, 0.05, 0.0]),
        "c": np.array([0.0, 0.0, 1.0]),
    }
    cfg = SimilarityConfig(knn_k=1, edge_threshold=0.5, require_mutual=True)
    g = build_similarity_graph(chunks, embeddings, cfg)
    assert g.vcount() == 3
    assert sorted(g.vs["chunk_id"]) == ["a", "b", "c"]


def test_similarity_graph_mutual_knn_forms_edge() -> None:
    """k=1 mutual: a and b are each other's top-1 → edge forms;
    c has no mutual partner → isolated."""
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    embeddings = {
        "a": np.array([1.0, 0.0, 0.0]),
        "b": np.array([0.95, 0.05, 0.0]),
        "c": np.array([0.0, 0.0, 1.0]),
    }
    cfg = SimilarityConfig(knn_k=1, edge_threshold=0.5, require_mutual=True)
    g = build_similarity_graph(chunks, embeddings, cfg)
    assert g.ecount() == 1
    assert g.es["weight"][0] > 0.9


def test_similarity_graph_drops_below_edge_threshold() -> None:
    """Orthogonal vectors at the top-k cutoff still don't form edges if
    cosine < edge_threshold."""
    chunks = [_chunk("a"), _chunk("b")]
    embeddings = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.0, 1.0]),  # cosine = 0
    }
    cfg = SimilarityConfig(knn_k=1, edge_threshold=0.5, require_mutual=True)
    g = build_similarity_graph(chunks, embeddings, cfg)
    assert g.ecount() == 0


def test_similarity_graph_non_mutual_path() -> None:
    """require_mutual=False: top-k from either side suffices (deduped to
    undirected). Useful escape hatch when mutual is too sparse."""
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    embeddings = {
        "a": np.array([1.0, 0.0, 0.0]),
        "b": np.array([0.9, 0.1, 0.0]),
        "c": np.array([0.8, 0.2, 0.0]),
    }
    cfg = SimilarityConfig(knn_k=1, edge_threshold=0.5, require_mutual=False)
    g = build_similarity_graph(chunks, embeddings, cfg)
    assert g.ecount() >= 1


def test_similarity_graph_normalizes_unnormalized_input() -> None:
    """Defensive: even if embeddings aren't L2-normalized on read, cosine
    should still be correct."""
    chunks = [_chunk("a"), _chunk("b")]
    embeddings = {
        "a": np.array([10.0, 0.0, 0.0]),    # norm 10
        "b": np.array([100.0, 5.0, 0.0]),   # norm ~100
    }
    cfg = SimilarityConfig(knn_k=1, edge_threshold=0.5, require_mutual=True)
    g = build_similarity_graph(chunks, embeddings, cfg)
    assert g.ecount() == 1
    # cosine on normalized inputs is ~0.998 (mostly-collinear with x-axis)
    assert g.es["weight"][0] > 0.99


def test_similarity_graph_empty_returns_empty_graph() -> None:
    cfg = SimilarityConfig()
    g = build_similarity_graph([], {}, cfg)
    assert g.vcount() == 0
    assert g.ecount() == 0


def test_similarity_graph_single_chunk_no_edges() -> None:
    """N=1: no possible edges; should not crash on k > n-1."""
    chunks = [_chunk("solo")]
    embeddings = {"solo": np.array([1.0, 0.0])}
    cfg = SimilarityConfig(knn_k=5, edge_threshold=0.5)
    g = build_similarity_graph(chunks, embeddings, cfg)
    assert g.vcount() == 1
    assert g.ecount() == 0
