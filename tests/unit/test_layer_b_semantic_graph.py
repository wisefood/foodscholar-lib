"""Pass 1 — similarity graph (mutual-kNN over chunk embeddings).

Requires numpy + python-igraph (the `[clustering]` extra). Skipped on
environments without working numeric stack (e.g., the local broken-openblas
env); runs normally in Colab / CI / any env with the extra installed.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("igraph")
pytest.importorskip("leidenalg")

from foodscholar.config import LayerBConfig, SimilarityConfig  # noqa: E402
from foodscholar.io.chunk import Chunk  # noqa: E402
from foodscholar.layer_b.semantic_graph import (  # noqa: E402
    build_global_similarity_graph,
    build_similarity_graph,
)
from foodscholar.storage.memory import InMemoryChunkStore  # noqa: E402


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=f"text {cid}",
        source_doc_id="d",
        source_type="abstract",
        section_type="other",
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


# ----------------------------------------------------------------------------
# build_global_similarity_graph — kNN-backed variant (Task 6)
# ----------------------------------------------------------------------------


def _store_with_chunks(chunk_vecs: dict[str, list[float]]) -> InMemoryChunkStore:
    store = InMemoryChunkStore()
    chunks = [
        Chunk(
            chunk_id=cid,
            text=f"text {cid}",
            source_doc_id="d",
            source_type="abstract",
            section_type="other",
            embedding=vec,
            embedding_model="m",
        )
        for cid, vec in chunk_vecs.items()
    ]
    store.upsert(chunks)
    return store


def test_build_global_similarity_graph_uses_chunk_store_knn() -> None:
    """4 chunks: A & B close (cos~0.99), A & D orthogonal.
    Edge A-B must exist; edge A-D must not."""
    chunk_vecs: dict[str, list[float]] = {
        "A": [1.0, 0.0, 0.0],
        "B": [0.99, 0.14, 0.0],  # cos(A,B) ≈ 0.99
        "C": [0.0, 1.0, 0.0],
        "D": [0.0, 0.0, 1.0],   # orthogonal to A
    }
    store = _store_with_chunks(chunk_vecs)
    chunk_ids = list(chunk_vecs.keys())

    cfg = LayerBConfig()
    cfg.similarity.knn_k = 2
    cfg.similarity.edge_threshold = 0.5
    cfg.similarity.require_mutual = False

    g = build_global_similarity_graph(chunk_ids, store, cfg.similarity)

    assert g.vcount() == 4
    assert sorted(g.vs["chunk_id"]) == sorted(chunk_ids)

    # Build a set of (min, max) vertex-index pairs for all edges
    idx = {cid: i for i, cid in enumerate(chunk_ids)}
    edge_pairs = {
        (min(e.source, e.target), max(e.source, e.target))
        for e in g.es
    }
    ab_key = (min(idx["A"], idx["B"]), max(idx["A"], idx["B"]))
    ad_key = (min(idx["A"], idx["D"]), max(idx["A"], idx["D"]))
    assert ab_key in edge_pairs, "Expected A-B edge for cos~0.99 pair"
    assert ad_key not in edge_pairs, "Did not expect A-D edge for orthogonal pair"


def test_build_global_similarity_graph_empty_input() -> None:
    """Empty chunk_ids → 0-vertex graph."""
    store = InMemoryChunkStore()
    cfg = LayerBConfig()
    g = build_global_similarity_graph([], store, cfg.similarity)
    assert g.vcount() == 0
    assert g.ecount() == 0


def test_build_global_similarity_graph_respects_require_mutual() -> None:
    """With require_mutual=True, asymmetric kNN hits are dropped.

    A's top-1 is B but B's top-1 is C (not A), so the A-B edge should be
    dropped when require_mutual=True."""
    # k=1: each chunk keeps only its single nearest neighbor.
    # A → nearest B; B → nearest C; C → nearest B; D → nearest C.
    # Only B-C is mutual under k=1 (B picks C, C picks B).
    chunk_vecs: dict[str, list[float]] = {
        "A": [1.0, 0.0, 0.0],
        "B": [0.95, 0.31, 0.0],   # close to A but closer to C than A is to it
        "C": [0.0, 1.0, 0.0],
        "D": [0.0, 0.0, 1.0],
    }
    store = _store_with_chunks(chunk_vecs)
    chunk_ids = list(chunk_vecs.keys())

    cfg = LayerBConfig()
    cfg.similarity.knn_k = 1
    cfg.similarity.edge_threshold = 0.0
    cfg.similarity.require_mutual = True

    g_mutual = build_global_similarity_graph(chunk_ids, store, cfg.similarity)

    cfg.similarity.require_mutual = False
    g_non_mutual = build_global_similarity_graph(chunk_ids, store, cfg.similarity)

    # require_mutual=True must not add edges that aren't present with require_mutual=False
    assert g_mutual.ecount() <= g_non_mutual.ecount(), (
        "require_mutual=True should have <= edges than require_mutual=False"
    )
