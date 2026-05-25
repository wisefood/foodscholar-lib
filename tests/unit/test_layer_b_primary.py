"""Per-pass-aware primary chunk picker.

Pass 1 (similarity) themes: closest-to-centroid in embedding space.
Pass 2 (relatedness) themes: max sum-of-edge-weights to other members.
Merged themes: max of (centroid-score, edge-degree-score) per chunk,
choose the chunk with the highest max. Lex-first chunk_id is the
deterministic tie-breaker.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("igraph")

from foodscholar.layer_b.primary import pick_primary  # noqa: E402


def test_pick_primary_similarity_chooses_closest_to_centroid() -> None:
    """Sim theme: the chunk whose normalized vector is closest to the
    centroid wins."""
    import igraph as ig

    chunk_ids = {"a", "b", "c"}
    embeddings = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.5, 0.5]),
        "c": np.array([0.0, 1.0]),
    }
    # Theme's centroid mid-way: a should be closest to centroid only if
    # centroid is biased toward +x. Make centroid = [1, 0] explicitly:
    centroid = [1.0, 0.0]
    sim_graph = ig.Graph()  # not used for sim picker
    rel_graph = ig.Graph()
    primary = pick_primary(
        chunk_ids=chunk_ids,
        discovery_pass="similarity",
        embeddings=embeddings,
        centroid=centroid,
        sim_graph=sim_graph,
        rel_graph=rel_graph,
    )
    assert primary == "a"


def test_pick_primary_relatedness_chooses_max_edge_degree() -> None:
    """Rel theme: chunk with highest sum-of-edge-weights to others wins."""
    import igraph as ig

    g = ig.Graph()
    g.add_vertices(3)
    g.vs["chunk_id"] = ["a", "b", "c"]
    # b is the hub: edges (a,b)=1.0 and (b,c)=1.0 → degree-weight 2.0
    # a has degree-weight 1.0, c has 1.0
    g.add_edges([(0, 1), (1, 2)])
    g.es["weight"] = [1.0, 1.0]

    primary = pick_primary(
        chunk_ids={"a", "b", "c"},
        discovery_pass="relatedness",
        embeddings={},
        centroid=None,
        sim_graph=ig.Graph(),
        rel_graph=g,
    )
    assert primary == "b"


def test_pick_primary_merged_uses_max_of_both_scores() -> None:
    """Merged theme: take max(centroid-score, edge-degree-score) per chunk;
    the chunk with the highest max wins."""
    import igraph as ig

    chunk_ids = {"a", "b"}
    embeddings = {
        "a": np.array([1.0, 0.0]),   # close to centroid
        "b": np.array([-1.0, 0.0]),  # far from centroid
    }
    centroid = [1.0, 0.0]
    # Rel graph: a is isolated, b is in one edge
    rel = ig.Graph()
    rel.add_vertices(2)
    rel.vs["chunk_id"] = ["a", "b"]
    rel.add_edges([(0, 1)])
    rel.es["weight"] = [0.3]
    # a's max-of(centroid=1.0, edge=0.3) = 1.0
    # b's max-of(centroid=-1.0, edge=0.3) = 0.3
    primary = pick_primary(
        chunk_ids=chunk_ids,
        discovery_pass="merged",
        embeddings=embeddings,
        centroid=centroid,
        sim_graph=ig.Graph(),
        rel_graph=rel,
    )
    assert primary == "a"


def test_pick_primary_lex_first_tie_break() -> None:
    """When the per-pass score ties, lex-first chunk_id breaks the tie —
    same chunks → same primary across runs (audit determinism)."""
    import igraph as ig

    chunk_ids = {"b", "a"}
    embeddings = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([1.0, 0.0]),  # identical → identical cosine to centroid
    }
    primary = pick_primary(
        chunk_ids=chunk_ids,
        discovery_pass="similarity",
        embeddings=embeddings,
        centroid=[1.0, 0.0],
        sim_graph=ig.Graph(),
        rel_graph=ig.Graph(),
    )
    assert primary == "a"


def test_pick_primary_single_chunk_returns_it() -> None:
    """One-chunk theme: that chunk is the primary, regardless of pass."""
    import igraph as ig

    chunk_ids = {"solo"}
    primary = pick_primary(
        chunk_ids=chunk_ids,
        discovery_pass="similarity",
        embeddings={"solo": np.array([1.0])},
        centroid=[1.0],
        sim_graph=ig.Graph(),
        rel_graph=ig.Graph(),
    )
    assert primary == "solo"
