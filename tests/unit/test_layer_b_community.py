"""Leiden community-detection runner for Layer B."""

from __future__ import annotations

import pytest

pytest.importorskip("igraph")
pytest.importorskip("leidenalg")

from foodscholar.config import LeidenConfig
from foodscholar.layer_b.community import run_leiden


def _two_cluster_graph():  # type: ignore[no-untyped-def]
    """Two cliques of 4, weakly bridged. Used by several tests."""
    import igraph as ig

    g = ig.Graph()
    g.add_vertices(8)
    g.vs["chunk_id"] = [f"c{i}" for i in range(8)]
    a_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    b_edges = [(4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]
    bridge = [(3, 4)]
    g.add_edges(a_edges + b_edges + bridge)
    g.es["weight"] = [1.0] * len(a_edges + b_edges) + [0.1] * len(bridge)
    return g


def test_run_leiden_finds_two_communities() -> None:
    g = _two_cluster_graph()
    cfg = LeidenConfig(min_community_size=2, random_state=42)
    communities = run_leiden(g, cfg)
    assert len(communities) == 2
    assert all(isinstance(c, set) for c in communities)
    # Every node ends up in exactly one community.
    union: set[int] = set()
    for c in communities:
        union |= c
    assert union == set(range(8))


def test_run_leiden_filters_below_min_community_size() -> None:
    """min_community_size=5 against two 4-cliques drops both."""
    g = _two_cluster_graph()
    cfg = LeidenConfig(min_community_size=5, random_state=42)
    assert run_leiden(g, cfg) == []


def test_run_leiden_deterministic_with_fixed_seed() -> None:
    """The determinism contract — same chunks + same seed = same partition.
    This is the load-bearing guarantee for audit cross-store parity."""
    g = _two_cluster_graph()
    cfg = LeidenConfig(min_community_size=2, random_state=42)
    out1 = sorted(sorted(c) for c in run_leiden(g, cfg))
    out2 = sorted(sorted(c) for c in run_leiden(g, cfg))
    assert out1 == out2


def test_run_leiden_empty_graph_returns_empty() -> None:
    import igraph as ig

    g = ig.Graph()
    cfg = LeidenConfig(min_community_size=1, random_state=42)
    assert run_leiden(g, cfg) == []


def test_run_leiden_graph_with_no_edges_returns_empty() -> None:
    """All-isolates graph (e.g., from an edge_threshold that filtered every
    edge): Leiden returns one singleton per node; min_community_size filters
    them out and we get []."""
    import igraph as ig

    g = ig.Graph()
    g.add_vertices(5)
    g.es["weight"] = []  # explicit empty edge weights
    cfg = LeidenConfig(min_community_size=2, random_state=42)
    assert run_leiden(g, cfg) == []
