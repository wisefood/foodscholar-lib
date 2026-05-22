"""Candidate-graph clustering: connected components + size-cap splitting."""

from __future__ import annotations

from foodscholar.layer_a.semantic_consolidation.cluster import cluster_candidates
from foodscholar.layer_a.semantic_consolidation.models import CandidatePair


def _pair(a: str, b: str, cos: float = 0.9) -> CandidatePair:
    return CandidatePair(shelf_a=a, shelf_b=b, cosine_similarity=cos)


def test_transitive_chain_is_one_cluster() -> None:
    clusters = cluster_candidates(
        [_pair("a", "b"), _pair("b", "c"), _pair("d", "e")]
    )
    assert sorted(clusters) == [["a", "b", "c"], ["d", "e"]]


def test_members_sorted_for_determinism() -> None:
    clusters = cluster_candidates([_pair("z", "a"), _pair("a", "m")])
    assert clusters == [["a", "m", "z"]]


def test_no_candidates_no_clusters() -> None:
    assert cluster_candidates([]) == []


def test_oversized_cluster_split_on_weakest_edge() -> None:
    # Two tight triangles (cos 0.95) bridged by one weak edge (cos 0.90).
    # cap=3 must drop the bridge and yield the two triangles.
    edges = [
        _pair("a", "b", 0.95), _pair("b", "c", 0.95), _pair("a", "c", 0.95),
        _pair("c", "d", 0.90),  # weak bridge
        _pair("d", "e", 0.95), _pair("e", "f", 0.95), _pair("d", "f", 0.95),
    ]
    clusters = cluster_candidates(edges, max_cluster_size=3)
    assert sorted(clusters) == [["a", "b", "c"], ["d", "e", "f"]]


def test_split_drops_isolated_singletons() -> None:
    # A star: hub connected to 4 leaves, all equal weight, cap=2.
    # Splitting peels leaves off; a leaf alone is not a cluster.
    edges = [_pair("hub", leaf, 0.93) for leaf in ("w", "x", "y", "z")]
    clusters = cluster_candidates(edges, max_cluster_size=2)
    # Each surviving cluster has exactly 2 nodes (hub + one leaf at a time is
    # impossible since the hub is shared); what we assert is the hard cap holds
    # and no singleton leaks.
    assert all(2 <= len(c) <= 2 for c in clusters)
    assert all(len(c) >= 2 for c in clusters)


def test_respects_cap_exactly() -> None:
    # A 5-node path with descending weights; cap=4 drops the single weakest
    # edge, leaving a 4-node and a 1-node piece (singleton dropped).
    edges = [
        _pair("a", "b", 0.99), _pair("b", "c", 0.98),
        _pair("c", "d", 0.97), _pair("d", "e", 0.95),
    ]
    clusters = cluster_candidates(edges, max_cluster_size=4)
    assert clusters == [["a", "b", "c", "d"]]  # weakest edge d-e dropped, e isolated
