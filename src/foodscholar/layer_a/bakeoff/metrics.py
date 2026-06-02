"""Pure metric functions over a MethodResult. No ontology, no I/O."""

from __future__ import annotations

import statistics

from foodscholar.layer_a.bakeoff.result import MethodResult, node_depths


def coverage(result: MethodResult, mentioned_leaves: set[str]) -> float:
    """Fraction of mentioned leaves that are homed under some node."""
    if not mentioned_leaves:
        return 0.0
    homed = sum(1 for fid in mentioned_leaves if fid in result.leaf_home)
    return homed / len(mentioned_leaves)


def fan_out(result: MethodResult) -> tuple[int, float]:
    """(max, median) children over internal (non-leaf) nodes."""
    sizes = [len(kids) for kids in result.edges.values() if kids]
    if not sizes:
        return 0, 0.0
    return max(sizes), float(statistics.median(sizes))


def tree_depth(result: MethodResult) -> tuple[int, float]:
    """(max, median) depth over the nodes users land on (homed leaf homes)."""
    depths = node_depths(result)
    home_depths = [
        depths[home] for home in result.leaf_home.values() if home in depths
    ]
    if not home_depths:
        return 0, 0.0
    return max(home_depths), float(statistics.median(home_depths))


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round((pct / 100) * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def findability(result: MethodResult, query_leaves: list[str], *, k: int) -> dict:
    """For each query leaf, clicks = depth of its home node from root.

    Unreachable leaves (no home) are excluded from the click stats but counted
    against pct_reachable and pct_within_k.
    """
    depths = node_depths(result)
    clicks: list[int] = []
    reachable = 0
    for fid in query_leaves:
        home = result.leaf_home.get(fid)
        if home is not None and home in depths:
            reachable += 1
            clicks.append(depths[home])
    total = len(query_leaves) or 1
    sorted_clicks = sorted(clicks)
    within_k = sum(1 for c in clicks if c <= k)
    return {
        "median_clicks": float(statistics.median(sorted_clicks)) if sorted_clicks else 0.0,
        "p90_clicks": _percentile([float(c) for c in sorted_clicks], 90),
        "pct_within_k": within_k / total,
        "pct_reachable": reachable / total,
    }


def sample_query_leaves(leaf_freq: dict[str, int], *, n: int) -> list[str]:
    """Deterministic stratified sample of leaves: take the most frequent half
    and the least frequent half so both common and rare foods are tested.
    No RNG (reproducible across runs)."""
    if n <= 0 or not leaf_freq:
        return []
    by_freq = sorted(leaf_freq, key=lambda fid: (-leaf_freq[fid], fid))
    if n >= len(by_freq):
        return by_freq
    head = n // 2
    tail = n - head
    return by_freq[:head] + by_freq[head:head + tail]
