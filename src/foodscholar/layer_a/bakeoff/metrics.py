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
