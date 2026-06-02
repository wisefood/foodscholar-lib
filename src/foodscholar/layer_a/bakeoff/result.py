"""Common representation for a constructed Layer-A method + tree helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class MethodResult:
    name: str
    root: str
    edges: dict[str, list[str]]
    labels: dict[str, str]
    counts: dict[str, int]
    leaf_home: dict[str, str]
    home_edge_type: dict[str, str]
    llm_calls: int = 0
    audit: list[dict] = field(default_factory=list)


def node_depths(result: MethodResult) -> dict[str, int]:
    """BFS depth of every node reachable from `root` (root = 0). Cycle-safe."""
    depths: dict[str, int] = {result.root: 0}
    queue: deque[str] = deque([result.root])
    while queue:
        node = queue.popleft()
        for child in result.edges.get(node, []):
            if child not in depths:
                depths[child] = depths[node] + 1
                queue.append(child)
    return depths
