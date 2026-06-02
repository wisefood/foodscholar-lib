"""Common representation for a constructed Layer-A method + tree helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foodscholar.ontology import FoodOnAPI


@dataclass
class MethodResult:
    name: str
    root: str
    edges: dict[str, list[str]]
    labels: dict[str, str]
    counts: dict[str, int]
    leaf_home: dict[str, str]
    home_edge_type: dict[str, str]
    home_distance: dict[str, int] = field(default_factory=dict)  # leaf -> is-a steps to its home
    llm_calls: int = 0
    audit: list[dict] = field(default_factory=list)


def home_distance(leaf: str, home: str, ontology: FoodOnAPI) -> int:
    """Number of is-a steps from `leaf` up to its `home` node (0 if home == leaf).

    Shared by every adapter + the agentic method so the distance definition stays
    identical across methods. Counts the leaf's ancestors that lie at-or-below
    `home` — i.e. the nodes on the path from leaf up to and including home."""
    if home == leaf:
        return 0
    return len([
        a for a in ontology.id_to_ancestors(leaf)
        if a == home or ontology.is_subclass_of(a, home)
    ])


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


def _deepest_tree_home(
    leaf: str, tree_nodes: set[str], ontology: FoodOnAPI
) -> str | None:
    """The most-specific tree node that is an is-a ancestor-or-self of `leaf`."""
    if leaf in tree_nodes:
        return leaf
    candidates = [a for a in ontology.id_to_ancestors(leaf) if a in tree_nodes]
    if not candidates:
        return None
    # deepest = the candidate with the longest ancestor chain (closest to leaf)
    return max(candidates, key=lambda a: len(ontology.id_to_ancestors(a)))


def from_children_map(
    name: str,
    *,
    root: str,
    children_map: dict[str, list[str]],
    counts: dict[str, int],
    labels: dict[str, str],
    ontology: FoodOnAPI,
    mentioned_leaves: set[str],
) -> MethodResult:
    """Wrap a projection method's explicit FoodOn children_map. Leaves home to
    their deepest kept is-a ancestor (membership is is-a → faithful)."""
    tree_nodes = {root, *children_map.keys()}
    for kids in children_map.values():
        tree_nodes.update(kids)
    leaf_home: dict[str, str] = {}
    home_edge_type: dict[str, str] = {}
    home_dist: dict[str, int] = {}
    for leaf in mentioned_leaves:
        home = _deepest_tree_home(leaf, tree_nodes, ontology)
        if home is not None:
            leaf_home[leaf] = home
            home_edge_type[leaf] = "is-a"
            home_dist[leaf] = home_distance(leaf, home, ontology)
    return MethodResult(
        name=name, root=root,
        edges={p: list(kids) for p, kids in children_map.items()},
        labels=dict(labels), counts=dict(counts),
        leaf_home=leaf_home, home_edge_type=home_edge_type, home_distance=home_dist,
    )


def _shelf_node_id(shelf) -> str:
    return shelf.foodon_id or shelf.shelf_id


def from_shelves(
    name: str,
    shelves: list,  # list[Shelf]
    *,
    ontology: FoodOnAPI,
    mentioned_leaves: set[str],
) -> MethodResult:
    """Wrap a list[Shelf] (prune / grouping output) into a MethodResult."""
    node_of = {s.shelf_id: _shelf_node_id(s) for s in shelves}

    root = next((s for s in shelves if s.parent_shelf_id is None), None)
    root_id = node_of[root.shelf_id] if root else (shelves[0].shelf_id if shelves else "")

    edges: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    counts: dict[str, int] = {}
    for s in shelves:
        nid = node_of[s.shelf_id]
        labels[nid] = s.display_label or s.label
        counts[nid] = s.chunk_count
        if s.parent_shelf_id is not None and s.parent_shelf_id in node_of:
            edges.setdefault(node_of[s.parent_shelf_id], []).append(nid)

    shelf_nodes = {node_of[s.shelf_id] for s in shelves if s.foodon_id}
    leaf_home: dict[str, str] = {}
    home_edge_type: dict[str, str] = {}
    home_dist: dict[str, int] = {}
    for leaf in mentioned_leaves:
        home_shelf = next((s for s in shelves if leaf in s.see_also), None)
        if home_shelf is not None:
            home_node = node_of[home_shelf.shelf_id]
            leaf_home[leaf] = home_node
            anchor = home_shelf.foodon_id
            if anchor and ontology.is_subclass_of(leaf, anchor):
                home_edge_type[leaf] = "is-a"
                home_dist[leaf] = home_distance(leaf, anchor, ontology)
            else:
                # fabricated: leaf isn't structurally under the anchor → treat as
                # maximally-far (penalizes label-grouping in specificity).
                home_edge_type[leaf] = "fabricated"
                home_dist[leaf] = len(ontology.id_to_ancestors(leaf))
            continue
        home = _deepest_tree_home(leaf, shelf_nodes, ontology)
        if home is not None:
            leaf_home[leaf] = home
            home_edge_type[leaf] = "is-a"
            home_dist[leaf] = home_distance(leaf, home, ontology)
    return MethodResult(
        name=name, root=root_id, edges=edges, labels=labels, counts=counts,
        leaf_home=leaf_home, home_edge_type=home_edge_type, home_distance=home_dist,
    )
