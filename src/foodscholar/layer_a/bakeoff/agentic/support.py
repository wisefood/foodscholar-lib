"""Per-node chunk support over the food-product subtree (descendant roll-up)."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foodscholar.ontology import FoodOnAPI


def rollup_support(
    leaf_chunks: dict[str, set[str]], ontology: FoodOnAPI, *, root: str
) -> dict[str, set[str]]:
    """node id -> set of chunk ids mentioning it or any is-a descendant.

    Only nodes that are `root` or a subclass of `root` are rolled up onto."""
    node_chunks: dict[str, set[str]] = defaultdict(set)
    for leaf, chunk_ids in leaf_chunks.items():
        if leaf not in ontology:
            continue
        targets = [leaf] + [
            a for a in ontology.id_to_ancestors(leaf)
            if a == root or ontology.is_subclass_of(a, root)
        ]
        for node in targets:
            node_chunks[node].update(chunk_ids)
    return dict(node_chunks)
