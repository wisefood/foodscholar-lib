"""Read-only MCP-style graph tools the agent queries while building the tree."""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.layer_a.bakeoff.agentic.relations import Relation

if TYPE_CHECKING:
    from foodscholar.ontology import FoodOnAPI


class GraphTools:
    """Read-only queries over FoodOn + the relation index + node support."""

    def __init__(
        self,
        ontology: FoodOnAPI,
        relation_index: dict[str, list[Relation]],
        *,
        node_support: dict[str, int],
        min_support: int,
        retriever=None,
    ) -> None:
        self._o = ontology
        self._rel = relation_index
        self._support = node_support
        self._min = min_support
        self._retriever = retriever

    def support(self, fid: str) -> int:
        return self._support.get(fid, 0)

    def label(self, fid: str) -> str:
        return self._o.id_to_label(fid) or fid

    def supported_children(self, fid: str) -> list[str]:
        """Direct is-a children whose rolled-up support clears the floor,
        most-supported first."""
        kids = [
            c for c in self._o.id_to_children(fid)
            if c in self._o and self.support(c) >= self._min
        ]
        return sorted(kids, key=lambda c: -self.support(c))

    def relation_targets(self, fid: str) -> list[tuple[str, str, str]]:
        """Non-is-a FoodOn relation targets (rel_id, rel_name, target_id)."""
        return [(r.rel_id, r.rel_name, r.target_id) for r in self._rel.get(fid, [])]

    def search(self, query: str, *, k: int = 8) -> list[str]:
        """Candidate FoodOn ids for a concept (retriever if given, else label search)."""
        if self._retriever is not None:
            return [c.id for c in self._retriever.retrieve(query, k=k)]
        return self._o.search(query, limit=k)

    def lowest_common_ancestor(self, ids: list[str]) -> str | None:
        """Deepest FoodOn node that is an ancestor-or-self of every id in `ids`."""
        if not ids:
            return None
        sets = [{fid, *self._o.id_to_ancestors(fid)} for fid in ids]
        common = set.intersection(*sets) if sets else set()
        if not common:
            return None
        return max(common, key=lambda a: len(self._o.id_to_ancestors(a)))
