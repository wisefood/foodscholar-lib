"""Write relations to the stores.

Two destinations, mirroring how entities are persisted:

- `RelationStore` — the queryable record (`for_chunks` is retrieval's hot path)
- `GraphStore` — ``(:Entity)-[:RELATED {predicate}]->(:Entity)`` for traversal

NIL endpoints become `(:Entity {prefix: "NIL"})` nodes so the graph stays
traversable; Cypher filters them with
``WHERE NOT e.ontology_id STARTS WITH 'NIL:'``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.io.relation import Relation
    from foodscholar.storage.protocols import GraphStore, RelationStore

_log = get_logger("foodscholar.relations.persist")


def persist(
    relations: list[Relation],
    *,
    relation_store: RelationStore,
    graph_store: GraphStore | None = None,
    clear_first: bool = False,
) -> dict[str, int]:
    """Upsert to the relation store and, when supported, the graph store."""
    if clear_first:
        relation_store.clear()
        clear_relations = getattr(graph_store, "clear_relations", None)
        if callable(clear_relations):
            clear_relations()

    relation_store.upsert(relations)

    n_graph = 0
    upsert_relations = getattr(graph_store, "upsert_relations", None)
    if callable(upsert_relations) and relations:
        upsert_relations(relations)
        n_graph = len(relations)

    counts = {"n_relations": len(relations), "n_graph_edges": n_graph}
    _log.info("relations.persist.done", **counts)
    return counts
