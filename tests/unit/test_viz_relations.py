"""`fs.viz.relation_neighborhood` — the Layer 0 view on `fs.viz`."""

from __future__ import annotations

from foodscholar import FoodScholar
from foodscholar.io.entity import Entity
from foodscholar.io.relation import Relation, make_relation_id
from foodscholar.viz.view import RenderableGraph


def rel(s: str, p: str, o: str, *, chunks: int = 2, object_linked: bool = True) -> Relation:
    return Relation(
        relation_id=make_relation_id(s, p, o),
        subject_id=s,
        predicate=p,
        object_id=o,
        chunk_ids=tuple(f"c{i}" for i in range(chunks)),
        chunk_count=chunks,
        object_linked=object_linked,
    )


def entity(oid: str) -> Entity:
    return Entity(ontology_id=oid, prefix=oid.split(":")[0], label=oid.lower())


def test_is_exposed_on_fs_viz_and_returns_a_renderable():
    fs = FoodScholar.in_memory()
    graph = fs.viz.relation_neighborhood("FOODON:1")
    assert isinstance(graph, RenderableGraph)


def test_empty_when_nothing_references_the_entity():
    fs = FoodScholar.in_memory()
    graph = fs.viz.relation_neighborhood("FOODON:1")
    assert len(graph.edges) == 0
    assert "no relations" in graph.title


def test_predicate_rides_on_edge_kind_and_chunk_count_on_weight():
    fs = FoodScholar.in_memory()
    fs.entity_store.upsert([entity("FOODON:1"), entity("CHEBI:2")])
    fs.relation_store.upsert([rel("FOODON:1", "reduces", "CHEBI:2", chunks=3)])

    graph = fs.viz.relation_neighborhood("FOODON:1")
    (edge,) = graph.edges
    assert edge.kind == "reduces"
    assert edge.weight == 3.0
    assert edge.source == "FOODON:1" and edge.target == "CHEBI:2"
    assert {n.id for n in graph.nodes} == {"FOODON:1", "CHEBI:2"}


def test_nil_endpoints_get_a_node_so_edges_do_not_dangle():
    fs = FoodScholar.in_memory()
    fs.entity_store.upsert([entity("FOODON:1")])
    fs.relation_store.upsert([rel("FOODON:1", "affects", "NIL:hba1c", object_linked=False)])

    graph = fs.viz.relation_neighborhood("FOODON:1")
    assert "NIL:hba1c" in {n.id for n in graph.nodes}
    assert len(graph.edges) == 1


def test_grounded_only_hides_nil_edges():
    fs = FoodScholar.in_memory()
    fs.entity_store.upsert([entity("FOODON:1"), entity("CHEBI:2")])
    fs.relation_store.upsert(
        [
            rel("FOODON:1", "reduces", "CHEBI:2"),
            rel("FOODON:1", "affects", "NIL:hba1c", object_linked=False),
        ]
    )
    assert len(fs.viz.relation_neighborhood("FOODON:1").edges) == 2
    assert len(fs.viz.relation_neighborhood("FOODON:1", grounded_only=True).edges) == 1


def test_two_hops_reach_the_neighbour_of_a_neighbour():
    fs = FoodScholar.in_memory()
    fs.entity_store.upsert([entity("A:1"), entity("B:2"), entity("C:3")])
    fs.relation_store.upsert([rel("A:1", "p", "B:2"), rel("B:2", "q", "C:3")])

    one = fs.viz.relation_neighborhood("A:1", hops=1)
    two = fs.viz.relation_neighborhood("A:1", hops=2)
    assert {n.id for n in one.nodes} == {"A:1", "B:2"}
    assert {n.id for n in two.nodes} == {"A:1", "B:2", "C:3"}


def test_an_edge_reachable_from_both_endpoints_is_not_duplicated():
    fs = FoodScholar.in_memory()
    fs.entity_store.upsert([entity("A:1"), entity("B:2")])
    fs.relation_store.upsert([rel("A:1", "p", "B:2")])
    graph = fs.viz.relation_neighborhood("A:1", hops=2)
    assert len(graph.edges) == 1
