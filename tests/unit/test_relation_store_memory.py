"""`InMemoryRelationStore` against the full `RelationStore` protocol."""

from __future__ import annotations

from foodscholar.io.relation import Relation, make_relation_id
from foodscholar.storage.memory import InMemoryRelationStore
from foodscholar.storage.protocols import RelationStore


def rel(s: str, p: str, o: str, chunks: list[str], mentions: int = 1) -> Relation:
    return Relation(
        relation_id=make_relation_id(s, p, o),
        subject_id=s,
        predicate=p,
        object_id=o,
        chunk_ids=tuple(chunks),
        chunk_count=len(chunks),
        mention_count=mentions,
    )


def populated() -> InMemoryRelationStore:
    store = InMemoryRelationStore()
    store.upsert(
        [
            rel("A", "reduces", "B", ["c1", "c2"]),
            rel("B", "causes", "C", ["c2"]),
            rel("A", "contains", "C", ["c3"], mentions=5),
        ]
    )
    return store


def test_satisfies_the_protocol():
    assert isinstance(InMemoryRelationStore(), RelationStore)


def test_init_is_a_noop():
    InMemoryRelationStore().init()


def test_get_and_get_many():
    store = populated()
    rid = make_relation_id("A", "reduces", "B")
    assert store.get(rid).triple == ("A", "reduces", "B")
    assert store.get("rel:missing") is None
    assert len(store.get_many([rid, "rel:missing"])) == 1


def test_for_chunks_uses_the_inverted_index():
    store = populated()
    assert {r.triple for r in store.for_chunks(["c2"])} == {
        ("A", "reduces", "B"),
        ("B", "causes", "C"),
    }
    assert store.for_chunks([]) == []
    assert store.for_chunks(["nope"]) == []


def test_for_chunks_deduplicates_across_ids():
    store = populated()
    hits = store.for_chunks(["c1", "c2"])
    assert len({r.relation_id for r in hits}) == len(hits)


def test_reupsert_with_fewer_chunks_drops_stale_index_entries():
    """Without this the relation stays reachable from a chunk it no longer cites."""
    store = populated()
    store.upsert([rel("A", "reduces", "B", ["c1"])])
    assert {r.triple for r in store.for_chunks(["c2"])} == {("B", "causes", "C")}


def test_for_entity_directions():
    store = populated()
    assert {r.predicate for r in store.for_entity("A", direction="out")} == {
        "reduces",
        "contains",
    }
    assert {r.predicate for r in store.for_entity("C", direction="in")} == {
        "contains",
        "causes",
    }
    assert len(store.for_entity("A", direction="both")) == 2
    assert store.for_entity("A", direction="in") == []


def test_for_entity_respects_k():
    assert len(populated().for_entity("A", k=1)) == 1


def test_by_predicate():
    store = populated()
    assert [r.triple for r in store.by_predicate("reduces")] == [("A", "reduces", "B")]
    assert store.by_predicate("nonexistent") == []


def test_iter_relations_batches():
    assert [len(b) for b in populated().iter_relations(batch_size=2)] == [2, 1]


def test_clear_empties_both_index_and_store():
    store = populated()
    store.clear()
    assert store.scan() == []
    assert store.for_chunks(["c1"]) == []
    store.clear()  # idempotent
