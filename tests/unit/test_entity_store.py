"""Tests for the in-memory `EntityStore` adapter."""

from __future__ import annotations

from foodscholar.io.entity import Entity
from foodscholar.storage.memory import InMemoryEntityStore
from foodscholar.storage.protocols import EntityStore


def _e(ontology_id: str, *, prefix: str | None = None, **kwargs) -> Entity:  # type: ignore[no-untyped-def]
    return Entity(
        ontology_id=ontology_id,
        prefix=prefix or ontology_id.split(":", 1)[0],
        label=kwargs.pop("label", ontology_id),
        **kwargs,
    )


def test_in_memory_entity_store_satisfies_protocol() -> None:
    assert isinstance(InMemoryEntityStore(), EntityStore)


def test_upsert_and_get_round_trip() -> None:
    store = InMemoryEntityStore()
    e = _e("FOODON:03309927", label="olive oil", synonyms=("EVOO",))
    store.upsert([e])
    fetched = store.get("FOODON:03309927")
    assert fetched is not None
    assert fetched.label == "olive oil"
    assert fetched.synonyms == ("EVOO",)


def test_upsert_replaces_existing() -> None:
    store = InMemoryEntityStore()
    store.upsert([_e("FOODON:1", label="old")])
    store.upsert([_e("FOODON:1", label="new")])
    assert store.get("FOODON:1").label == "new"  # type: ignore[union-attr]


def test_get_many_filters_unknown_ids() -> None:
    store = InMemoryEntityStore()
    store.upsert([_e("FOODON:1"), _e("FOODON:2")])
    out = store.get_many(["FOODON:1", "FOODON:does-not-exist", "FOODON:2"])
    assert sorted(e.ontology_id for e in out) == ["FOODON:1", "FOODON:2"]


def test_list_by_prefix_filters_and_orders_by_chunk_count() -> None:
    store = InMemoryEntityStore()
    store.upsert(
        [
            _e("FOODON:big", chunk_count=100),
            _e("FOODON:small", chunk_count=2),
            _e("CHEBI:x", chunk_count=999),
        ]
    )
    out = store.list_by_prefix("FOODON")
    assert [e.ontology_id for e in out] == ["FOODON:big", "FOODON:small"]


def test_search_label_and_synonyms() -> None:
    store = InMemoryEntityStore()
    store.upsert(
        [
            _e("FOODON:1", label="olive oil", synonyms=("EVOO",)),
            _e("FOODON:2", label="vitamin C", synonyms=("ascorbic acid",)),
            _e("CHEBI:3", label="iron"),
        ]
    )
    # label match
    [hit] = store.search("olive", k=1)
    assert hit.ontology_id == "FOODON:1"
    # synonym match
    [hit2] = store.search("ascorbic", k=1)
    assert hit2.ontology_id == "FOODON:2"
    # prefix filter excludes non-matching prefix
    assert store.search("iron", prefix="FOODON") == []
    assert store.search("iron", prefix="CHEBI")[0].ontology_id == "CHEBI:3"


def test_search_empty_query_returns_empty() -> None:
    store = InMemoryEntityStore()
    store.upsert([_e("FOODON:1", label="x")])
    assert store.search("") == []
