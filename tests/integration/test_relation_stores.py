"""Integration tests for the Layer 0 remote adapters.

Requires a live Elasticsearch at localhost:9200 and/or Neo4j at
bolt://localhost:7687 (`docker compose up -d elasticsearch neo4j`). Each
fixture skips when its service is unreachable, so the module is safe to run
without either.
"""

from __future__ import annotations

import contextlib
import os
import uuid

import pytest

from foodscholar.io.relation import Relation, make_relation_id

pytestmark = pytest.mark.integration

_ES_URL = "http://localhost:9200"
_NEO4J_URL = "bolt://localhost:7687"


def relation(
    subject: str,
    predicate: str,
    obj: str,
    chunks: list[str],
    *,
    mentions: int = 1,
    object_linked: bool = True,
) -> Relation:
    return Relation(
        relation_id=make_relation_id(subject, predicate, obj),
        subject_id=subject,
        predicate=predicate,
        object_id=obj,
        subject_surfaces=("olive oil",),
        object_surfaces=("LDL cholesterol",),
        predicate_surfaces=(predicate,),
        chunk_ids=tuple(chunks),
        chunk_count=len(chunks),
        mention_count=mentions,
        object_linked=object_linked,
        extractor_version="integration-test",
        dedupe_version="integration-test",
    )


SAMPLE = [
    relation("FOODON:1", "reduces", "CHEBI:2", ["c1", "c2"], mentions=3),
    relation("CHEBI:2", "causes", "MONDO:3", ["c2"]),
    relation("FOODON:1", "contains", "NIL:mystery", ["c3"], object_linked=False),
]


# ---------------------------------------------------------------- Elasticsearch


@pytest.fixture
def es_store():
    pytest.importorskip("elasticsearch")
    from elasticsearch import Elasticsearch

    from foodscholar.storage.elastic_relations import ElasticRelationStore

    client = Elasticsearch(_ES_URL, request_timeout=10)
    try:
        if not client.ping():
            pytest.skip("Elasticsearch unreachable at localhost:9200")
        # A red cluster cannot allocate a new shard — most often the host disk
        # is past the 90% high watermark. Writes then time out rather than
        # failing fast, which looks like an adapter bug but is not one.
        health = client.cluster.health()
        if health.get("status") == "red":
            pytest.skip(
                f"Elasticsearch cluster is red "
                f"({health.get('unassigned_shards')} unassigned shards) — "
                f"usually disk pressure; cannot allocate a test index"
            )
    except Exception as e:
        pytest.skip(f"Elasticsearch unreachable: {e}")

    index = f"fs_test_relations_{uuid.uuid4().hex[:8]}"
    store = ElasticRelationStore(url=_ES_URL, index=index)
    store.init()
    try:
        yield store
    finally:
        # Best-effort teardown: a dead cluster must not mask the test result.
        with contextlib.suppress(Exception):
            client.indices.delete(index=index, ignore_unavailable=True)


def test_es_round_trip_preserves_every_field(es_store):
    es_store.upsert(SAMPLE)
    stored = es_store.get(SAMPLE[0].relation_id)
    assert stored is not None
    assert stored.model_dump() == SAMPLE[0].model_dump()


def test_es_for_chunks_is_a_terms_filter(es_store):
    es_store.upsert(SAMPLE)
    hits = es_store.for_chunks(["c2"])
    assert {r.triple for r in hits} == {
        ("FOODON:1", "reduces", "CHEBI:2"),
        ("CHEBI:2", "causes", "MONDO:3"),
    }


def test_es_for_chunks_deduplicates_across_ids(es_store):
    es_store.upsert(SAMPLE)
    hits = es_store.for_chunks(["c1", "c2"])
    assert len({r.relation_id for r in hits}) == len(hits)


def test_es_for_entity_directions(es_store):
    es_store.upsert(SAMPLE)
    assert {r.predicate for r in es_store.for_entity("FOODON:1", direction="out")} == {
        "reduces",
        "contains",
    }
    assert {r.predicate for r in es_store.for_entity("CHEBI:2", direction="in")} == {
        "reduces"
    }
    assert len(es_store.for_entity("CHEBI:2", direction="both")) == 2


def test_es_by_predicate_and_scan(es_store):
    es_store.upsert(SAMPLE)
    assert [r.triple for r in es_store.by_predicate("causes")] == [
        ("CHEBI:2", "causes", "MONDO:3")
    ]
    assert len(es_store.scan()) == 3


def test_es_upsert_is_idempotent(es_store):
    es_store.upsert(SAMPLE)
    es_store.upsert(SAMPLE)
    assert len(es_store.scan()) == 3


def test_es_nil_endpoint_survives_the_round_trip(es_store):
    es_store.upsert(SAMPLE)
    nil = es_store.get(make_relation_id("FOODON:1", "contains", "NIL:mystery"))
    assert nil.object_id == "NIL:mystery"
    assert nil.object_linked is False
    assert not nil.is_fully_grounded


def test_es_clear_keeps_the_index(es_store):
    es_store.upsert(SAMPLE)
    es_store.clear()
    assert es_store.scan() == []
    es_store.upsert(SAMPLE[:1])  # mapping still present
    assert len(es_store.scan()) == 1


# ----------------------------------------------------------------------- Neo4j


@pytest.fixture
def neo4j_store():
    pytest.importorskip("neo4j")
    from foodscholar.storage.neo4j import Neo4jGraphStore

    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        pytest.skip("NEO4J_PASSWORD not set")
    try:
        store = Neo4jGraphStore(url=_NEO4J_URL, user="neo4j", password=password)
        store.init()
    except Exception as e:
        pytest.skip(f"Neo4j unreachable: {e}")
    try:
        store.clear_relations()
        yield store
    finally:
        with contextlib.suppress(Exception):
            store.clear_relations()


def _related_count(store) -> int:
    with store._driver.session() as session:
        return session.run(
            "MATCH ()-[r:RELATED]->() RETURN count(r) AS n"
        ).single()["n"]


def test_neo4j_creates_related_edges(neo4j_store):
    neo4j_store.upsert_relations(SAMPLE)
    assert _related_count(neo4j_store) == 3


def test_neo4j_upsert_is_idempotent(neo4j_store):
    """MERGE on (subject, predicate, object) — re-running must not duplicate."""
    neo4j_store.upsert_relations(SAMPLE)
    neo4j_store.upsert_relations(SAMPLE)
    assert _related_count(neo4j_store) == 3


def test_neo4j_materializes_nil_endpoints_as_entities(neo4j_store):
    """NIL nodes keep the graph traversable; Cypher filters them by prefix."""
    neo4j_store.upsert_relations(SAMPLE)
    with neo4j_store._driver.session() as session:
        nil = session.run(
            "MATCH (e:Entity) WHERE e.ontology_id STARTS WITH 'NIL:' "
            "RETURN e.prefix AS prefix"
        ).single()
    assert nil["prefix"] == "NIL"


def test_neo4j_edge_carries_provenance(neo4j_store):
    neo4j_store.upsert_relations(SAMPLE)
    with neo4j_store._driver.session() as session:
        row = session.run(
            "MATCH (:Entity {ontology_id: 'FOODON:1'})-[r:RELATED {predicate: 'reduces'}]->() "
            "RETURN r.chunk_ids AS chunk_ids, r.chunk_count AS n, r.mention_count AS m"
        ).single()
    assert sorted(row["chunk_ids"]) == ["c1", "c2"]
    assert row["n"] == 2
    assert row["m"] == 3


def test_neo4j_clear_relations_removes_edges_and_orphan_nil_nodes(neo4j_store):
    neo4j_store.upsert_relations(SAMPLE)
    neo4j_store.clear_relations()
    assert _related_count(neo4j_store) == 0
    with neo4j_store._driver.session() as session:
        n_nil = session.run(
            "MATCH (e:Entity) WHERE e.ontology_id STARTS WITH 'NIL:' RETURN count(e) AS n"
        ).single()["n"]
    assert n_nil == 0
