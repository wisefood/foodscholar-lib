"""Audit section F — Layer 0 invariants."""

from __future__ import annotations

from foodscholar.evaluation.audit import audit
from foodscholar.io.chunk import Chunk
from foodscholar.io.relation import Relation, make_relation_id
from foodscholar.storage.memory import (
    InMemoryChunkStore,
    InMemoryGraphStore,
    InMemoryRelationStore,
)


def chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="text",
        source_doc_id="d",
        source_type="guide",
        section_type="guideline",
    )


def rel(s: str, p: str, o: str, chunks: list[str], *, object_linked: bool = True):
    return Relation(
        relation_id=make_relation_id(s, p, o),
        subject_id=s,
        predicate=p,
        object_id=o,
        chunk_ids=tuple(chunks),
        chunk_count=len(chunks),
        object_linked=object_linked,
    )


def run(relations, chunk_ids=("c1", "c2")):
    chunk_store = InMemoryChunkStore()
    chunk_store.upsert([chunk(c) for c in chunk_ids])
    relation_store = InMemoryRelationStore()
    relation_store.upsert(relations)
    return audit(
        chunk_store,
        InMemoryGraphStore(),
        config_hash="test",
        relation_store=relation_store,
    )


def find(report, name_fragment: str):
    return next(c for c in report.checks if name_fragment in c.name)


def test_section_is_absent_when_layer_0_is_unused():
    report = audit(
        InMemoryChunkStore(), InMemoryGraphStore(), config_hash="test",
        relation_store=InMemoryRelationStore(),
    )
    assert not any(c.section == "F. Relations" for c in report.checks)


def test_section_is_absent_when_no_store_is_supplied():
    report = audit(InMemoryChunkStore(), InMemoryGraphStore(), config_hash="test")
    assert not any(c.section == "F. Relations" for c in report.checks)


def test_resolvable_provenance_passes():
    report = run([rel("A", "p", "B", ["c1", "c2"])])
    assert find(report, "chunk_ids resolve").passed
    assert report.passed


def test_dangling_provenance_is_a_critical_failure():
    """The signature of a corpus re-chunked under uuid4 ids."""
    report = run([rel("A", "p", "B", ["c1", "ghost"])])
    check = find(report, "chunk_ids resolve")
    assert not check.passed
    assert check.severity == "critical"
    assert not report.passed
    assert "ghost" in check.sample[0]["missing_chunk_ids"]
    assert "content_hash" in check.details["hint"]


def test_inventory_counts_relations():
    report = run([rel("A", "p", "B", ["c1"]), rel("B", "q", "C", ["c2"])])
    assert report.inventory["relations_total"] == 2
    assert report.inventory["relation_predicates"] == 2
    assert report.inventory["relation_entities"] == 3


def test_grounded_share_is_reported():
    report = run(
        [
            rel("A", "p", "B", ["c1"]),
            rel("A", "q", "NIL:x", ["c2"], object_linked=False),
        ]
    )
    assert find(report, "both endpoints grounded").metric == 0.5


def test_predicate_sprawl_warns():
    relations = [rel("A", f"p{i}", f"B{i}", ["c1"]) for i in range(10)]
    check = find(run(relations), "predicate cardinality")
    assert not check.passed
    assert check.severity == "warning"


def test_predicate_reuse_passes():
    relations = [rel(f"A{i}", "reduces", f"B{i}", ["c1"]) for i in range(10)]
    assert find(run(relations), "predicate cardinality").passed


def test_nil_endpoints_are_reported_not_judged():
    report = run(
        [
            rel("A", "p", "NIL:hba1c", ["c1"], object_linked=False),
            rel("B", "q", "NIL:hba1c", ["c2"], object_linked=False),
        ]
    )
    check = find(report, "NIL endpoints")
    assert check.passed  # informational
    assert check.sample[0] == {"nil_id": "NIL:hba1c", "count": 2}
