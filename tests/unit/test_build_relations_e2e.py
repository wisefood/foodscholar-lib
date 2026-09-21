"""`fs.build_relations()` end to end on the in-memory facade.

Stub LLM + stub linker, so the whole Layer 0 pipeline — extract, dedupe,
ground, aggregate, persist — is exercised in the unit gate with no network,
no models and no services.
"""

from __future__ import annotations

import pytest

from foodscholar import FoodScholar
from foodscholar.io.chunk import Chunk
from tests.unit.conftest_relations import StubLinker, StubLLM

ENTITIES = {"entities": ["olive oil", "LDL cholesterol"]}
RELATIONS = {
    "relations": [
        {"subject": "olive oil", "predicate": "reduces", "object": "LDL cholesterol"}
    ]
}
LINK_TABLE = {
    "olive oil": ("FOODON:03301710", 0.95),
    "ldl cholesterol": ("CHEBI:39026", 0.85),
}


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id="doc",
        source_type="guide",
        section_type="guideline",
    )


@pytest.fixture
def fs() -> FoodScholar:
    instance = FoodScholar.in_memory()
    instance.upsert_chunks([chunk("c1", "Olive oil reduces LDL cholesterol.")])
    instance.attach_linker(StubLinker(LINK_TABLE))
    return instance


def test_refuses_to_persist_mock_llm_output(fs):
    with pytest.raises(RuntimeError, match="needs a real LLM"):
        fs.build_relations()


def test_dry_run_is_allowed_with_the_mock_and_writes_nothing(fs):
    fs.llm = StubLLM([ENTITIES, RELATIONS])
    meta = fs.build_relations(dry_run=True)
    assert meta.record_count == 1
    assert len(fs.relations) == 0


def test_persists_grounded_relations(fs):
    fs.llm = StubLLM([ENTITIES, RELATIONS])
    meta = fs.build_relations()

    assert meta.phase == "build_relations"
    assert len(fs.relations) == 1
    relation = fs.relations.grounded()[0]
    assert relation.triple == ("FOODON:03301710", "reduces", "CHEBI:39026")
    assert relation.chunk_ids == ("c1",)
    assert relation.is_fully_grounded


def test_endpoints_join_the_entity_universe(fs):
    """The whole point of grounding: relation endpoints are ontology ids that
    `fs.entities` can resolve, not free-text strings."""
    fs.llm = StubLLM([ENTITIES, RELATIONS])
    fs.build_relations()
    relation = fs.relations.grounded()[0]
    assert relation.subject_id.startswith("FOODON:")
    assert not relation.subject_id.startswith("NIL:")


def test_provenance_resolves_in_the_chunk_store(fs):
    """The invariant the integration exists to preserve."""
    fs.llm = StubLLM([ENTITIES, RELATIONS])
    fs.build_relations()
    for relation in fs.relations:
        for chunk_id in relation.chunk_ids:
            assert fs.chunk_store.get(chunk_id) is not None


def test_for_chunks_round_trips(fs):
    fs.llm = StubLLM([ENTITIES, RELATIONS])
    fs.build_relations()
    assert len(fs.relations.for_chunks(["c1"])) == 1
    assert fs.relations.for_chunks(["nope"]) == []


def test_rerun_skips_covered_chunks(fs):
    """The store is the resume log — no savepoint files."""
    fs.llm = StubLLM([ENTITIES, RELATIONS])
    fs.build_relations()
    fs.llm = StubLLM([])  # any LLM call would raise
    meta = fs.build_relations()
    assert meta.record_count == 0
    assert len(fs.relations) == 1


def test_force_reextracts_and_is_idempotent(fs):
    fs.llm = StubLLM([ENTITIES, RELATIONS])
    fs.build_relations()
    first = fs.relations.grounded()[0]

    fs.llm = StubLLM([ENTITIES, RELATIONS])
    fs.build_relations(force=True)
    second = fs.relations.grounded()[0]

    assert len(fs.relations) == 1
    assert first.relation_id == second.relation_id  # content-addressed


def test_nil_endpoints_are_kept_not_dropped(fs):
    fs.upsert_chunks([chunk("c2", "Olive oil affects HbA1c.")])
    fs.llm = StubLLM(
        [
            {"entities": ["olive oil", "HbA1c"]},
            {"relations": [{"subject": "olive oil", "predicate": "affects", "object": "HbA1c"}]},
        ]
    )
    fs.build_relations(chunk_ids=["c2"])
    relation = next(iter(fs.relations))
    assert relation.object_id == "NIL:hba1c"
    assert not relation.object_linked
    assert not relation.is_fully_grounded


def test_summary_and_predicates(fs):
    fs.llm = StubLLM([ENTITIES, RELATIONS])
    fs.build_relations()
    assert fs.relations.summary() == {
        "relations": 1,
        "fully_grounded": 1,
        "predicates": 1,
        "entities": 2,
    }
    assert fs.relations.predicates() == [("reduces", 1)]
