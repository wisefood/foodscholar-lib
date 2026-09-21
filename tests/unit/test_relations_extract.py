"""Extraction on a stubbed `LLMClient` — no network, no models."""

from __future__ import annotations

from foodscholar.relations.extract import (
    MAX_ENUM_ENTITIES,
    extract_chunk,
    parse_entities,
    parse_relations,
    relations_schema,
)
from tests.unit.conftest_relations import RaisingLLM, StubLLM

ENTS = {"entities": ["olive oil", "LDL cholesterol"]}
RELS = {"relations": [{"subject": "olive oil", "predicate": "reduces", "object": "LDL cholesterol"}]}


def test_endpoints_are_constrained_to_the_step_one_entities():
    """The enum is what keeps triples aligned with the extracted entities."""
    schema = relations_schema(["olive oil", "LDL"])
    subject = schema["properties"]["relations"]["items"]["properties"]["subject"]
    assert subject["enum"] == ["olive oil", "LDL"]


def test_large_entity_sets_fall_back_to_an_unconstrained_schema():
    schema = relations_schema([f"e{i}" for i in range(MAX_ENUM_ENTITIES + 1)])
    subject = schema["properties"]["relations"]["items"]["properties"]["subject"]
    assert "enum" not in subject


def test_parse_entities_dedups_case_insensitively_keeping_first():
    assert parse_entities({"entities": ["Olive Oil", "olive oil", "LDL"]}) == [
        "Olive Oil",
        "LDL",
    ]


def test_parse_entities_drops_short_and_blank():
    assert parse_entities({"entities": ["a", "", "  ", "ok"]}) == ["ok"]


def test_parse_entities_tolerates_garbage():
    assert parse_entities({}) == []
    assert parse_entities({"entities": "not a list"}) == []


def test_parse_relations_filters_endpoints_not_in_the_entity_set():
    triples = parse_relations(
        {
            "relations": [
                {"subject": "olive oil", "predicate": "reduces", "object": "LDL"},
                {"subject": "INVENTED", "predicate": "x", "object": "LDL"},
            ]
        },
        ["olive oil", "LDL"],
    )
    assert triples == [("olive oil", "reduces", "LDL")]


def test_parse_relations_drops_self_loops():
    assert (
        parse_relations(
            {"relations": [{"subject": "LDL", "predicate": "is", "object": "LDL"}]},
            ["LDL"],
        )
        == []
    )


def test_parse_relations_accepts_a_bare_list():
    assert parse_relations(
        [{"subject": "a b", "predicate": "p", "object": "c d"}], ["a b", "c d"]
    ) == [("a b", "p", "c d")]


def test_parse_relations_canonicalizes_endpoint_casing():
    assert parse_relations(
        {"relations": [{"subject": "OLIVE OIL", "predicate": "p", "object": "ldl"}]},
        ["olive oil", "LDL"],
    ) == [("olive oil", "p", "LDL")]


def test_extract_chunk_records_provenance():
    llm = StubLLM([ENTS, RELS])
    graph = extract_chunk("some text", "c1", llm=llm)
    triple = ("olive oil", "reduces", "LDL cholesterol")
    assert graph.triple_chunks[triple] == {"c1"}
    assert graph.entity_chunks["olive oil"] == {"c1"}


def test_extract_chunk_is_empty_for_blank_text():
    llm = StubLLM([])
    assert extract_chunk("   ", "c1", llm=llm).triples == set()


def test_entity_call_failure_returns_an_empty_graph_not_an_exception():
    """One bad chunk must not abort a corpus-wide run."""
    llm = RaisingLLM(RuntimeError("boom"), fail_on=0)
    assert extract_chunk("text", "c1", llm=llm).triples == set()


def test_relation_call_retries_unconstrained_then_gives_up():
    llm = RaisingLLM(RuntimeError("schema rejected"), fail_on=1)
    graph = extract_chunk("text", "c1", llm=llm)
    # entities succeeded, relations failed twice (enum + fallback)
    assert graph.entities == {"alpha", "beta"}
    assert graph.triples == set()
    assert llm.calls == 3


def test_context_length_error_skips_without_retrying():
    llm = RaisingLLM(RuntimeError("context length exceeded"), fail_on=1)
    extract_chunk("text", "c1", llm=llm)
    assert llm.calls == 2  # no unconstrained retry
