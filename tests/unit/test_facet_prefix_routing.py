"""route_link_to_facet: prefix→facet fallback so non-FOODON OBO entities
(CHEBI, UBERON, ENVO, …) populate nutrients/health/sustainability facets when
the NER left entity_type='other'."""

from __future__ import annotations

from foodscholar.io.chunk import EntityLink, Mention
from foodscholar.layer_a.facet import (
    PREFIX_TO_FACET,
    facet_for_prefix,
    route_link_to_facet,
)


def _link(ontology_id: str, entity_type: str = "other") -> EntityLink:
    return EntityLink(
        mention=Mention(text="x", start=0, end=1, score=0.9,
                        ner_model_version="test", entity_type=entity_type),
        ontology_id=ontology_id, confidence=0.9, method="dense", linker_version="test",
    )


def test_entity_type_still_wins() -> None:
    # explicit entity_type takes precedence over the prefix fallback
    assert route_link_to_facet(_link("CHEBI:16646", entity_type="food")) == "foods"


def test_foodon_still_routes_to_foods() -> None:
    assert route_link_to_facet(_link("FOODON:00001234")) == "foods"


def test_chebi_routes_to_nutrients() -> None:
    assert route_link_to_facet(_link("CHEBI:16646")) == "nutrients"
    assert route_link_to_facet(_link("CDNO:0000005")) == "nutrients"


def test_uberon_routes_to_health() -> None:
    assert route_link_to_facet(_link("UBERON:0000948")) == "health"
    assert route_link_to_facet(_link("MONDO:0005010")) == "health"


def test_envo_routes_to_sustainability() -> None:
    assert route_link_to_facet(_link("ENVO:00003862")) == "sustainability"


def test_unknown_prefix_returns_none() -> None:
    assert route_link_to_facet(_link("ZZZ:123")) is None
    assert facet_for_prefix("ZZZ") is None


def test_prefix_map_is_exposed() -> None:
    assert PREFIX_TO_FACET["CHEBI"] == "nutrients"
    assert PREFIX_TO_FACET["UBERON"] == "health"
