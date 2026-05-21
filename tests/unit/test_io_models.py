from foodscholar.io.chunk import Chunk, EntityLink, Mention
from foodscholar.io.graph import Card, Shelf, Theme


def test_chunk_roundtrip() -> None:
    c = Chunk(
        chunk_id="c-1",
        text="Olive oil and heart health.",
        source_doc_id="d-1",
        source_type="abstract",
        section_type="abstract",
    )
    dumped = c.model_dump_json()
    restored = Chunk.model_validate_json(dumped)
    assert restored == c
    assert restored.shelf_ids == []
    assert restored.theme_ids == []
    assert restored.source_metadata == {}


def test_chunk_preserves_source_metadata() -> None:
    c = Chunk(
        chunk_id="c-1",
        text="Olive oil and heart health.",
        source_doc_id="10.123/example",
        source_type="abstract",
        section_type="abstract",
        year=2024,
        source_metadata={
            "title": "Olive oil study",
            "DOI": "https://doi.org/10.123/example",
            "citationCount": 12,
        },
    )
    restored = Chunk.model_validate_json(c.model_dump_json())
    assert restored.source_metadata["title"] == "Olive oil study"
    assert restored.source_metadata["citationCount"] == 12


def test_entity_link_validates_method() -> None:
    m = Mention(text="olive oil", start=0, end=9, score=0.9, ner_model_version="v0")
    link = EntityLink(
        mention=m,
        ontology_id="FOODON:03309927",
        confidence=0.92,
        method="lexical_exact",
        linker_version="v0",
    )
    assert link.method == "lexical_exact"


def test_mention_entity_type_defaults_to_other() -> None:
    # NER impls that don't classify (or pass an unknown label) leave it as "other".
    m = Mention(text="olive oil", start=0, end=9, score=1.0, ner_model_version="v0")
    assert m.entity_type == "other"


def test_mention_entity_type_accepts_valid_value() -> None:
    for et in (
        "food",
        "nutrient",
        "micronutrient",
        "macronutrient",
        "food component",
        "dietary supplement",
        "dietary pattern",
        "medical condition",
        "biomarker",
        "Country",
        "Measurement",
        "Population",
        "Time expression",
        "other",
    ):
        m = Mention(
            text="x",
            start=0,
            end=1,
            score=1.0,
            ner_model_version="v0",
            entity_type=et,  # type: ignore[arg-type]
        )
        assert m.entity_type == et


def test_mention_entity_type_rejects_unknown() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Mention(
            text="x",
            start=0,
            end=1,
            score=1.0,
            ner_model_version="v0",
            entity_type="condiment",  # not a valid EntityType
        )


def test_shelf_theme_card_basic() -> None:
    s = Shelf(shelf_id="s-1", label="Mediterranean Diet", facet="dietary_patterns", depth=1)
    t = Theme(
        theme_id="t-1",
        label="Olive oil cardiovascular benefits",
        shelf_ids=[s.shelf_id],
        discovered_by="leiden",
        discovery_version="v0",
    )
    card = Card(
        card_id="card-1",
        target_id=s.shelf_id,
        target_type="shelf",
        title="Mediterranean Diet",
        summary="A dietary pattern with strong cardiovascular evidence.",
        evidence_quality="high",
        cited_chunk_ids=["c-1"],
        llm_model="mock-llm-v0",
        prompt_version="v1",
    )
    assert card.target_type == "shelf"
    assert t.shelf_ids == ["s-1"]
