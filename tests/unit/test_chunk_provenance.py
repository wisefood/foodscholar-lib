"""`ChunkProvenance` — the three source shapes and the casing trap."""

from __future__ import annotations

from foodscholar.io.chunk import Chunk, ChunkProvenance


def test_guide_shape_exposes_every_citation_field():
    p = ChunkProvenance.model_validate(
        {
            "file": "ie-key-messages.pdf",
            "urn": "urn:ie:1",
            "pdf_name": "ie-key-messages",
            "country": "IE",
            "title": "Key Messages",
            "audience": "adults",
            "pages": 12,
            "heading": "Vegetables",
            "page_number": 4,
        }
    )
    assert p.country == "IE"
    assert p.pdf_name == "ie-key-messages"
    assert set(p.citation_fields()) == {
        "file",
        "title",
        "country",
        "heading",
        "page_number",
    }


def test_textbook_shape_carries_only_three_fields():
    """Textbook chunks have no title and no country — renderers must degrade."""
    p = ChunkProvenance.model_validate(
        {"file": "Human-Nutrition.pdf", "heading": "Ch 1", "page_number": 7}
    )
    assert p.title is None
    assert p.country is None
    assert set(p.citation_fields()) == {"file", "heading", "page_number"}


def test_abstract_shape_is_bibliographic():
    p = ChunkProvenance.model_validate(
        {"title": "A trial", "year": 2019, "doi": "10.1/x", "paperId": "p1"}
    )
    assert p.paper_id == "p1"
    assert p.page_number is None


def test_uppercase_doi_is_folded_not_lost():
    assert ChunkProvenance.model_validate({"DOI": "10.1/x"}).doi == "10.1/x"


def test_lowercase_doi_wins_when_both_present():
    p = ChunkProvenance.model_validate({"DOI": "10.1/UPPER", "doi": "10.1/lower"})
    assert p.doi == "10.1/lower"


def test_numeric_strings_and_floats_coerce():
    p = ChunkProvenance.model_validate({"page_number": "4", "pages": 12.0, "year": "2019"})
    assert (p.page_number, p.pages, p.year) == (4, 12, 2019)


def test_unparseable_numerics_become_none_rather_than_raising():
    p = ChunkProvenance.model_validate({"page_number": "n/a", "year": ""})
    assert p.page_number is None
    assert p.year is None


def test_unknown_keys_are_preserved():
    p = ChunkProvenance.model_validate({"file": "a.pdf", "custom_field": "kept"})
    assert p.model_extra == {"custom_field": "kept"}


def test_chunk_provenance_property_is_cached():
    chunk = Chunk(
        chunk_id="c1",
        text="x",
        source_doc_id="d",
        source_type="guide",
        section_type="guideline",
        source_metadata={"file": "f.pdf", "page_number": "3"},
    )
    assert chunk.provenance.page_number == 3
    assert chunk.provenance is chunk.provenance
