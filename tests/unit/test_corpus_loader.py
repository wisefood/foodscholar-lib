from __future__ import annotations

from pathlib import Path

import pytest

from foodscholar.corpus import iter_chunks, load_chunks, write_chunks_parquet

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_load_legacy_csv_preserves_source_metadata() -> None:
    chunks = load_chunks(FIXTURES / "corpus_chunks.csv")
    assert len(chunks) == 3

    abstract = chunks[0]
    assert abstract.chunk_id == "c-abstract"
    assert abstract.text.startswith("Olive oil intake")
    assert abstract.source_type == "abstract"
    assert abstract.section_type == "abstract"
    assert abstract.source_doc_id == "https://doi.org/10.123/olive"
    assert abstract.year == 2024
    assert abstract.source_metadata["title"] == "Olive oil and heart health"
    assert abstract.source_metadata["citationCount"] == 3


def test_legacy_csv_derives_guide_and_textbook_fields() -> None:
    _, guide, textbook = load_chunks(FIXTURES / "corpus_chunks.csv")

    assert guide.source_type == "guide"
    assert guide.section_type == "guideline"
    assert guide.source_doc_id == "OK_Ireland_guide.pdf"
    assert guide.year == 2011
    assert guide.source_metadata["heading"] == "Vegetables"
    assert guide.source_metadata["page_number"] == 7

    assert textbook.source_type == "textbook"
    assert textbook.section_type == "textbook"
    assert textbook.source_doc_id == "Human-Nutrition.pdf"
    assert textbook.year is None


def test_iter_chunks_accepts_directory_in_sorted_order() -> None:
    chunks = list(iter_chunks(FIXTURES / "corpus_dir"))
    assert [c.chunk_id for c in chunks] == ["c-a", "c-b"]


def test_malformed_metadata_raises_in_strict_mode() -> None:
    with pytest.raises(SyntaxError):
        load_chunks(FIXTURES / "corpus_bad_metadata.csv")


def test_malformed_metadata_can_be_skipped_in_non_strict_mode() -> None:
    assert load_chunks(FIXTURES / "corpus_bad_metadata.csv", strict=False) == []


def test_parquet_round_trip_preserves_source_metadata(tmp_path: Path) -> None:
    chunks = load_chunks(FIXTURES / "corpus_chunks.csv")
    out = tmp_path / "chunks.parquet"
    assert write_chunks_parquet(chunks, out) == 3

    restored = load_chunks(out)
    assert restored == chunks
    assert restored[0].source_metadata["DOI"] == "https://doi.org/10.123/olive"
