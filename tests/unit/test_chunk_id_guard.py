"""Chunk-id strategy and the destructive-re-chunk guard."""

from __future__ import annotations

import pytest

from foodscholar.config import ChunkerConfig
from foodscholar.corpus.chunker import (
    CorpusOverwriteError,
    _pages_for,
    chunk_documents,
    chunks_to_rows,
    load_excluded_pages,
    make_chunk_id,
)
from foodscholar.corpus.window import MergedChunk


def test_content_hash_is_stable_for_the_same_text():
    a = make_chunk_id("content_hash", source_doc_id="d", text="hello  world")
    b = make_chunk_id("content_hash", source_doc_id="d", text="hello world")
    assert a == b


def test_content_hash_differs_across_documents():
    assert make_chunk_id("content_hash", source_doc_id="a", text="t") != make_chunk_id(
        "content_hash", source_doc_id="b", text="t"
    )


def test_uuid4_is_fresh_every_call():
    """Why re-chunking orphans downstream provenance."""
    assert make_chunk_id("uuid4", source_doc_id="d", text="t") != make_chunk_id(
        "uuid4", source_doc_id="d", text="t"
    )


def test_guard_blocks_rechunking_an_existing_corpus(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "chunks_guide_a.csv").write_text("chunk_id\n")
    src = tmp_path / "pdfs"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"%PDF")
    with pytest.raises(CorpusOverwriteError, match="orphan every relation"):
        chunk_documents(src, out_dir=out, source_type="guide", cfg=ChunkerConfig())


def test_guard_allows_content_hash_ids(tmp_path):
    """Idempotent ids survive a re-chunk, so there is nothing to guard."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "chunks_guide_a.csv").write_text("chunk_id\n")
    src = tmp_path / "pdfs"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"%PDF")
    cfg = ChunkerConfig(chunk_id_strategy="content_hash")
    with pytest.raises(Exception) as excinfo:
        chunk_documents(src, out_dir=out, source_type="guide", cfg=cfg)
    assert not isinstance(excinfo.value, CorpusOverwriteError)


def test_guard_allows_an_empty_output_dir(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    src = tmp_path / "pdfs"
    src.mkdir()
    with pytest.raises(ValueError, match="no PDFs"):
        chunk_documents(src, out_dir=out, source_type="guide", cfg=ChunkerConfig())


def test_manifest_parsing_and_stem_matching(tmp_path):
    manifest = tmp_path / "m.txt"
    manifest.write_text(
        "filename: a.pdf | removed_pages: [1, 2, 3]\n"
        "filename: b | removed_pages: []\n"
        "garbage line\n"
    )
    excluded = load_excluded_pages(manifest)
    assert excluded == {"a.pdf": {1, 2, 3}, "b": set()}
    assert _pages_for("a.pdf", excluded) == {1, 2, 3}
    assert _pages_for("b.pdf", excluded) == set()
    assert _pages_for("unknown.pdf", excluded) == set()


def test_excluded_page_drops_the_whole_chunk():
    """Reproduces the notebooks: a chunk spanning a cover page is lost entirely."""
    rows = chunks_to_rows(
        [
            MergedChunk(text="good content", group_key="H", page_numbers=(5,)),
            MergedChunk(text="spans a cover", group_key="H", page_numbers=(1, 2)),
        ],
        source_type="guide",
        base_metadata={"file": "a.pdf"},
        cfg=ChunkerConfig(),
        source_doc_id="a.pdf",
        excluded_pages={1},
    )
    assert len(rows) == 1
    assert rows[0]["chunk_text"] == "good content"


def test_punctuation_only_chunks_are_dropped():
    rows = chunks_to_rows(
        [MergedChunk(text="...", group_key="H")],
        source_type="guide",
        base_metadata={},
        cfg=ChunkerConfig(),
        source_doc_id="a.pdf",
    )
    assert rows == []


def test_metadata_records_first_page_only():
    rows = chunks_to_rows(
        [MergedChunk(text="content", group_key="Ch 1", page_numbers=(7, 8, 9))],
        source_type="textbook",
        base_metadata={"file": "tb.pdf"},
        cfg=ChunkerConfig(),
        source_doc_id="tb.pdf",
    )
    metadata = eval(rows[0]["chunk_metadata"])
    assert metadata["page_number"] == 7
    assert metadata["heading"] == "Ch 1"
