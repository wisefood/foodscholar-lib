"""The legacy profile is unchanged; v2 output feeds the existing ingest path."""

from __future__ import annotations

import ast

import foodscholar.chunking.legacy as legacy
import foodscholar.corpus.chunker as old_chunker
import foodscholar.corpus.window as old_window
from foodscholar.chunking import Chunker, Furniture, Heading, Paragraph, Region, write_corpus_csv
from foodscholar.chunking.window import FineUnit, build_overlapping_chunks
from foodscholar.corpus.csv_reader import iter_csv_chunks
from tests.unit.chunking_fixtures import COUNTER, config, ir


def _words(text: str) -> int:
    return len(text.split())


def test_old_import_paths_are_the_same_objects():
    assert old_chunker.chunk_documents is legacy.chunk_documents
    assert old_chunker.chunks_to_rows is legacy.chunks_to_rows
    assert old_window.build_overlapping_chunks is build_overlapping_chunks


def test_fine_units_from_ir_reproduce_the_top_level_grouping():
    doc = ir(
        [
            Furniture(id="f", text="header"),
            Heading(id="h1", text="Chapter 1", level=1),
            Heading(id="h2", text="Section 1.1", level=2),
            Paragraph(id="a", text="alpha beta", regions=[Region(page=3)]),
            Heading(id="h3", text="Chapter 2", level=1),
            Paragraph(id="b", text="gamma", regions=[Region(page=4), Region(page=5)]),
        ]
    )
    assert legacy.fine_units_from_ir(doc) == [
        FineUnit(text="alpha beta", group_key="Chapter 1", page_numbers=(3,)),
        FineUnit(text="gamma", group_key="Chapter 2", page_numbers=(4, 5)),
    ]


def test_legacy_assemble_is_the_legacy_window_over_the_same_units():
    elements = [Heading(id="h", text="Ch", level=1)] + [Paragraph(id=f"p{i}", text=f"w{i} " * 30, regions=[Region(page=i)]) for i in range(1, 9)]
    doc = ir(elements)
    expected = build_overlapping_chunks(legacy.fine_units_from_ir(doc), count_tokens=_words, max_tokens=100, overlap=20)
    got = legacy.legacy_assemble(doc, count_tokens=_words, max_tokens=100, overlap=20)
    assert [(m.text, m.page_numbers, m.group_key) for m in got] == [(m.text, m.page_numbers, m.group_key) for m in expected]


def test_v2_rows_ingest_through_the_existing_csv_reader(tmp_path):
    doc = ir(
        [
            Heading(id="h", text="Fats", level=1),
            Paragraph(id="p", text="Use olive oil. " * 50, regions=[Region(page=7), Region(page=8)]),
        ],
        source_id="SRC-IE-001",
        source_type="guide",
        file="fsai.pdf",
        short_name="FSAI guide",
        licence="IE-FSAI",
        page_labels={7: "5", 8: "6"},
        extra={"country": "Ireland"},
    )
    result = Chunker(config(), token_counter=COUNTER).assemble(doc)
    path = write_corpus_csv(result.corpus_rows(), tmp_path / "chunks_guide.csv")
    chunks = list(iter_csv_chunks(path))
    assert len(chunks) == len(result.chunks) > 1
    first = chunks[0]
    assert first.source_doc_id == "SRC-IE-001"
    assert first.text == result.chunks[0].source_text
    assert first.retrieval_text == "FSAI guide > Fats\n" + first.text
    prov = first.provenance
    assert prov.section_path == ["Fats"]
    assert (prov.page_start, prov.page_end, prov.printed_page_start) == (7, 8, "5")
    assert prov.next_chunk_id == chunks[1].chunk_id and prov.ordinal == 0
    assert prov.country == "Ireland" and prov.licence == "IE-FSAI"
    assert prov.citation_fields()["page_end"] == 8


def test_legacy_chunks_have_no_header_so_retrieval_text_is_the_text(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text('chunk_id,chunk_text,type,chunk_metadata\nc1,Some text,guide,"{\'file\': \'a.pdf\', \'pages\': \'90\'}"\n')
    chunk = next(iter_csv_chunks(path))
    assert chunk.retrieval_text == "Some text"
    assert chunk.provenance.pages == 90  # the guide page count still coerces


def test_metadata_drops_empty_values_and_keeps_domain_extras_under_core_keys():
    doc = ir([Paragraph(id="p", text="x")], extra={"country": "Ireland", "source_id": "spoofed"})
    result = Chunker(config(), token_counter=COUNTER).assemble(doc)
    md = ast.literal_eval(result.corpus_rows()[0]["chunk_metadata"])
    assert md["source_id"] == "DOC" and md["country"] == "Ireland"
    assert "figure_ids" not in md and "supersedes" not in md
