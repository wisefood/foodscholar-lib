"""The record head (journal abstracts) and balanced packing."""

from __future__ import annotations

import json

import pytest

from foodscholar.chunking import Chunker, SourceRegistry, write_corpus_csv
from foodscholar.chunking.assembler import balanced_groups
from foodscholar.chunking.config import (
    AssemblyConfig,
    ChunkingConfig,
    RecordHeadOptions,
    SourceProfile,
)
from foodscholar.chunking.evaluation import (
    ChunkRecord,
    GoldStatement,
    Query,
    evaluate,
    evaluate_retrieval,
    load_queries_jsonl,
)
from foodscholar.chunking.heads import get_head
from foodscholar.chunking.heads.record import RecordHead, section_type, split_sections
from foodscholar.corpus.csv_reader import iter_csv_chunks
from tests.unit.chunking_fixtures import COUNTER


def _alnum(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


@pytest.mark.parametrize(
    "text",
    [
        "Purpose of review: Diets matter. Recent findings: Fibre helps. Summary: Eat fibre.",
        "Background Two trials disagreed. Objective This study compared doses. Results Doses differed.",
        "Purpose of reviewThe first years matter. Recent findings156 studies agree.",
        "PURPOSE OF REVIEW Obesity rises. RECENT FINDINGS Tools improved.",
    ],
)
def test_three_label_shapes_split_losslessly(text):
    sections = split_sections(text)
    assert len(sections) >= 2 and all(label for label, _ in sections)
    assert _alnum(" ".join(f"{label}: {body}" for label, body in sections)) == _alnum(text)


def test_unstructured_and_lookalikes_stay_one_paragraph():
    assert split_sections("The diet matters. Results were mixed.") == [(None, "The diet matters. Results were mixed.")]
    assert split_sections("Objective measures of diet are rare. They help.")[0][0] is None
    assert split_sections("Background: only one labelled section here.")[0][0] is None
    assert split_sections("Diets matter. Methods: we ran trials. Results: they worked.")[0][0] is None  # must start with a label


def test_section_types_normalise_and_unknown_labels_are_other():
    assert section_type("Recent findings") == "results"
    assert section_type("PURPOSE OF REVIEW") == "introduction"
    assert section_type("Systematic review registration") == "other"
    assert section_type(None) is None


RECORDS = [
    {"id": "J1-01", "doi": "10.1/a", "title": "Fibre and health", "year": 2024, "type": "Review", "pages": "1-24",
     "license": "CC BY 4.0", "keywords": "fibre, health",
     "abstract": "Purpose of review: Fibre matters. Recent findings: Fibre lowers risk. Summary: Eat more fibre."},
    {"id": "J1-02", "doi": "", "title": "No DOI", "year": 2025, "keywords": ["salt", "blood pressure"],
     "abstract": "Salt raises blood pressure in adults."},
    {"id": "J1-03", "doi": "10.1/c", "title": "Empty", "abstract": "  "},
]


@pytest.fixture
def records_file(tmp_path):
    d = tmp_path / "Sources" / "Journals" / "Journal One"
    d.mkdir(parents=True)
    (d / "J1_manifest.xlsx").write_bytes(b"")
    path = d / "J1_abstracts.json"
    path.write_text(json.dumps(RECORDS))
    return path


@pytest.fixture
def registry():
    return SourceRegistry.from_records([
        {"Source ID": "SRC-JN-001", "Short name": "Journal One", "Language": "English", "License ID": "CC-BY",
         "Local copy": "Sources/Journals/Journal One/J1_manifest.xlsx", "Statement ID range": "AR0001-AR0010"},
    ])


def test_the_record_head_is_registered_for_its_genre():
    assert isinstance(get_head("record"), RecordHead)


def test_records_become_documents_with_ids_metadata_and_sections(records_file, registry):
    record = registry.resolve(records_file)  # found through its catalogued sibling
    assert record is not None and record.source_id == "SRC-JN-001"
    docs = RecordHead().extract(records_file, record=record, config=ChunkingConfig(), source_type="abstract")
    assert [d.meta.source_id for d in docs] == ["10.1/a", "SRC-JN-001:J1-02"]  # the empty one is skipped
    a = docs[0]
    assert a.meta.short_name == "Fibre and health" and a.meta.licence == "CC BY 4.0" and a.meta.language == "en"
    assert a.meta.extra["collection_id"] == "SRC-JN-001" and a.meta.extra["journal"] == "Journal One"
    assert a.meta.extra["article_type"] == "Review" and a.meta.extra["article_pages"] == "1-24"
    assert a.meta.extra["record_id"] == "J1-01" and "pages" not in a.meta.extra and "abstract" not in a.meta.extra
    assert [(p.section_label, p.section_type) for p in a.elements] == [
        ("Purpose of review", "introduction"), ("Recent findings", "results"), ("Summary", "conclusion"),
    ]
    assert docs[1].meta.extra["keywords"] == "salt, blood pressure"
    assert docs[1].meta.licence == "CC-BY"  # falls back to the catalogued licence


def test_title_keywords_header_option(records_file):
    cfg = ChunkingConfig(sources={"abstract": SourceProfile(genre="record", record=RecordHeadOptions(header="title_keywords"))})
    doc = RecordHead().extract(records_file, record=None, config=cfg, source_type="abstract")[0]
    assert doc.meta.short_name == "Fibre and health | Keywords: fibre, health"
    assert doc.meta.extra["collection_id"] == "J1_abstracts"


def test_jsonl_and_custom_fields_for_another_domain(tmp_path):
    path = tmp_path / "releases.jsonl"
    path.write_text(json.dumps({"ref": "Q3", "headline": "Results Q3", "body": "Revenue rose 4%."}) + "\n")
    opts = RecordHeadOptions(text_field="body", title_field="headline", id_field="ref")
    cfg = ChunkingConfig(sources={"release": SourceProfile(genre="record", record=opts)})
    doc = RecordHead().extract(path, record=None, config=cfg, source_type="release")[0]
    assert doc.meta.source_id == "releases:Q3" and doc.elements[0].text == "Revenue rose 4%."


def _record_config(max_tokens: int) -> ChunkingConfig:
    return ChunkingConfig(sources={"abstract": SourceProfile(genre="record", assembly=AssemblyConfig(max_tokens=max_tokens, target_tokens=max_tokens, overlap_max_tokens=8, min_tokens=0))})


def test_an_abstract_that_fits_is_one_chunk_and_reads_back_as_an_abstract(records_file, registry, tmp_path):
    chunker = Chunker(_record_config(128), token_counter=COUNTER, registry=registry)
    result = chunker.chunk_path(records_file)
    assert len(result.chunks) == 2
    path = write_corpus_csv(result.corpus_rows(), tmp_path / "c.csv")
    chunks = {c.source_doc_id: c for c in iter_csv_chunks(path)}
    a = chunks["10.1/a"]
    assert a.section_type == "abstract" and a.year == 2024
    assert a.retrieval_text.startswith("Fibre and health\n")
    assert a.provenance.section_labels == ["Purpose of review", "Recent findings", "Summary"]
    assert a.provenance.collection_id == "SRC-JN-001"


def test_an_overflowing_abstract_is_cut_at_section_boundaries(tmp_path, registry):
    body = " ".join(f"Sentence {i} about fibre intake." for i in range(8))
    abstract = f"Background: {body} Methods: {body} Results: {body} Conclusions: {body}"
    path = tmp_path / "x_abstracts.json"
    path.write_text(json.dumps([{"id": "1", "doi": "10.9/z", "title": "T", "abstract": abstract}]))
    result = Chunker(_record_config(110), token_counter=COUNTER).chunk_path(path, source_type="abstract")
    assert len(result.chunks) == 2
    assert [c.section_types for c in result.chunks] == [["introduction", "methods"], ["results", "conclusion"]]
    assert len({c.parent_id for c in result.chunks}) == 1  # expand(..., "section") returns the whole abstract
    assert _alnum("".join(c.source_text for c in result.chunks)) == _alnum(abstract)
    rows = write_corpus_csv(result.corpus_rows(), tmp_path / "c.csv")
    assert {c.section_type for c in iter_csv_chunks(rows)} == {"abstract"}  # two sections each → generic type


def test_a_chunk_that_is_one_section_declares_its_type(tmp_path):
    body = " ".join(f"Sentence {i} about salt." for i in range(14))
    abstract = f"Objective: {body} Results: {body}"
    path = tmp_path / "y_abstracts.json"
    path.write_text(json.dumps([{"id": "1", "doi": "10.9/y", "title": "T", "abstract": abstract}]))
    result = Chunker(_record_config(110), token_counter=COUNTER).chunk_path(path, source_type="abstract")
    rows = write_corpus_csv(result.corpus_rows(), tmp_path / "c.csv")
    assert [c.section_type for c in iter_csv_chunks(rows)] == ["introduction", "results"]


@pytest.mark.parametrize(
    ("sizes", "available", "target", "expected"),
    [
        ([50, 50, 50], 400, 400, [(0, 3)]),            # fits, near target: whole
        ([300, 300, 20], 400, 512, [(0, 1), (1, 3)]),  # must cut: no stray 20-token tail
        ([100] * 6, 400, 200, [(0, 2), (2, 4), (4, 6)]),
        ([500, 10], 400, 400, [(0, 1), (1, 2)]),       # a single oversized unit stands alone
        ([], 400, 400, []),
    ],
)
def test_balanced_groups(sizes, available, target, expected):
    assert balanced_groups(sizes, available, target) == expected


def test_collection_level_gold_matches_record_chunks_and_doc_queries(tmp_path):
    chunks = [
        ChunkRecord("a1", "10.1/a", "Fibre lowers cardiovascular risk in adults.", collection_id="SRC-JN-001"),
        ChunkRecord("b1", "10.1/b", "Salt raises blood pressure.", collection_id="SRC-JN-001"),
    ]
    gold = [GoldStatement(statement_id="AR0001", text="Fibre lowers cardiovascular risk.", source_id="SRC-JN-001")]
    report = evaluate(gold, chunks)
    assert report.outcomes[0].scored and report.outcomes[0].single
    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps({"query_id": "q1", "text": "does salt raise blood pressure", "evidence_docs": ["10.1/b"]}) + "\n")
    queries = load_queries_jsonl(q)
    assert queries == [Query("q1", "does salt raise blood pressure", (), ("10.1/b",))]
    assert evaluate_retrieval(queries, gold, chunks, ks=(1,))["recall@1"] == 1.0
