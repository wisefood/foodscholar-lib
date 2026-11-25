"""The three-stage harness, the gold loader and the BM25 baseline."""

from __future__ import annotations

import json

from foodscholar.chunking import Chunker, Heading, Paragraph, Region, SourceRegistry
from foodscholar.chunking.evaluation import (
    BM25Retriever,
    ChunkRecord,
    GoldStatement,
    evaluate,
    evaluate_retrieval,
    load_chunk_records,
    load_fskg_statements,
    statement_queries,
    support,
)
from tests.unit.chunking_fixtures import COUNTER, config, ir, write_xlsx


def test_support_is_content_token_containment_and_numbers_must_match():
    s = support("Adults should limit salt to 6 g a day.", "For adults, limit salt intake to 6 g per day.")
    assert s.score == 1.0 and s.numbers_ok
    wrong = support("Adults should limit salt to 6 g a day.", "For adults, limit salt intake to 5 g per day.")
    assert not wrong.numbers_ok and not wrong.supported(0.8)
    assert support("Oily fish weekly.", "Bread and rice.").score == 0.0


def _gold():
    return [
        GoldStatement(statement_id="S1", text="Eat oily fish once a week.", source_id="D", page=2, section="Fish"),
        GoldStatement(statement_id="S2", text="Limit salt to 6 g a day.", source_id="D", page=3, section="Salt"),
        GoldStatement(statement_id="S3", text="Drink water with meals.", source_id="MISSING", page=1),
    ]


def _chunks():
    return [
        ChunkRecord("c1", "D", "Eat oily fish such as salmon once a week.", 2, 2, "Fish", ("Fish",), ordinal=0),
        ChunkRecord("c2", "D", "Limit salt to", 3, 3, "Salt", ("Salt",), ordinal=1),
        ChunkRecord("c3", "D", "6 g a day for adults.", 3, 3, "Salt", ("Salt",), ordinal=2),
    ]


def test_assembly_stage_single_bundle_page_and_section():
    report = evaluate(_gold(), _chunks(), threshold=0.8)
    o = {x.statement_id: x for x in report.outcomes}
    assert o["S1"].single and o["S1"].page_ok and o["S1"].section_ok
    assert not o["S2"].single and o["S2"].bundle  # split across two chunks; their union supports it
    assert not o["S3"].scored
    t = report.totals
    assert t["n_scored"] == 2 and t["assembly_single"] == 0.5 and t["assembly_bundle"] == 1.0
    assert t["missing_sources"] == ["MISSING (1)"]
    assert "Supported by one chunk | 50.0%" in report.to_markdown()


def test_extraction_stage_and_page_straddle_subset():
    gold = [GoldStatement(statement_id="S", text="Eat oily fish once a week for omega-3 fats.", source_id="D", page=2)]
    report = evaluate(
        gold,
        [ChunkRecord("c", "D", "Eat oily fish once a week for omega-3 fats.", 2, 3)],
        ir_texts={"D": ["Eat oily fish", "once a week for omega-3 fats."]},
        ir_pages={"D": {2: "Eat oily fish once", 3: "a week for omega-3 fats."}},
    )
    o = report.outcomes[0]
    assert o.extraction == 1.0
    assert o.straddles_page_break is True
    assert report.totals["straddle_single"] == 1.0


def test_hygiene_flags_fragments_duplicates_and_estimates():
    report = evaluate([], [ChunkRecord("a", "D", "same text"), ChunkRecord("b", "D", "same  TEXT", tokens=600)])
    h = report.hygiene
    assert h["duplicates"] == 1 and h["share_over_max"] == 0.5 and h["estimated"] is True


def test_bm25_and_retrieval_at_k_and_token_budget():
    chunks = [ChunkRecord(f"x{i}", "D", f"filler text number {i} about bread", tokens=300) for i in range(30)]
    chunks.append(ChunkRecord("fish", "D", "Eat oily fish once a week.", tokens=300))
    gold = [GoldStatement(statement_id="S1", text="Eat oily fish once a week.", source_id="D")]
    assert BM25Retriever(chunks)("oily fish", 1) == ["fish"]
    metrics = evaluate_retrieval(statement_queries(gold), gold, chunks, ks=(1, 5), token_budgets=(200, 512))
    assert metrics["recall@1"] == 1.0 and metrics["mrr"] == 1.0
    assert metrics["recall@200tok"] == 0.0 and metrics["recall@512tok"] == 1.0


def test_fskg_workbook_maps_statements_to_sources_through_the_registry(tmp_path):
    reg = SourceRegistry.from_records([
        {"Source ID": "SRC-IE-001", "Source type": "guide", "Local copy": "a.pdf", "Statement ID range": "C0001-C0002"},
    ])
    wb = write_xlsx(tmp_path / "gold.xlsx", {
        "README": [["x"]],
        "Statements": [
            ["Statement ID", "Page (printed)", "Page (PDF)", "Section", "Statement text", "Statement type", "Quantity", "Population"],
            ["C0001", None, 3, "Fish", "Eat oily fish once a week.", "Recommendation", "1 per week", "adults"],
            ["C0002", 4, 6, "Salt", "Limit salt.", "Knowledge", "n/a", None],
        ],
    })
    gold = load_fskg_statements(wb, reg)
    assert [(g.statement_id, g.source_id, g.page, g.printed_page) for g in gold] == [("C0001", "SRC-IE-001", 3, None), ("C0002", "SRC-IE-001", 6, "4")]
    assert gold[0].qualifiers == {"Quantity": "1 per week", "Population": "adults"}


def test_any_pipeline_output_loads_jsonl_csv_and_maps_files_to_sources(tmp_path):
    reg = SourceRegistry.from_records([{"Source ID": "SRC-1", "Source type": "guide", "Local copy": "Guides/fsai-healthy-eating.pdf"}])
    (tmp_path / "other.jsonl").write_text(json.dumps({"doc_id": "fsai-healthy-eating.pdf", "text": "abc"}) + "\n" + json.dumps({"doc_id": "x.pdf", "text": "def"}) + "\n")
    recs = load_chunk_records(tmp_path / "other.jsonl", registry=reg)
    assert [(r.doc_id, r.ordinal) for r in recs] == [("SRC-1", 0), ("x.pdf", 0)]
    doc = ir([Heading(id="h", text="S", level=1), Paragraph(id="p", text="Eat fish.", regions=[Region(page=2)])], source_id="SRC-1")
    Chunker(config(), token_counter=COUNTER).assemble(doc).write(tmp_path / "run")
    v2 = load_chunk_records(tmp_path / "run" / "chunks.jsonl")
    csv_recs = load_chunk_records(tmp_path / "run")
    assert v2[0].doc_id == csv_recs[0].doc_id == "SRC-1"
    assert v2[0].page_start == csv_recs[0].page_start == 2
    assert csv_recs[0].section_path == ("S",)
