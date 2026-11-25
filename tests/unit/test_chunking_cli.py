"""`foodscholar chunk …` runs end to end without a facade or services."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from foodscholar.chunking import Heading, Paragraph, Region
from foodscholar.chunking.cli import app as chunk_app
from foodscholar.cli.main import app as main_app
from tests.unit.chunking_fixtures import ir, write_xlsx

runner = CliRunner()


def _ir_dir(tmp_path):
    d = tmp_path / "ir"
    d.mkdir()
    for sid in ("SRC-1", "SRC-2"):
        ir(
            [Heading(id="h", text="Fish", level=1), Paragraph(id="p", text="Eat oily fish once a week. " * 3, regions=[Region(page=2)])],
            source_id=sid,
            source_type="guide",
        ).save(d / f"{sid}.json")
    return d


def test_run_assembles_ir_into_corpus_csv_and_jsonl(tmp_path):
    out = tmp_path / "out"
    res = runner.invoke(chunk_app, ["run", "--ir", str(_ir_dir(tmp_path)), "--out", str(out), "--tokenizer", "whitespace"])
    assert res.exit_code == 0, res.output
    assert "2 document(s), 2 chunk(s)" in res.output
    assert (out / "chunks_guide.csv").exists() and (out / "chunks.jsonl").exists()
    again = runner.invoke(chunk_app, ["run", "--ir", str(tmp_path / "ir"), "--out", str(tmp_path / "out2"), "--previous", str(out / "chunks.jsonl"), "--tokenizer", "whitespace"])
    assert "2 reused, 0 new" in again.output


def test_run_on_source_files_says_which_phase_brings_the_head(tmp_path):
    src = tmp_path / "x.pdf"
    src.write_bytes(b"%PDF")
    unknown = runner.invoke(chunk_app, ["run", str(src), "--out", str(tmp_path / "o"), "--tokenizer", "whitespace"])
    assert unknown.exit_code == 2 and "cannot tell the source type" in unknown.output
    res = runner.invoke(chunk_app, ["run", str(src), "--source-type", "textbook", "--out", str(tmp_path / "o"), "--tokenizer", "whitespace"])
    assert res.exit_code == 2
    assert "long-document head (textbooks) is not implemented yet" in res.output


def test_eval_scores_a_run_against_a_gold_workbook(tmp_path):
    out = tmp_path / "out"
    runner.invoke(chunk_app, ["run", "--ir", str(_ir_dir(tmp_path)), "--out", str(out), "--tokenizer", "whitespace"])
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps([{"Source ID": "SRC-1", "Source type": "guide", "Local copy": "a.pdf", "Statement ID range": "C0001-C0009"}]))
    gold = write_xlsx(tmp_path / "g.xlsx", {"Statements": [["Statement ID", "Page (PDF)", "Section", "Statement text"], ["C0001", 2, "Fish", "Eat oily fish once a week."]]})
    report = tmp_path / "report.json"
    res = runner.invoke(chunk_app, ["eval", "--chunks", str(out / "chunks.jsonl"), "--gold", str(gold), "--registry", str(reg), "--retrieval", "--out", str(report)])
    assert res.exit_code == 0, res.output
    assert "Supported by one chunk | 100.0%" in res.output
    data = json.loads(report.read_text())
    assert data["totals"]["page_accuracy"] == 1.0 and data["retrieval"]["recall@5"] == 1.0


def test_inspect_ir_and_chunks(tmp_path):
    d = _ir_dir(tmp_path)
    res = runner.invoke(chunk_app, ["inspect", str(d / "SRC-1.json")])
    assert "heading=1, paragraph=1" in res.output
    runner.invoke(chunk_app, ["run", "--ir", str(d), "--out", str(tmp_path / "o"), "--tokenizer", "whitespace"])
    res = runner.invoke(chunk_app, ["inspect", str(tmp_path / "o" / "chunks.jsonl")])
    assert json.loads(res.output)["n_chunks"] == 2


def test_the_group_is_mounted_on_the_main_cli():
    res = runner.invoke(main_app, ["chunk", "--help"])
    assert res.exit_code == 0 and "eval" in res.output and "run" in res.output


def test_run_on_a_records_file_found_through_the_registry(tmp_path):
    d = tmp_path / "Sources" / "Journals" / "J"
    d.mkdir(parents=True)
    (d / "J_abstracts.json").write_text(json.dumps([
        {"id": "1", "doi": "10.1/x", "title": "Fibre", "abstract": "Purpose of review: Fibre. Recent findings: It helps. Summary: Eat it."},
    ]))
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps([{"Source ID": "SRC-JN-009", "Local copy": "Sources/Journals/J/J_manifest.xlsx"}]))
    out = tmp_path / "out"
    res = runner.invoke(chunk_app, ["run", str(d / "J_abstracts.json"), "--registry", str(reg), "--out", str(out), "--tokenizer", "whitespace"])
    assert res.exit_code == 0, res.output
    assert "1 document(s), 1 chunk(s)" in res.output
    record = json.loads((out / "chunks.jsonl").read_text().splitlines()[0])
    assert record["doc_id"] == "10.1/x" and record["collection_id"] == "SRC-JN-009"
