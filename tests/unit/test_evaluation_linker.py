"""Tests for the linker evaluation gate (BRIEF §17)."""

from __future__ import annotations

from pathlib import Path

import pytest

from foodscholar.annotate.linker import ThreeTierLinker
from foodscholar.evaluation import evaluate_linker, load_linker_gold
from foodscholar.ontology import FoodOnAPI, load_ontology

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def api() -> FoodOnAPI:
    return FoodOnAPI(load_ontology(FIXTURES / "mini_foodon.obo"), prefix_filter=None)


def test_load_gold_parses_jsonl() -> None:
    gold = load_linker_gold(FIXTURES / "linker_gold.jsonl")
    assert len(gold) > 20
    assert gold[0].text == "olive oil"
    assert gold[0].expected_id == "TEST:0000008"


def test_evaluate_meets_brief_coverage_gate(api: FoodOnAPI) -> None:
    """BRIEF §17 requires entity-linking coverage ≥ 70% on the gold set."""
    linker = ThreeTierLinker(api)
    report = evaluate_linker(linker, load_linker_gold(FIXTURES / "linker_gold.jsonl"))
    assert report.coverage >= 0.70, report.summary()


def test_evaluate_negative_cases_correct(api: FoodOnAPI) -> None:
    linker = ThreeTierLinker(api)
    report = evaluate_linker(linker, load_linker_gold(FIXTURES / "linker_gold.jsonl"))
    miss_total = report.by_tier_total.get("miss", 0)
    miss_correct = report.by_tier_correct.get("miss", 0)
    assert miss_total > 0
    # Negative cases (foods absent from the mini ontology) should all be linked to None
    assert miss_correct == miss_total


def test_report_summary_fields(api: FoodOnAPI) -> None:
    linker = ThreeTierLinker(api)
    report = evaluate_linker(linker, load_linker_gold(FIXTURES / "linker_gold.jsonl"))
    summary = report.summary()
    assert {"total", "correct", "linked", "coverage", "accuracy", "by_tier"} <= summary.keys()
