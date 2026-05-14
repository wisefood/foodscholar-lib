"""Tests for the NER adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from foodscholar.annotate.ner import KeywordNER
from foodscholar.ontology import FoodOnAPI, load_ontology
from foodscholar.storage.protocols import NER

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def api() -> FoodOnAPI:
    return FoodOnAPI(load_ontology(FIXTURES / "mini_foodon.obo"))


def test_keyword_ner_implements_protocol() -> None:
    assert isinstance(KeywordNER(["olive"]), NER)


def test_keyword_ner_finds_match() -> None:
    ner = KeywordNER(["olive oil", "apple"])
    out = ner.extract("Mediterranean diet rich in olive oil and apple slices.")
    assert {m.text for m in out} == {"olive oil", "apple"}


def test_keyword_ner_is_case_insensitive() -> None:
    ner = KeywordNER(["olive oil"])
    out = ner.extract("Olive Oil reduces risk.")
    assert len(out) == 1
    assert out[0].text == "Olive Oil"


def test_keyword_ner_word_boundary() -> None:
    ner = KeywordNER(["oil"])
    out = ner.extract("Toiling does not contain the word.")
    assert out == []


def test_keyword_ner_prefers_longest_match() -> None:
    ner = KeywordNER(["olive", "olive oil"])
    out = ner.extract("She uses olive oil daily.")
    # Both regex alternatives match the same span, longer one should win
    assert len(out) == 1
    assert out[0].text == "olive oil"


def test_keyword_ner_offsets_are_correct() -> None:
    ner = KeywordNER(["apple"])
    text = "An apple a day."
    out = ner.extract(text)
    assert len(out) == 1
    assert text[out[0].start : out[0].end] == "apple"


def test_keyword_ner_from_ontology(api: FoodOnAPI) -> None:
    ner = KeywordNER.from_ontology(api)
    out = ner.extract("Olive oil and apple slices are foods.")
    assert {m.text.lower() for m in out} >= {"olive oil", "apple"}


def test_keyword_ner_from_ontology_excludes_obsolete(api: FoodOnAPI) -> None:
    ner = KeywordNER.from_ontology(api)
    out = ner.extract("This contains a legacy term reference.")
    assert all(m.text.lower() != "legacy term" for m in out)


def test_keyword_ner_no_matches_empty_list() -> None:
    ner = KeywordNER(["olive"])
    assert ner.extract("Nothing here.") == []


def test_keyword_ner_empty_keywords_returns_no_matches() -> None:
    ner = KeywordNER([])
    assert ner.extract("olive oil and apples") == []
