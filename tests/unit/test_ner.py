"""Tests for the NER adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from foodscholar.annotate.ner import KeywordNER, simplify_label
from foodscholar.io.ontology import OntologyTerm
from foodscholar.ontology import FoodOnAPI, load_ontology
from foodscholar.storage.protocols import NER

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def api() -> FoodOnAPI:
    return FoodOnAPI(load_ontology(FIXTURES / "mini_foodon.obo"), prefix_filter=None)


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


# --------------------------------------------------------------- simplify_label


def test_simplify_label_strips_parenthetical() -> None:
    assert simplify_label("red meat (raw)") == "red meat"
    assert simplify_label("red meat (eurofir)") == "red meat"


def test_simplify_label_strips_code_prefix() -> None:
    assert simplify_label("10210 - legumes (efsa foodex2)") == "legumes"


def test_simplify_label_strips_trailing_category() -> None:
    assert simplify_label("legume food product") == "legume"
    assert simplify_label("legume animal feed plant") == "legume"


def test_simplify_label_leaves_clean_label_untouched() -> None:
    assert simplify_label("olive oil") == "olive oil"


# --------------------------------------------------- from_ontology(expand_labels)


def _qualified_ontology() -> FoodOnAPI:
    """Ontology with FoodOn-style over-qualified labels."""
    terms = [
        OntologyTerm(id="FOODON:1", label="red meat (raw)"),
        OntologyTerm(id="FOODON:2", label="legume food product"),
        OntologyTerm(id="FOODON:3", label="10210 - cereals (efsa foodex2)"),
    ]
    return FoodOnAPI(terms, prefix_filter=None)


def test_from_ontology_expand_labels_catches_simplified_form() -> None:
    ner = KeywordNER.from_ontology(_qualified_ontology(), expand_labels=True)
    # KeywordNER is exact word-boundary match — use the simplified forms that
    # expand_labels produces ("red meat", "legume", "cereals"). Plurals like
    # "legumes" are the linker's fuzzy tier's job, not the NER's.
    out = ner.extract("This sample contains legume, red meat, and cereals.")
    found = {m.text.lower() for m in out}
    assert "red meat" in found    # from "red meat (raw)"
    assert "legume" in found      # from "legume food product"
    assert "cereals" in found     # from "10210 - cereals (efsa foodex2)"


def test_from_ontology_without_expand_misses_qualified_labels() -> None:
    ner = KeywordNER.from_ontology(_qualified_ontology(), expand_labels=False)
    out = ner.extract("This sentence mentions red meat directly.")
    # Raw label is "red meat (raw)" — bare "red meat" won't match without expansion.
    assert all(m.text.lower() != "red meat" for m in out)


def test_from_ontology_min_keyword_len_drops_short_terms() -> None:
    terms = [
        OntologyTerm(id="FOODON:1", label="an"),       # 2 chars — noise
        OntologyTerm(id="FOODON:2", label="olive oil"),
    ]
    ner = KeywordNER.from_ontology(FoodOnAPI(terms, prefix_filter=None), min_keyword_len=3)
    out = ner.extract("an olive oil sample")
    assert all(m.text.lower() != "an" for m in out)
    assert any(m.text.lower() == "olive oil" for m in out)
