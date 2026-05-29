"""Theme labeling — keyword (c-TF-IDF) + LLM polish."""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

from foodscholar.config import LabelingConfig
from foodscholar.io.chunk import Chunk
from foodscholar.layer_b.label import label_by_keywords, label_by_llm


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=text,
        source_doc_id="d",
        source_type="abstract",
        section_type="abstract",
    )


def test_label_by_keywords_returns_top_k_terms() -> None:
    """Theme 0 (calcium/bone) vs theme 1 (cholesterol/cardiovascular) —
    c-TF-IDF should foreground each theme's distinctive vocabulary."""
    theme_chunks = {
        0: [
            _chunk("c1", "bone density calcium intake adolescents"),
            _chunk("c2", "calcium absorption bone health postmenopausal"),
            _chunk("c3", "dietary calcium and bone mineral content"),
        ],
        1: [
            _chunk("c4", "cholesterol cardiovascular disease risk"),
            _chunk("c5", "LDL cholesterol heart disease mortality"),
            _chunk("c6", "cardiovascular health statins cholesterol"),
        ],
    }
    cfg = LabelingConfig(strategy="keyword", top_keywords=3)
    labels = label_by_keywords(theme_chunks, cfg)
    assert set(labels.keys()) == {0, 1}
    t0_terms = labels[0]
    t1_terms = labels[1]
    assert any("calcium" in t or "bone" in t for t in t0_terms)
    assert any("cholesterol" in t or "cardiovascular" in t for t in t1_terms)


def test_label_by_keywords_single_theme_returns_top_tokens() -> None:
    """Single theme: TF-IDF degenerates to raw TF, but the function still
    returns up to top_k tokens."""
    theme_chunks = {0: [_chunk("c1", "olive oil mediterranean diet")]}
    cfg = LabelingConfig(strategy="keyword", top_keywords=2)
    labels = label_by_keywords(theme_chunks, cfg)
    assert 0 in labels
    assert 1 <= len(labels[0]) <= 2


def test_label_by_keywords_empty_returns_empty() -> None:
    cfg = LabelingConfig(strategy="keyword", top_keywords=3)
    assert label_by_keywords({}, cfg) == {}


def test_label_by_keywords_filters_ocr_codes_and_id_leakage() -> None:
    """OCR codes (`h18567`), ontology-id leakage (`FOODON1234`), and 1-2 char
    fragments must not surface as keywords — they pollute LLM labels."""
    theme_chunks = {
        0: [
            _chunk("c1", "calcium intake bone density h18567 mg supplementation"),
            _chunk("c2", "FOODON12345 calcium bone health w x intake"),
        ],
    }
    cfg = LabelingConfig(strategy="keyword", top_keywords=5)
    labels = label_by_keywords(theme_chunks, cfg)
    for term in labels[0]:
        assert "h18567" not in term
        assert "FOODON" not in term
        # No single letters or all-uppercase IDs.
        for tok in term.split():
            assert len(tok) >= 3
            assert not (tok.isupper() and len(tok) >= 4)


def test_label_by_llm_passes_keywords_and_sample_chunks() -> None:
    """label_by_llm formats a prompt with keywords + sample chunks and
    returns the LLM's stripped single-line label per theme."""

    class MockLLM:
        model_id = "mock"
        def generate(self, prompt: str, max_tokens: int = 64) -> str:
            assert "calcium" in prompt
            assert "bone" in prompt
            return "Bone health and calcium intake"
        def generate_json(self, prompt, schema, max_tokens=1024):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    theme_chunks = {
        0: [
            _chunk("c1", "calcium intake and bone density in adolescents"),
            _chunk("c2", "calcium absorption and bone health"),
        ],
    }
    keywords = {0: ["calcium", "bone", "density"]}
    cfg = LabelingConfig(strategy="llm", llm_max_tokens=32)
    labels = label_by_llm(theme_chunks, keywords, MockLLM(), cfg)
    assert labels[0] == "Bone health and calcium intake"


def test_label_by_llm_strips_quotes_around_label() -> None:
    """LLMs sometimes return quoted labels; the labeler strips them."""

    class QuotingLLM:
        model_id = "mock"
        def generate(self, prompt: str, max_tokens: int = 64) -> str:
            return '"Olive oil and cardiovascular health"'
        def generate_json(self, prompt, schema, max_tokens=1024):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    theme_chunks = {0: [_chunk("c1", "olive oil mediterranean")]}
    keywords = {0: ["olive", "oil"]}
    cfg = LabelingConfig(strategy="llm")
    labels = label_by_llm(theme_chunks, keywords, QuotingLLM(), cfg)
    assert labels[0] == "Olive oil and cardiovascular health"


def test_label_by_llm_falls_back_to_keyword_when_llm_blank() -> None:
    """If the LLM returns whitespace-only, fall back to the top keyword
    rather than leaving an unlabeled theme in the graph."""

    class BlankLLM:
        model_id = "mock"
        def generate(self, prompt: str, max_tokens: int = 64) -> str:
            return "   \n  "
        def generate_json(self, prompt, schema, max_tokens=1024):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    theme_chunks = {0: [_chunk("c1", "calcium bone")]}
    keywords = {0: ["calcium", "bone"]}
    cfg = LabelingConfig(strategy="llm")
    labels = label_by_llm(theme_chunks, keywords, BlankLLM(), cfg)
    assert labels[0] == "calcium"


def test_label_by_llm_handles_themes_with_fewer_than_3_chunks() -> None:
    """A theme with 1-2 chunks shouldn't crash the 3-slot prompt — cycle to fill."""

    class EchoLLM:
        model_id = "mock"
        def generate(self, prompt: str, max_tokens: int = 64) -> str:
            return "Single-chunk theme label"
        def generate_json(self, prompt, schema, max_tokens=1024):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    theme_chunks = {0: [_chunk("c1", "only one chunk in this theme")]}
    keywords = {0: ["chunk"]}
    cfg = LabelingConfig(strategy="llm")
    labels = label_by_llm(theme_chunks, keywords, EchoLLM(), cfg)
    assert labels[0] == "Single-chunk theme label"
