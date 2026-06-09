"""Stage 2: extract -> Card via a mock LLMClient."""

from __future__ import annotations

import pytest

from foodscholar.config import LayerCConfig
from foodscholar.layer_c.models import Stage1Output
from foodscholar.layer_c.stage2 import run_stage2


class _StubTheme:
    def __init__(self, tid: str, label: str, facet: str = "foods",
                 keyword_terms=None) -> None:
        self.theme_id = tid
        self.label = label
        self.facet = facet
        self.keyword_terms = keyword_terms or ["oat", "fiber"]


class _OKJsonLLM:
    model_id = "stub-llm"

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.last_prompt = None

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:  # pragma: no cover
        return ""

    def generate_json(self, prompt, schema, max_tokens=1024):
        self.last_prompt = prompt
        return dict(self._payload)


_GOOD = {
    "title": "Oats and heart health",
    "summary": "Oats contain beta glucan, a soluble fiber linked to lower cholesterol.",
    "tip": "Choose whole-grain oats.",
    "evidence_quality": "high",
    "controversy_note": None,
    "confidence_note": None,
}


def _stage1() -> Stage1Output:
    return Stage1Output(text="Oats have beta glucan.", n_input_chunks=3,
                        n_input_chars=50, strategy="single", n_groups=1)


def test_stage2_builds_card() -> None:
    llm = _OKJsonLLM(_GOOD)
    card = run_stage2(llm, _stage1(), _StubTheme("t1", "Oats"),
                      ["c1", "c2", "c3"], LayerCConfig())
    assert card.target_id == "t1"
    assert card.target_type == "theme"
    assert card.title == "Oats and heart health"
    assert card.evidence_quality == "high"
    assert card.cited_chunk_ids == ["c1", "c2", "c3"]
    assert card.llm_model == LayerCConfig().llm_model
    assert card.prompt_version == "v1"
    assert card.safety_flagged is False


def test_stage2_keeps_evidence_sentences_from_extract() -> None:
    """The Card retains the intermediary extractive sentences fed to the LLM,
    so the card's provenance can be inspected after the fact."""
    llm = _OKJsonLLM(_GOOD)
    s1 = Stage1Output(
        text="Oats have beta glucan. Beta glucan lowers cholesterol.",
        n_input_chunks=3, n_input_chars=80, strategy="single", n_groups=1,
    )
    card = run_stage2(llm, s1, _StubTheme("t1", "Oats"), ["c1"], LayerCConfig())
    assert card.evidence_sentences == [
        "Oats have beta glucan.",
        "Beta glucan lowers cholesterol.",
    ]


def test_stage2_prompt_uses_extract_not_chunks() -> None:
    llm = _OKJsonLLM(_GOOD)
    run_stage2(llm, _stage1(), _StubTheme("t1", "Oats"), ["c1"], LayerCConfig())
    assert "beta glucan" in llm.last_prompt  # the extract is in the prompt
    assert "Oats" in llm.last_prompt          # theme label too


def test_stage2_safety_flag_on_sensitive_facet() -> None:
    llm = _OKJsonLLM(_GOOD)
    cfg = LayerCConfig(safety_sensitive_facets=["allergies"])
    card = run_stage2(llm, _stage1(), _StubTheme("t2", "Peanut", facet="allergies"),
                      ["c1"], cfg)
    assert card.safety_flagged is True


def test_stage2_strict_grounding_rejects_overlong() -> None:
    over = {**_GOOD, "summary": "x" * 10}
    llm = _OKJsonLLM(over)
    cfg = LayerCConfig(grounding_check="strict", max_summary_chars=5)
    with pytest.raises(ValueError):
        run_stage2(llm, _stage1(), _StubTheme("t3", "Oats"), ["c1"], cfg)


def test_stage2_off_grounding_allows_overlong() -> None:
    over = {**_GOOD, "summary": "x" * 10}
    llm = _OKJsonLLM(over)
    cfg = LayerCConfig(grounding_check="off", max_summary_chars=5)
    card = run_stage2(llm, _stage1(), _StubTheme("t3", "Oats"), ["c1"], cfg)
    assert len(card.summary) == 10
