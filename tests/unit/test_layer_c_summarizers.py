"""Layer C extractive summarizers + the BaseSummarizer contract."""

from __future__ import annotations

import pytest

from foodscholar.layer_c.base import BaseSummarizer, split_sentences


class _Echo(BaseSummarizer):
    name = "echo"

    def summarize(self, chunks: list[str]) -> str:
        return " ".join(chunks)


def test_base_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseSummarizer()  # type: ignore[abstract]


def test_concrete_subclass_runs() -> None:
    assert _Echo().summarize(["a", "b"]) == "a b"


def test_split_sentences_counts() -> None:
    text = "Apples are sweet. Pears are juicy. Rice is a grain."
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0].startswith("Apples")


def test_split_sentences_empty() -> None:
    assert split_sentences("") == []
    assert split_sentences("   ") == []


from foodscholar.layer_c.summarizers import NLTKFrequencySummarizer  # noqa: E402

_NLTK = pytest.importorskip("nltk")

_DOCS = [
    "Oats are a whole grain rich in soluble fiber called beta glucan.",
    "Beta glucan in oats can lower cholesterol and improve heart health.",
    "Rice is a staple cereal grain eaten across the world.",
    "Wheat flour is milled from wheat and used to bake bread.",
    "Barley is another cereal grain used in soups and brewing.",
]


def test_nltk_freq_respects_budget() -> None:
    s = NLTKFrequencySummarizer(n=2)
    out = s.summarize(_DOCS)
    assert out  # non-empty
    assert len(split_sentences(out)) <= 2


def test_nltk_freq_empty_input() -> None:
    assert NLTKFrequencySummarizer(n=3).summarize([]) == ""
    assert NLTKFrequencySummarizer(n=3).summarize(["", "   "]) == ""


def test_nltk_freq_fewer_than_budget_returns_all() -> None:
    s = NLTKFrequencySummarizer(n=10)
    out = s.summarize(["Only one sentence here."])
    assert "Only one sentence here." in out
