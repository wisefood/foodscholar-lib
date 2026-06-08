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


def test_split_sentences_drops_markdown_table_rows() -> None:
    # Tabular nutrition rows (lots of pipes / digits) must NOT survive as
    # "sentences" — they defeated the splitter on the real corpus.
    text = (
        "Milk provides most of the calcium in the U.S. diet. "
        "| whole | 150 | 275 | 8 | 4.5 | 25 | "
        "| 1% low-fat | 100 | 290 | 2 | 1.5 | 10 | "
        "About 30 percent of calcium is absorbed from milk."
    )
    sents = split_sentences(text)
    joined = " ".join(sents)
    assert "Milk provides most of the calcium" in joined
    assert "About 30 percent of calcium is absorbed" in joined
    assert "| whole |" not in joined
    assert "| 1% low-fat |" not in joined


def test_split_sentences_keeps_prose_with_a_few_numbers() -> None:
    # A normal sentence that merely mentions numbers must be kept.
    text = "Adults need about 1,200 milligrams of calcium and 600 IU of vitamin D daily."
    sents = split_sentences(text)
    assert len(sents) == 1
    assert sents[0].startswith("Adults need")


def test_split_sentences_clamps_giant_pseudo_sentence() -> None:
    # A pipe-free but absurdly long run with no terminal punctuation (e.g. a
    # mangled table) must not pass through as one multi-KB "sentence".
    blob = "col " * 4000  # ~16 KB, no '.', no pipes
    sents = split_sentences(blob)
    assert all(len(s) <= 2000 for s in sents)


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


pytest.importorskip("sumy")
from foodscholar.layer_c.summarizers import (  # noqa: E402
    SumyLexRankSummarizer,
    SumyLsaSummarizer,
    SumyLuhnSummarizer,
    SumyTextRankSummarizer,
)


@pytest.mark.parametrize(
    "cls",
    [SumyLexRankSummarizer, SumyLsaSummarizer, SumyLuhnSummarizer, SumyTextRankSummarizer],
)
def test_sumy_methods_respect_budget(cls) -> None:
    out = cls(n=2).summarize(_DOCS)
    assert out
    assert len(split_sentences(out)) <= 2


@pytest.mark.parametrize(
    "cls",
    [SumyLexRankSummarizer, SumyLsaSummarizer, SumyLuhnSummarizer, SumyTextRankSummarizer],
)
def test_sumy_methods_empty_input(cls) -> None:
    assert cls(n=3).summarize([]) == ""


def test_sumy_names() -> None:
    assert SumyLexRankSummarizer(n=1).name == "lexrank"
    assert SumyLsaSummarizer(n=1).name == "lsa"
    assert SumyLuhnSummarizer(n=1).name == "luhn"
    assert SumyTextRankSummarizer(n=1).name == "textrank"
