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
