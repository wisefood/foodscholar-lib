"""Layer C summarizer registry."""

from __future__ import annotations

import pytest

from foodscholar.config import LayerCConfig
from foodscholar.layer_c.registry import (
    SUMMARIZERS,
    all_summarizers,
    build_summarizer,
)


def test_registry_has_five_methods() -> None:
    assert set(SUMMARIZERS) == {"lexrank", "lsa", "luhn", "textrank", "nltk_freq"}


def test_build_summarizer_returns_named() -> None:
    cfg = LayerCConfig(stage1_sentences=5)
    s = build_summarizer("nltk_freq", cfg)
    assert s.name == "nltk_freq"
    assert s.n == 5


def test_build_summarizer_unknown_raises() -> None:
    with pytest.raises(KeyError):
        build_summarizer("bogus", LayerCConfig())


def test_all_summarizers_returns_five() -> None:
    methods = {s.name for s in all_summarizers(LayerCConfig())}
    assert methods == {"lexrank", "lsa", "luhn", "textrank", "nltk_freq"}
