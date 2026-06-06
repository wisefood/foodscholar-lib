"""Name → factory registry for Stage-1 summarizers.

Single source of truth: the builder selects one method via
`config.layer_c.stage1_method`; the benchmark harness iterates all of them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from foodscholar.layer_c.base import BaseSummarizer
from foodscholar.layer_c.summarizers import (
    NLTKFrequencySummarizer,
    SumyLexRankSummarizer,
    SumyLsaSummarizer,
    SumyLuhnSummarizer,
    SumyTextRankSummarizer,
)

if TYPE_CHECKING:
    from foodscholar.config import LayerCConfig

SUMMARIZERS: dict[str, Callable[[LayerCConfig], BaseSummarizer]] = {
    "lexrank": lambda c: SumyLexRankSummarizer(n=c.stage1_sentences),
    "lsa": lambda c: SumyLsaSummarizer(n=c.stage1_sentences),
    "luhn": lambda c: SumyLuhnSummarizer(n=c.stage1_sentences),
    "textrank": lambda c: SumyTextRankSummarizer(n=c.stage1_sentences),
    "nltk_freq": lambda c: NLTKFrequencySummarizer(n=c.stage1_sentences),
}


def build_summarizer(name: str, cfg: LayerCConfig) -> BaseSummarizer:
    """Return the BaseSummarizer for `name`, configured from `cfg`."""
    return SUMMARIZERS[name](cfg)


def all_summarizers(cfg: LayerCConfig) -> list[BaseSummarizer]:
    """Return one instance of every registered summarizer (for the harness)."""
    return [factory(cfg) for factory in SUMMARIZERS.values()]
