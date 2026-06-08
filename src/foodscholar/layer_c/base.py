"""BaseSummarizer contract + a lightweight sentence splitter.

The splitter is regex-based and dependency-free so the ABC and helpers import
without sumy/nltk. Concrete summarizers in `summarizers.py` lazy-import their
heavy backends.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MAX_SENT_CHARS = 2000  # clamp pseudo-sentences (mangled tables with no terminator)
_WORD = re.compile(r"[A-Za-z]{2,}")
# A run of pipe-delimited cells (`| a | 1 | 2 |`) — a markdown table fragment that
# can be embedded inside an otherwise-prose split.
_TABLE_RUN = re.compile(r"(?:\|[^|]*){2,}\|")


def _is_prose(s: str) -> bool:
    """True if `s` reads like a claim, not a numeric/symbol fragment."""
    n_words = len(_WORD.findall(s))
    if n_words >= 3:
        return True
    alpha = sum(c.isalpha() for c in s)
    return alpha >= 0.4 * max(1, len(s))


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty, stripped prose sentences (regex, no deps).

    Embedded markdown-table runs are stripped, non-prose fragments are dropped,
    and over-long pseudo-sentences are clamped — so a single mangled table can't
    masquerade as one giant sentence (it defeated the splitter on the real
    corpus).
    """
    if not text or not text.strip():
        return []
    out: list[str] = []
    for raw in _SENT_SPLIT.split(text.strip()):
        # excise table runs first, then keep whatever prose remains
        s = _TABLE_RUN.sub(" ", raw).strip()
        if not s or not _is_prose(s):
            continue
        # clamp a runaway pseudo-sentence (no terminal punctuation) to whole words
        while len(s) > _MAX_SENT_CHARS:
            cut = s.rfind(" ", 0, _MAX_SENT_CHARS)
            if cut <= 0:
                cut = _MAX_SENT_CHARS
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            out.append(s)
    return out


class BaseSummarizer(ABC):
    """Common interface for every Stage-1 extractive method.

    Implementations take a list of chunk texts and return a single extractive
    summary string. The sentence budget is supplied at construction time.
    """

    name: str = "base"

    @abstractmethod
    def summarize(self, chunks: list[str]) -> str:
        """Return an extractive summary of `chunks` (joined text)."""
        ...
