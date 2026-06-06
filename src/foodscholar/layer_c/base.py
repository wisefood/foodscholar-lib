"""BaseSummarizer contract + a lightweight sentence splitter.

The splitter is regex-based and dependency-free so the ABC and helpers import
without sumy/nltk. Concrete summarizers in `summarizers.py` lazy-import their
heavy backends.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty, stripped sentences (regex, no deps)."""
    if not text or not text.strip():
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


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
