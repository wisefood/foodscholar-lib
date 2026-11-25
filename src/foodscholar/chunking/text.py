"""Text utilities with no model behind them: token counters and a sentence splitter.

The defaults here are deliberately simple and dependency-free so the assembler
and its tests run anywhere. Production runs plug in the embedder's tokenizer
(:func:`hf_token_counter`) and, for Hungarian and Slovene, a trained splitter.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

_SPACE_RE = re.compile(r"\s+")


def normalize_space(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


class WhitespaceTokenCounter:
    """One token per whitespace-separated word, plus two special tokens.

    A stand-in for tests and dry runs. It undercounts WordPiece by roughly a
    third on English prose, so never use it to size production chunks.
    """

    special_tokens = 2

    def count(self, text: str) -> int:
        return len(text.split())


class HFTokenCounter:
    """Counts with a Hugging Face tokenizer, loaded on first use.

    `count` uses ``tokenize`` (no special tokens), which for WordPiece is
    additive over whitespace-joined pieces — the assembler relies on that and
    still re-checks every finished chunk against the full encoding.
    """

    def __init__(self, name: str, *, use_fast: bool = True) -> None:
        self.name = name
        self.use_fast = use_fast
        self._tok: Any = None
        self._special: int | None = None

    def _load(self) -> Any:
        if self._tok is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as e:  # pragma: no cover - environment dependent
                raise ImportError(
                    "counting tokens with a Hugging Face tokenizer needs `transformers`. "
                    "Install with: pip install 'foodscholar[chunking]'"
                ) from e
            self._tok = AutoTokenizer.from_pretrained(self.name, use_fast=self.use_fast)
            self._special = int(self._tok.num_special_tokens_to_add(pair=False))
        return self._tok

    @property
    def special_tokens(self) -> int:
        self._load()
        return int(self._special or 0)

    def count(self, text: str) -> int:
        return len(self._load().tokenize(text)) if text else 0


@lru_cache(maxsize=4)
def hf_token_counter(name: str, use_fast: bool = True) -> HFTokenCounter:
    """A cached :class:`HFTokenCounter` per tokenizer name."""
    return HFTokenCounter(name, use_fast=use_fast)


# Abbreviations that end in a period but do not end a sentence. Lower-cased,
# without the trailing period.
_ABBREVIATION_TEXT = (
    "e.g i.e etc vs fig figs eq eqs no nos vol vols al approx ca cf dr mr mrs ms prof st "
    "jan feb mar apr jun jul aug sep sept oct nov dec min max incl excl resp"
)
_ABBREVIATIONS = frozenset(_ABBREVIATION_TEXT.split())
# Typographic quotes are intended: a sentence may end inside one.
_BOUNDARY_RE = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+(?=[\"'“‘(\[]?[A-ZÁÉÍÓÖŐÚÜŰČŠŽ0-9])")  # noqa: RUF001


class RegexSentenceSplitter:
    """Splits on terminal punctuation followed by a capital or a digit.

    Language-agnostic and model-free. Protects common abbreviations and
    decimal numbers. Good enough for packing decisions; Hungarian and Slovene
    get a trained splitter (SaT) in the leaflet head.
    """

    def split(self, text: str, language: str | None = None) -> list[str]:
        text = text.strip()
        if not text:
            return []
        out: list[str] = []
        start = 0
        for match in _BOUNDARY_RE.finditer(text):
            candidate = text[start : match.start()].rstrip()
            last = candidate.rsplit(None, 1)[-1] if candidate else ""
            token = last.rstrip(".").lower().lstrip("(")
            if token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
                continue  # "e.g. Apples", "J. Smith"
            out.append(text[start : match.end()].strip())
            start = match.end()
        tail = text[start:].strip()
        if tail:
            out.append(tail)
        return [s for s in out if s]
