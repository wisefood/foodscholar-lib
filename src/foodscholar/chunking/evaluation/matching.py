"""Deciding whether a passage supports a gold statement — deterministically.

Gold statements are hand-extracted and often paraphrased, so exact string
matching under-counts. The score here is *content-token containment*: the
share of the statement's content words (lower-cased, lightly stemmed, stop
words dropped) that the passage contains, counted with multiplicity. A
statement is supported when the score clears a threshold **and** every number
in the statement appears in the passage — a quantity is the qualifier a
nutrition claim most often loses.

This is a triage signal, not ground truth. Borderline cases go to human
adjudication; an LLM judge may help triage but is never the sole validator.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

_WORD_RE = re.compile(r"[^\W_]+(?:[.,]\d+)?", re.UNICODE)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

# English function words. Kept small on purpose: content words carry the claim.
_STOPWORD_TEXT = """a an the and or but if then than so of in on at to for from by with without about into over
    under is are was were be been being am do does did done has have had having it its this that these
    those there here as such can could may might must should would will shall not no nor very more most
    less least also just only each every any all some your you yours our we they them their he she his
    her i me my which who whom whose what when where why how other per up out off"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


@lru_cache(maxsize=65536)
def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed") and not word.endswith("eed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def content_tokens(text: str) -> list[str]:
    out: list[str] = []
    for raw in _WORD_RE.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        if raw[0].isdigit():
            out.append(raw.replace(",", "."))
        elif len(raw) > 1:
            out.append(_stem(raw))
    return out


def numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUMBER_RE.findall(text)}


@dataclass(frozen=True)
class Support:
    score: float
    """Share of the statement's content tokens found in the passage, 0..1."""
    numbers_ok: bool
    """Every number in the statement appears in the passage."""

    def supported(self, threshold: float) -> bool:
        return self.numbers_ok and self.score >= threshold


class PassageIndex:
    """Pre-tokenised passages, so one statement is scored against many cheaply."""

    def __init__(self, texts: list[str]) -> None:
        self.counts = [Counter(content_tokens(t)) for t in texts]
        self.numbers = [numbers(t) for t in texts]

    def support(self, statement: str, i: int) -> Support:
        return support_counts(Counter(content_tokens(statement)), numbers(statement), self.counts[i], self.numbers[i])

    def support_union(self, statement: str, indices: list[int]) -> Support:
        merged: Counter[str] = Counter()
        nums: set[str] = set()
        for i in indices:
            merged |= self.counts[i]
            nums |= self.numbers[i]
        return support_counts(Counter(content_tokens(statement)), numbers(statement), merged, nums)


def support_counts(stmt: Counter[str], stmt_numbers: set[str], passage: Counter[str], passage_numbers: set[str]) -> Support:
    total = sum(stmt.values())
    if total == 0:
        return Support(0.0, not stmt_numbers or stmt_numbers <= passage_numbers)
    hit = sum(min(n, passage[t]) for t, n in stmt.items())
    return Support(hit / total, stmt_numbers <= passage_numbers)


def support(statement: str, passage: str) -> Support:
    return PassageIndex([passage]).support(statement, 0)
