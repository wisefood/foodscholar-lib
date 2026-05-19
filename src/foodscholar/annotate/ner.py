"""Keyword NER.

`KeywordNER` — deterministic, dependency-free. Surfaces any term in a provided
dictionary that appears in the text. Used by tests and as the safe offline
default for the in-memory facade. Implements the `NER` protocol from
`foodscholar.storage.protocols`.

The LLM-driven NER (`AgenticNER`) lives in `agent_ner.py`. A bespoke
fine-tuned model adapter (SciFoodNER) was removed — the project moved to
agentic NER and away from proprietary ML models.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from foodscholar.io.chunk import Mention

if TYPE_CHECKING:
    from foodscholar.ontology import FoodOnAPI

# FoodOn labels carry systematic NLP noise: a leading EFSA/EC numeric code,
# parenthetical qualifiers like "(raw)" / "(eurofir)" / "(efsa foodex2)", and a
# trailing " food product" category word. None of these appear in prose, so a
# verbatim keyword built from the raw label never matches real text. The
# simplifier strips them so e.g. "red meat (raw)" also yields "red meat".
_CODE_PREFIX_RE = re.compile(r"^\s*[\d.]+\s*-\s*")
_PAREN_RE = re.compile(r"\s*\([^)]*\)")
_TRAILING_CATEGORY_RE = re.compile(
    r"\s+(food product|animal feed plant|plant)$", re.IGNORECASE
)


def simplify_label(label: str) -> str:
    """Strip FoodOn label noise (codes, parentheticals, trailing category words).

    Returns the cleaned label; may return the input unchanged. Always safe to
    call — the caller decides whether the simplified form is worth keeping.
    """
    out = _CODE_PREFIX_RE.sub("", label)
    out = _PAREN_RE.sub("", out)
    out = _TRAILING_CATEGORY_RE.sub("", out)
    return out.strip()


class KeywordNER:
    """Deterministic NER that finds any provided keyword inside the text.

    Matches are case-insensitive and word-boundary aware. Useful for tests
    and for cheap end-to-end runs against a known vocabulary. Pair with
    `from_ontology` to surface every FoodOn term that appears in a chunk.

    Produced mentions are typed `entity_type="food"`: the keyword vocabulary is
    food-domain (typically the FoodOn labels themselves), and this keeps the
    mentions linkable past the linker's semantic-type gate.
    """

    model_id = "keyword-ner-v0"

    def __init__(self, keywords: list[str], *, score: float = 1.0) -> None:
        # Sort by length desc so longer matches win when overlaps exist
        # (e.g. "olive oil" preferred over "olive").
        unique = list({k.strip() for k in keywords if k.strip()})
        unique.sort(key=len, reverse=True)
        self._keywords = unique
        self._score = score
        if unique:
            # Build a single regex with word boundaries. Escape every keyword.
            self._regex = re.compile(
                r"\b(" + "|".join(re.escape(k) for k in unique) + r")\b",
                re.IGNORECASE,
            )
        else:
            self._regex = None

    @classmethod
    def from_ontology(
        cls,
        ontology: FoodOnAPI,
        *,
        include_synonyms: bool = True,
        expand_labels: bool = True,
        min_keyword_len: int = 3,
        score: float = 1.0,
    ) -> KeywordNER:
        """Build a keyword NER from every (non-obsolete) ontology term.

        With ``expand_labels`` (default), each label/synonym also contributes a
        `simplify_label`-cleaned variant — FoodOn labels like "red meat (raw)"
        or "legume food product" are over-qualified for prose, so the cleaned
        forms ("red meat", "legume") are what actually match real text.

        ``min_keyword_len`` drops keywords shorter than N characters after
        simplification; a 1-2 char keyword (e.g. FoodOn's "an") generates
        nothing but false positives.
        """
        keywords: list[str] = []
        for term in ontology:
            if term.obsolete:
                continue
            raw = [term.label]
            if include_synonyms:
                raw.extend(term.synonyms)
            for name in raw:
                keywords.append(name)
                if expand_labels:
                    simplified = simplify_label(name)
                    if simplified and simplified.lower() != name.lower():
                        keywords.append(simplified)
        keywords = [k for k in keywords if len(k.strip()) >= min_keyword_len]
        return cls(keywords, score=score)

    def extract(self, text: str) -> list[Mention]:
        if self._regex is None:
            return []
        mentions: list[Mention] = []
        seen: set[tuple[int, int]] = set()
        for m in self._regex.finditer(text):
            span = (m.start(), m.end())
            if span in seen:
                continue
            seen.add(span)
            mentions.append(
                Mention(
                    text=m.group(),
                    start=m.start(),
                    end=m.end(),
                    score=self._score,
                    ner_model_version=self.model_id,
                    # KeywordNER is built `from_ontology`: every keyword IS a
                    # FoodOn term, so "food" is correct — and it keeps these
                    # mentions linkable past the linker's semantic-type gate.
                    entity_type="food",
                )
            )
        return mentions
