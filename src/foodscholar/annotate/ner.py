"""NER adapters.

Two implementations:
  - `KeywordNER` — deterministic, dependency-free. Surfaces any term in a
    provided dictionary that appears in the text. Used by tests and as a
    sensible default for the in-memory facade.
  - `SciFoodNERAdapter` — wraps a HuggingFace token-classification pipeline
    (SciFoodNER per BRIEF §2). Lazy-imports transformers so the core
    package stays slim. Gated by the `[annotate]` extra.

Both implement the `NER` protocol from `foodscholar.storage.protocols`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from foodscholar.io.chunk import Mention

if TYPE_CHECKING:
    from foodscholar.ontology import FoodOnAPI


class KeywordNER:
    """Deterministic NER that finds any provided keyword inside the text.

    Matches are case-insensitive and word-boundary aware. Useful for tests
    and for cheap end-to-end runs against a known vocabulary. Pair with
    `from_ontology` to surface every FoodOn term that appears in a chunk.
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
        score: float = 1.0,
    ) -> KeywordNER:
        """Build a keyword NER from every (non-obsolete) ontology term."""
        keywords: list[str] = []
        for term in ontology:
            if term.obsolete:
                continue
            keywords.append(term.label)
            if include_synonyms:
                keywords.extend(term.synonyms)
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
                )
            )
        return mentions


class SciFoodNERAdapter:
    """HuggingFace SciFoodNER pipeline adapter (BRIEF §2).

    Lazy-imports `transformers`. Construction loads the model and tokenizer.
    For unit tests use `KeywordNER` instead — this class is gated by
    `pytest -m slow` because it downloads ~500MB on first run.
    """

    def __init__(self, model_name: str = "Maouriyan/Sci_food_NER") -> None:
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as e:
            raise ImportError(
                "the 'transformers' package is required for SciFoodNERAdapter. "
                "Install with: pip install 'foodscholar[annotate]'"
            ) from e

        self.model_id = model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        self._pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )

    def extract(self, text: str) -> list[Mention]:
        results = self._pipeline(text)
        return [
            Mention(
                text=str(r["word"]),
                start=int(r["start"]),
                end=int(r["end"]),
                score=float(r["score"]),
                ner_model_version=self.model_id,
            )
            for r in results
        ]
