"""Three-tier entity linker per BRIEF §2.

Tries in order, first hit wins:
  1. `lexical_exact` — case-insensitive match against label or exact synonym.
  2. `lexical_fuzzy` — rapidfuzz token_set_ratio against all labels + synonyms.
  3. `dense`         — cosine similarity between the mention embedding and
                       precomputed term embeddings, with optional
                       semantic-type gate.

The dense tier is opt-in: pass `dense_embedder=None` (the default) and the
linker degrades gracefully to exact+fuzzy. Useful for fast unit tests and
for running v0.1.0 without SapBERT installed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from foodscholar.io.chunk import EntityLink, Mention
from foodscholar.io.ontology import OntologyId

if TYPE_CHECKING:
    from foodscholar.ontology import FoodOnAPI
    from foodscholar.storage.protocols import Embedder


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class ThreeTierLinker:
    """Reference implementation of the BRIEF §2 linker.

    Construction is heavy (precomputes dense term embeddings if an embedder
    is provided); querying is O(N_terms) in the dense tier and O(N_names) in
    fuzzy. Acceptable for FoodOn (~30k terms after pruning).
    """

    linker_id = "three-tier-v1"

    def __init__(
        self,
        ontology: FoodOnAPI,
        *,
        fuzzy_threshold: float = 0.85,
        dense_threshold: float = 0.78,
        dense_embedder: Embedder | None = None,
        semantic_type_gate: bool = True,
    ) -> None:
        self._ontology = ontology
        self._fuzzy_threshold = fuzzy_threshold
        self._dense_threshold = dense_threshold
        self._embedder = dense_embedder
        self._gate = semantic_type_gate

        # Precompute the (name, id) pairs for fuzzy. Labels + exact synonyms only.
        # We deliberately do not include related synonyms in fuzzy matching: too noisy.
        names: list[tuple[str, OntologyId]] = []
        for term in ontology:
            if term.obsolete:
                continue
            names.append((term.label.lower(), term.id))
            for syn in term.synonyms:
                names.append((syn.lower(), term.id))
        self._fuzzy_pool = names

        # Precompute dense term embeddings if a dense embedder is provided.
        # We embed `label || synonyms` so multi-name terms get a representative vector.
        self._term_ids: list[OntologyId] = []
        self._term_vecs: list[list[float]] = []
        if dense_embedder is not None:
            texts: list[str] = []
            for term in ontology:
                if term.obsolete:
                    continue
                text = " ".join([term.label, *term.synonyms])
                self._term_ids.append(term.id)
                texts.append(text)
            if texts:
                self._term_vecs = dense_embedder.embed(texts)

    # ------------------------------------------------------------------ public API

    def link(self, mention: Mention) -> EntityLink | None:
        text = mention.text.strip()
        if not text:
            return None

        # Tier 1: exact (case-insensitive against label or exact synonym).
        exact_id = self._ontology.name_to_id(text)
        if exact_id is not None:
            return EntityLink(
                mention=mention,
                ontology_id=exact_id,
                confidence=1.0,
                method="lexical_exact",
                linker_version=self.linker_id,
            )

        # Tier 2: fuzzy.
        fuzzy = self._fuzzy_lookup(text)
        if fuzzy is not None:
            term_id, score = fuzzy
            return EntityLink(
                mention=mention,
                ontology_id=term_id,
                confidence=score,
                method="lexical_fuzzy",
                linker_version=self.linker_id,
            )

        # Tier 3: dense.
        if self._embedder is not None and self._term_vecs:
            dense = self._dense_lookup(text)
            if dense is not None:
                term_id, score = dense
                return EntityLink(
                    mention=mention,
                    ontology_id=term_id,
                    confidence=score,
                    method="dense",
                    linker_version=self.linker_id,
                )
        return None

    def dry_run(self, text: str) -> EntityLink | None:
        """Convenience for notebooks: build a Mention from raw text and link it."""
        m = Mention(
            text=text,
            start=0,
            end=len(text),
            score=1.0,
            ner_model_version="dry-run",
        )
        return self.link(m)

    # ------------------------------------------------------------------ tiers

    def _fuzzy_lookup(self, text: str) -> tuple[OntologyId, float] | None:
        from rapidfuzz import fuzz, process

        if not self._fuzzy_pool:
            return None

        query = text.lower()
        # token_set_ratio handles plurals, word-order, and minor edits.
        # process.extractOne returns (matched_name, score_0_to_100, index).
        result = process.extractOne(
            query,
            [n for n, _ in self._fuzzy_pool],
            scorer=fuzz.token_set_ratio,
        )
        if result is None:
            return None
        _, score, idx = result
        normalized = score / 100.0
        if normalized < self._fuzzy_threshold:
            return None
        return self._fuzzy_pool[idx][1], normalized

    def _dense_lookup(self, text: str) -> tuple[OntologyId, float] | None:
        assert self._embedder is not None
        if not self._term_vecs:
            return None
        [q_vec] = self._embedder.embed([text])
        best_id: OntologyId | None = None
        best_score = -1.0
        for term_id, vec in zip(self._term_ids, self._term_vecs, strict=True):
            score = _cosine(q_vec, vec)
            if score > best_score:
                best_score = score
                best_id = term_id
        if best_id is None or best_score < self._dense_threshold:
            return None
        return best_id, best_score
