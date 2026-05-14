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

        # Precompute the (name, id, kind) pairs for fuzzy. Labels + exact synonyms.
        # `kind` is "label" or "synonym" — used to break ties in favor of labels.
        # Related synonyms are deliberately excluded: too noisy.
        names: list[tuple[str, OntologyId, str]] = []
        for term in ontology:
            if term.obsolete:
                continue
            names.append((term.label.lower(), term.id, "label"))
            for syn in term.synonyms:
                names.append((syn.lower(), term.id, "synonym"))
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
        """Fuzzy match against labels + exact synonyms.

        Uses `WRatio` (which penalizes length differences) rather than
        `token_set_ratio` (which doesn't, and on a 30k+ ontology routinely
        prefers single-word generic terms like "oil" over multi-word
        specifics like "olive oil").

        Among ties (within 1 point), prefer label matches over synonym
        matches, then prefer the shortest target name (closest length to
        the query). This stops "arachis" from matching "peanut oil" when
        "Arachis hypogaea" is also in the pool.

        Hard gate: reject any match where the target's character length is
        less than half the query's, even if the score is high — that's the
        "oliv oil" → "oil" failure mode in disguise.
        """
        from rapidfuzz import fuzz, process

        if not self._fuzzy_pool:
            return None

        query = text.lower()
        names = [n for n, _, _ in self._fuzzy_pool]
        results = process.extract(query, names, scorer=fuzz.WRatio, limit=10)
        if not results:
            return None

        # Short queries (≤4 chars) demand a much higher fuzzy threshold —
        # WRatio finds spurious matches against any random short label
        # otherwise. e.g. "evo" → "devonshire cream" at 0.90 with default
        # threshold; bump to 0.95 and it's correctly rejected.
        threshold = max(self._fuzzy_threshold, 0.95) if len(query) <= 4 else self._fuzzy_threshold

        # Filter by threshold and length-ratio gate.
        q_len = max(len(query), 1)
        candidates: list[tuple[float, str, OntologyId, str]] = []
        for matched, score, idx in results:
            normalized = score / 100.0
            if normalized < threshold:
                continue
            if len(matched) / q_len < 0.5:
                continue
            _, term_id, kind = self._fuzzy_pool[idx]
            candidates.append((normalized, matched, term_id, kind))

        if not candidates:
            return None

        # Tie-break (within 0.5pt only — WRatio scores are coarse but not THAT
        # coarse, and a wider window starves real top hits in favor of shorter
        # nearby labels). Within the window prefer label > synonym, then the
        # match whose length is closest to the query length.
        top_score = candidates[0][0]
        tied = [c for c in candidates if (top_score - c[0]) <= 0.005]
        tied.sort(key=lambda c: (0 if c[3] == "label" else 1, abs(len(c[1]) - q_len)))
        best_score, _, best_id, _ = tied[0]
        return best_id, best_score

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
