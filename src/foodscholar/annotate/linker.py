"""Tiered entity linker (BRIEF §2, extended per §3.5).

Tries in order, first confident hit wins:
  1. `lexical_exact`  — case-insensitive, punctuation-insensitive match
                        against a label or exact synonym.
  2. `lexical_fuzzy`  — rapidfuzz `WRatio` against all labels + synonyms.
  3. `dense`          — cosine kNN between the mention embedding and
                        precomputed term embeddings (via `DenseIndex`).
  4. `llm`            — *opt-in.* When the lexical/dense tiers don't produce a
                        confident hit, an LLM is shown the top-k candidates
                        (with mention context) and picks one — or rejects all.

Tiers 1-3 are deterministic. Tier 4 is gated behind `llm_client` being
provided AND a confidence threshold, so it fires only on the hard residue —
it is NOT a per-mention LLM call. This is a documented deviation from the
literal BRIEF §2 ("lexical then dense"); see BRIEF §3.5.

Disabling tiers: pass `dense_embedder=None` (default) to skip tier 3, and
`llm_client=None` (default) to skip tier 4. With both off the linker is the
pure-lexical v0.1 behaviour.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from foodscholar.annotate.dense_index import DenseIndex
from foodscholar.io.chunk import EntityLink, Mention
from foodscholar.io.ontology import OntologyId
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.ontology import FoodOnAPI
    from foodscholar.storage.protocols import Embedder, LLMClient

_log = get_logger("foodscholar.annotate.linker")

# Parses the LLM selector's reply: a candidate index, or "none".
_LLM_CHOICE_RE = re.compile(r"\b(\d+|none)\b", re.IGNORECASE)


class ThreeTierLinker:
    """Reference linker implementation.

    Despite the name (kept for import stability), this is a 3-or-4 tier linker
    depending on which optional backends are supplied. Construction is heavy
    when a dense embedder is given (it builds the `DenseIndex`); querying is
    cheap.
    """

    linker_id = "tiered-linker-v2"

    def __init__(
        self,
        ontology: FoodOnAPI,
        *,
        fuzzy_threshold: float = 0.85,
        dense_threshold: float = 0.78,
        dense_embedder: Embedder | None = None,
        dense_cache_path: str | None = None,
        semantic_type_gate: bool = True,
        llm_client: LLMClient | None = None,
        llm_select_threshold: float = 0.90,
        llm_candidate_k: int = 8,
    ) -> None:
        self._ontology = ontology
        self._fuzzy_threshold = fuzzy_threshold
        self._dense_threshold = dense_threshold
        self._embedder = dense_embedder
        self._gate = semantic_type_gate
        self._llm = llm_client
        self._llm_select_threshold = llm_select_threshold
        self._llm_candidate_k = llm_candidate_k

        # Fuzzy pool: (name, id, kind) for labels + exact synonyms.
        # `kind` ∈ {"label", "synonym"} — used to break score ties.
        names: list[tuple[str, OntologyId, str]] = []
        for term in ontology:
            if term.obsolete:
                continue
            names.append((term.label.lower(), term.id, "label"))
            for syn in term.synonyms:
                names.append((syn.lower(), term.id, "synonym"))
        self._fuzzy_pool = names

        # Dense index — only built when a dense embedder is supplied.
        self._dense_index: DenseIndex | None = None
        if dense_embedder is not None:
            self._dense_index = DenseIndex.build(
                ontology, dense_embedder, cache_path=dense_cache_path
            )

    # ------------------------------------------------------------------ public API

    def link(self, mention: Mention) -> EntityLink | None:
        text = mention.text.strip()
        if not text:
            return None

        # Tier 1: exact.
        exact_id = self._ontology.name_to_id(text)
        if exact_id is not None:
            return self._link(mention, exact_id, 1.0, "lexical_exact")

        # Tier 2: fuzzy.
        fuzzy = self._fuzzy_lookup(text)

        # Tier 3: dense. Skipped when fuzzy already cleared the confidence
        # bar — no point paying an embed call to confirm a strong fuzzy hit.
        dense = None
        fuzzy_is_confident = fuzzy is not None and fuzzy[1] >= self._llm_select_threshold
        if self._dense_index is not None and not fuzzy_is_confident:
            dense = self._dense_lookup(text)

        # Take the better of fuzzy / dense as the "best deterministic" hit.
        best = self._best_of(fuzzy, dense)
        if best is not None:
            term_id, score, method = best
            # Tier 4: if the best deterministic hit is below the LLM-select
            # threshold (or there's no hit at all), let the LLM adjudicate.
            if self._llm is not None and score < self._llm_select_threshold:
                llm = self._llm_select(text)
                if llm is not None:
                    return self._link(mention, llm[0], llm[1], "llm")
            return self._link(mention, term_id, score, method)

        # No deterministic hit at all — last-chance LLM selection.
        if self._llm is not None:
            llm = self._llm_select(text)
            if llm is not None:
                return self._link(mention, llm[0], llm[1], "llm")
        return None

    def dry_run(self, text: str) -> EntityLink | None:
        """Convenience for notebooks: build a Mention from raw text and link it."""
        return self.link(
            Mention(text=text, start=0, end=len(text), score=1.0, ner_model_version="dry-run")
        )

    # ------------------------------------------------------------------ helpers

    def _link(
        self, mention: Mention, term_id: OntologyId, score: float, method: str
    ) -> EntityLink:
        return EntityLink(
            mention=mention,
            ontology_id=term_id,
            confidence=score,
            method=method,  # type: ignore[arg-type]
            linker_version=self.linker_id,
        )

    @staticmethod
    def _best_of(
        fuzzy: tuple[OntologyId, float] | None,
        dense: tuple[OntologyId, float] | None,
    ) -> tuple[OntologyId, float, str] | None:
        cands: list[tuple[OntologyId, float, str]] = []
        if fuzzy is not None:
            cands.append((fuzzy[0], fuzzy[1], "lexical_fuzzy"))
        if dense is not None:
            cands.append((dense[0], dense[1], "dense"))
        if not cands:
            return None
        return max(cands, key=lambda c: c[1])

    # ------------------------------------------------------------------ tiers

    def _fuzzy_lookup(self, text: str) -> tuple[OntologyId, float] | None:
        """Fuzzy match against labels + exact synonyms.

        Uses `WRatio` (penalizes length mismatch) rather than `token_set_ratio`
        (which doesn't, and on a 30k+ ontology prefers single-word generic
        terms like "oil" over "olive oil"). Short queries (≤4 chars) require a
        stricter 0.95 threshold; a length-ratio gate rejects matches far
        shorter than the query; ties break label > synonym then closest length.
        """
        from rapidfuzz import fuzz, process

        if not self._fuzzy_pool:
            return None

        query = text.lower()
        names = [n for n, _, _ in self._fuzzy_pool]
        results = process.extract(query, names, scorer=fuzz.WRatio, limit=10)
        if not results:
            return None

        threshold = max(self._fuzzy_threshold, 0.95) if len(query) <= 4 else self._fuzzy_threshold

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

        top_score = candidates[0][0]
        tied = [c for c in candidates if (top_score - c[0]) <= 0.005]
        tied.sort(key=lambda c: (0 if c[3] == "label" else 1, abs(len(c[1]) - q_len)))
        best_score, _, best_id, _ = tied[0]
        return best_id, best_score

    def _dense_lookup(self, text: str) -> tuple[OntologyId, float] | None:
        assert self._embedder is not None and self._dense_index is not None
        [q_vec] = self._embedder.embed([text])
        hits = self._dense_index.query(q_vec, k=1)
        if not hits:
            return None
        term_id, score = hits[0]
        if score < self._dense_threshold:
            return None
        return term_id, score

    def _llm_select(self, text: str) -> tuple[OntologyId, float] | None:
        """Tier 4: show the LLM the top-k candidates, let it pick one or reject.

        Candidates come from the dense index (if available) else the fuzzy
        pool. The LLM sees the mention plus a numbered candidate list and
        replies with an index or "none". A successful pick is returned with a
        fixed `0.85` confidence — it's a model judgement, not a similarity
        score.
        """
        assert self._llm is not None
        candidates = self._llm_candidates(text)
        if not candidates:
            return None

        listing = "\n".join(
            f"{i}. {self._ontology.id_to_label(tid)}  [{tid}]"
            for i, tid in enumerate(candidates)
        )
        prompt = (
            "You are linking a food mention to a FoodOn ontology term.\n"
            f'Mention: "{text}"\n\n'
            "Candidates:\n"
            f"{listing}\n\n"
            "Reply with ONLY the number of the single best matching candidate, "
            'or the word "none" if no candidate is a correct match for the '
            "mention (for example if the mention is not a food).\n"
            "Answer:"
        )
        try:
            reply = self._llm.generate(prompt, max_tokens=8)
        except Exception as e:
            # Broad catch is deliberate: an LLM backend failure must degrade
            # the linker to "no llm hit", never crash the whole annotate phase.
            _log.warning("linker.llm_select_failed", error=str(e))
            return None

        m = _LLM_CHOICE_RE.search(reply or "")
        if m is None:
            return None
        token = m.group(1).lower()
        if token == "none":
            return None
        idx = int(token)
        if 0 <= idx < len(candidates):
            return candidates[idx], 0.85
        return None

    def _llm_candidates(self, text: str) -> list[OntologyId]:
        """Top-k candidate ids for the LLM selector — dense first, fuzzy fallback."""
        k = self._llm_candidate_k
        if self._embedder is not None and self._dense_index is not None:
            [q_vec] = self._embedder.embed([text])
            return [tid for tid, _ in self._dense_index.query(q_vec, k=k)]

        from rapidfuzz import fuzz, process

        if not self._fuzzy_pool:
            return []
        names = [n for n, _, _ in self._fuzzy_pool]
        results = process.extract(text.lower(), names, scorer=fuzz.WRatio, limit=k)
        seen: list[OntologyId] = []
        for _, _, idx in results:
            tid = self._fuzzy_pool[idx][1]
            if tid not in seen:
                seen.append(tid)
        return seen
