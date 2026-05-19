"""Tri-hybrid OntoRAG retriever — Whoosh + FAISS(MiniLM) + FAISS(SapBERT).

Each arm produces a ranked list of FoodOn term ids for a query; the three
lists are merged by Reciprocal Rank Fusion (RRF) into one ranked
`RetrievedCandidate` list. Retrieval only — the caller (agent / linker)
selects. Adapted from https://github.com/jan3657/onto_rag.

`OntoRagRetriever` holds a loaded `OntoRagIndex` plus the two query-side
embedders. Build the index once with `ontorag.build_index`, then construct a
retriever over it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from foodscholar.io.ontology import OntologyId
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.annotate.ontorag.index import OntoRagIndex
    from foodscholar.ontology import FoodOnAPI
    from foodscholar.storage.protocols import Embedder

_log = get_logger("foodscholar.annotate.ontorag.retriever")

# RRF constant. 60 is the value from the original RRF paper and OntoRAG.
_RRF_K = 60

# Arm priority for deterministic tie-breaking when fusion scores are equal.
_SOURCE_PRIORITY = {"lexical": 0, "sapbert": 1, "minilm": 2}


class RetrievedCandidate(BaseModel):
    """One merged retrieval candidate.

    `fusion_score` is the RRF-combined score (higher = better). `sources`
    lists every arm that surfaced this term; `source` is the highest-priority
    one (used for tie-breaking and payload provenance).
    """

    id: OntologyId
    label: str
    fusion_score: float
    source: str
    sources: list[str]


def _rrf_ranks(ranked_ids: list[OntologyId]) -> dict[OntologyId, float]:
    """RRF contribution of one arm: 1 / (k + rank), rank 0-based."""
    return {tid: 1.0 / (_RRF_K + rank) for rank, tid in enumerate(ranked_ids)}


class OntoRagRetriever:
    """Tri-hybrid retriever over a built `OntoRagIndex`."""

    def __init__(
        self,
        index: OntoRagIndex,
        ontology: FoodOnAPI,
        *,
        minilm: Embedder,
        sapbert: Embedder,
        k_lexical: int = 15,
        k_vector: int = 15,
    ) -> None:
        self._index = index
        self._ontology = ontology
        self._minilm = minilm
        self._sapbert = sapbert
        self._k_lexical = k_lexical
        self._k_vector = k_vector

    def retrieve(self, query: str, *, k: int = 10) -> list[RetrievedCandidate]:
        """Return up to `k` candidates for `query`, RRF-merged across arms."""
        query = query.strip()
        if not query:
            return []

        arms: dict[str, list[OntologyId]] = {
            "lexical": self._lexical(query),
            "minilm": self._dense(query, self._minilm, self._index._faiss_minilm),
            "sapbert": self._dense(query, self._sapbert, self._index._faiss_sapbert),
        }

        # Reciprocal Rank Fusion across the three arms.
        fused: dict[OntologyId, float] = {}
        contributors: dict[OntologyId, list[str]] = {}
        for arm, ranked in arms.items():
            for tid, contrib in _rrf_ranks(ranked).items():
                fused[tid] = fused.get(tid, 0.0) + contrib
                contributors.setdefault(tid, []).append(arm)

        if not fused:
            return []

        def _sort_key(tid: OntologyId) -> tuple[float, int]:
            # Primary: fusion score desc. Tie-break: best contributing arm.
            best_arm = min(contributors[tid], key=lambda a: _SOURCE_PRIORITY[a])
            return (-fused[tid], _SOURCE_PRIORITY[best_arm])

        ranked_ids = sorted(fused, key=_sort_key)[:k]
        out: list[RetrievedCandidate] = []
        for tid in ranked_ids:
            srcs = sorted(contributors[tid], key=lambda a: _SOURCE_PRIORITY[a])
            out.append(
                RetrievedCandidate(
                    id=tid,
                    label=self._ontology.id_to_label(tid) or tid,
                    fusion_score=round(fused[tid], 6),
                    source=srcs[0],
                    sources=srcs,
                )
            )
        return out

    # ------------------------------------------------------------------ arms

    def _lexical(self, query: str) -> list[OntologyId]:
        """Whoosh BM25 arm — returns term ids ranked by BM25 score."""
        from whoosh import index as windex
        from whoosh.qparser import OrGroup, QueryParser

        from foodscholar.annotate.ontorag.index import _F_ID, _F_TEXT

        ix = windex.open_dir(str(self._index.whoosh_dir))
        with ix.searcher() as searcher:
            # OrGroup so a multi-word query matches terms sharing *any* word.
            parser = QueryParser(_F_TEXT, schema=ix.schema, group=OrGroup)
            results = searcher.search(parser.parse(query), limit=self._k_lexical)
            return [r[_F_ID] for r in results]

    def _dense(
        self, query: str, embedder: Embedder, faiss_index: object
    ) -> list[OntologyId]:
        """One FAISS arm — embed the query, kNN, return term ids by rank."""
        import numpy as np

        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "the 'faiss-cpu' package is required for the OntoRAG retriever. "
                "Install with: pip install 'foodscholar[ontorag]'"
            ) from e

        [vec] = embedder.embed([query])
        q = np.asarray([vec], dtype=np.float32)
        faiss.normalize_L2(q)
        _scores, idxs = faiss_index.search(q, self._k_vector)  # type: ignore[attr-defined]
        return [
            self._index.term_ids[i]
            for i in idxs[0]
            if 0 <= i < len(self._index.term_ids)
        ]
