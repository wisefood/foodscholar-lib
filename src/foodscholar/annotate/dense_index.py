"""Dense nearest-neighbour index over ontology term embeddings.

Backs the linker's `dense` tier. Embeds every (non-obsolete) ontology term
once — `label` plus its synonyms joined — L2-normalizes the vectors, and
stacks them into a single matrix. A query is then one matrix-vector product:
cosine similarity against ~29k FoodOn terms is well under 2ms in numpy, so no
FAISS dependency is needed at this scale.

The matrix is cached to a `.npz` file keyed on the embedder's `model_id` plus
a fingerprint of the term set, so rebuilding only happens when the ontology
or the embedding model changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foodscholar.ontology import FoodOnAPI
    from foodscholar.storage.protocols import Embedder

from foodscholar.io.ontology import OntologyId
from foodscholar.logging import get_logger

_log = get_logger("foodscholar.annotate.dense")


def _term_text(label: str, synonyms: tuple[str, ...]) -> str:
    """Representative text for a term: label first, then synonyms."""
    return " ".join([label, *synonyms])


def _fingerprint(ids: list[OntologyId], model_id: str, dim: int) -> str:
    """Stable hash of (term-id set, embedder id, embedding dim) — the cache key.

    `dim` is included so two embedders sharing a `model_id` but producing
    different-width vectors don't collide on the same cache entry.
    """
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(f"|dim={dim}".encode())
    for tid in sorted(ids):
        h.update(b"\x00")
        h.update(tid.encode("utf-8"))
    return h.hexdigest()[:16]


class DenseIndex:
    """In-memory cosine kNN over ontology term embeddings.

    Build with `DenseIndex.build(ontology, embedder)`. The build embeds every
    term and is the expensive step; `query` is cheap. Use `build(..., cache_path=...)`
    to persist/reuse the embedding matrix across runs.
    """

    def __init__(self, term_ids: list[OntologyId], matrix: object) -> None:
        # `matrix` is an (n_terms, dim) float32 numpy array, L2-normalized per row.
        self._term_ids = term_ids
        self._matrix = matrix

    @property
    def size(self) -> int:
        return len(self._term_ids)

    @classmethod
    def build(
        cls,
        ontology: FoodOnAPI,
        embedder: Embedder,
        *,
        cache_path: str | Path | None = None,
    ) -> DenseIndex:
        import numpy as np

        terms = [t for t in ontology if not t.obsolete]
        term_ids = [t.id for t in terms]
        fp = _fingerprint(term_ids, embedder.model_id, embedder.dim)

        cache = Path(cache_path) if cache_path else None
        if cache is not None and cache.exists():
            data = np.load(cache, allow_pickle=False)
            if str(data["fingerprint"]) == fp:
                _log.info("dense_index.cache_hit", path=str(cache), n_terms=len(term_ids))
                # term_ids stored as a newline-joined string to keep the .npz
                # free of object arrays (which would force allow_pickle=True).
                cached_ids = str(data["term_ids"]).split("\n") if str(data["term_ids"]) else []
                return cls(cached_ids, data["matrix"])
            _log.info("dense_index.cache_stale", path=str(cache))

        _log.info("dense_index.building", n_terms=len(term_ids))
        texts = [_term_text(t.label, t.synonyms) for t in terms]
        vectors = embedder.embed(texts)
        matrix = _l2_normalize(np.asarray(vectors, dtype=np.float32), dim_hint=embedder.dim)

        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache,
                term_ids=np.array("\n".join(term_ids)),
                matrix=matrix,
                fingerprint=np.array(fp),
            )
            _log.info("dense_index.cached", path=str(cache))

        return cls(term_ids, matrix)

    def query(self, vector: list[float], *, k: int = 5) -> list[tuple[OntologyId, float]]:
        """Return the top-k (term_id, cosine_similarity) for a query vector."""
        import numpy as np

        if not self._term_ids:
            return []
        q = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            return []
        q = q / norm
        sims = self._matrix @ q  # (n_terms,) cosine — matrix rows are pre-normalized
        k = min(k, len(self._term_ids))
        # argpartition for top-k, then sort just those k.
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(self._term_ids[i], float(sims[i])) for i in top_idx]


def _l2_normalize(matrix: object, *, dim_hint: int = 0) -> object:
    import numpy as np

    m = np.asarray(matrix, dtype=np.float32)
    if m.size == 0:
        # Empty ontology — return a well-shaped (0, dim) array so downstream
        # matrix ops don't hit an axis error.
        return m.reshape(0, dim_hint or 1)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return m / norms
