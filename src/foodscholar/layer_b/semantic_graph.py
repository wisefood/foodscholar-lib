"""Pass 1 — similarity graph over chunk embeddings.

Mutual-kNN weighted undirected graph: each chunk has an edge to its top-k
cosine neighbors, filtered by `edge_threshold` and (optionally) by the
mutual-neighbor requirement. Per `layer_b_construction_brief.md` §4.1.

Embeddings flow in already L2-normalized (the BGE/SPECTER2 router returns
normalized vectors), so cosine similarity = dot product. The implementation
re-normalizes defensively so an unnormalized input doesn't silently produce
wrong weights.

Numpy + python-igraph are lazy-imported — gated by the `[clustering]`
extra. Module import is cheap and doesn't depend on the numeric stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from foodscholar.config import SimilarityConfig
    from foodscholar.io.chunk import Chunk, ChunkId


def build_similarity_graph(
    chunks: list[Chunk],
    embeddings: dict[ChunkId, np.ndarray],
    cfg: SimilarityConfig,
) -> Any:
    """Build an undirected weighted igraph from chunk embeddings.

    Vertices carry the `chunk_id` attribute; edges carry `weight = cosine`.
    Returns an empty graph if `chunks` is empty. Vertices with no surviving
    neighbors stay in the graph as isolates — Leiden's `min_community_size`
    filter drops them downstream.
    """
    import igraph as ig
    import numpy as np

    g = ig.Graph()
    if not chunks:
        return g

    ids: list[str] = [c.chunk_id for c in chunks]

    g.add_vertices(len(ids))
    g.vs["chunk_id"] = ids

    n = len(ids)
    if n < 2:
        # Single-chunk shelves can't form edges; bail before the kNN math.
        return g

    M = np.stack([np.asarray(embeddings[cid], dtype=np.float32) for cid in ids])
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    M = M / norms

    sims = M @ M.T
    np.fill_diagonal(sims, -1.0)

    k = min(cfg.knn_k, n - 1)
    # argpartition picks the top-k unordered indices per row — cheaper than a
    # full argsort. We filter by edge_threshold below, so the within-top-k
    # order doesn't matter.
    topk_idx: list[set[int]] = []
    for i in range(n):
        row = sims[i]
        idx = np.argpartition(-row, k - 1)[:k]
        topk_idx.append(set(int(j) for j in idx))

    edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in topk_idx[i]:
            if i == j:
                continue
            w = float(sims[i, j])
            if w < cfg.edge_threshold:
                continue
            if cfg.require_mutual and i not in topk_idx[j]:
                continue
            key = (i, j) if i < j else (j, i)
            # Dict assignment is idempotent — same edge from i→j and j→i
            # collapses to one (i<j) entry with the same weight.
            edges[key] = w

    if edges:
        edge_list = sorted(edges)
        g.add_edges(edge_list)
        g.es["weight"] = [edges[e] for e in edge_list]
    return g
