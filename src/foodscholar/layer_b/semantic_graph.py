"""Pass 1 — similarity graph over chunk embeddings.

Mutual-kNN weighted undirected graph: each chunk has an edge to its top-k
cosine neighbors, filtered by `edge_threshold` and (optionally) by the
mutual-neighbor requirement. Per `layer_b_construction_brief.md` §4.1.

Embeddings flow in already L2-normalized (BGE-base via `HFEmbedder` returns
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


def build_global_similarity_graph(
    chunk_ids: list[ChunkId],
    chunk_store: Any,  # ChunkStore protocol
    cfg: SimilarityConfig,
) -> Any:
    """Build a similarity graph across `chunk_ids` using ChunkStore.knn_search_chunks.

    Unlike `build_similarity_graph` (per-shelf, in-memory all-pairs), this fans
    out one kNN call per chunk against the chunk store's HNSW index — meant for
    ~thousands of chunks where O(n^2) would blow up.

    Output shape matches `build_similarity_graph`: vertices carry `chunk_id`,
    edges carry `weight = cosine`. Empty input → empty graph. The kNN is
    restricted to `candidate_ids=chunk_ids` so the global graph stays inside
    the attached corpus even when the underlying store has more chunks.
    """
    import igraph as ig

    g = ig.Graph()
    if not chunk_ids:
        return g

    g.add_vertices(len(chunk_ids))
    g.vs["chunk_id"] = chunk_ids

    if len(chunk_ids) < 2:
        return g

    chunks = chunk_store.get_many(chunk_ids)
    qvecs: dict[ChunkId, list[float]] = {
        c.chunk_id: c.embedding for c in chunks if c.embedding is not None
    }

    chunk_id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
    candidate_set = list(chunk_ids)

    neighbors: dict[ChunkId, set[ChunkId]] = {}
    edge_weights: dict[tuple[int, int], float] = {}
    for cid in chunk_ids:
        qvec = qvecs.get(cid)
        if qvec is None:
            neighbors[cid] = set()
            continue
        hits = chunk_store.knn_search_chunks(
            query_vector=qvec,
            k=cfg.knn_k,
            exclude_ids=[cid],
            candidate_ids=candidate_set,
        )
        neighbors[cid] = {nid for nid, _ in hits}
        for nid, score in hits:
            if score < cfg.edge_threshold:
                continue
            if nid not in chunk_id_to_idx:
                continue
            i, j = chunk_id_to_idx[cid], chunk_id_to_idx[nid]
            key = (i, j) if i < j else (j, i)
            prev = edge_weights.get(key)
            if prev is None or score > prev:
                edge_weights[key] = score

    if cfg.require_mutual:
        edge_weights = {
            (i, j): w
            for (i, j), w in edge_weights.items()
            if chunk_ids[j] in neighbors.get(chunk_ids[i], set())
            and chunk_ids[i] in neighbors.get(chunk_ids[j], set())
        }

    if edge_weights:
        edge_list = sorted(edge_weights)
        g.add_edges(edge_list)
        g.es["weight"] = [edge_weights[e] for e in edge_list]
    return g
