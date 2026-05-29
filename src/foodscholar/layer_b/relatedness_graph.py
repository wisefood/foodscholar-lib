"""Pass 2 — relatedness graph from shared FoodOn entity IDs.

Edge weight = sum over shared IDs of `1 / log(1 + doc_freq[id])`. Inspired
by SiReRAG and TF-IDF: rare entities count more than ubiquitous ones. Per
`layer_b_construction_brief.md` §4.2.

The "three knobs that make or break Pass 2" (per brief):
  - `tau_strict` — entity-link confidence floor for participation
  - `min_shared_ids` — minimum shared entities to form any edge
  - `max_doc_frequency` — drop entities that appear in too many of the
    shelf's chunks (no discriminative signal)

`always_exclude_iris` is a permanent kill-list: ontology classes that
survived Layer A but ancestor-propagate onto nearly every chunk (the
default is the FOODON:00001002 'food product' umbrella). These are never
edge-creators regardless of doc_freq.

python-igraph is lazy-imported (gated by the `[clustering]` extra).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foodscholar.config import RelatednessConfig
    from foodscholar.io.chunk import Chunk


def build_relatedness_graph(
    chunks: list[Chunk],
    cfg: RelatednessConfig,
) -> Any:
    """Return an undirected weighted igraph; vertices carry `chunk_id`."""
    import igraph as ig

    g = ig.Graph()
    if not chunks:
        return g

    # 1. High-confidence entity sets per chunk
    chunk_entities: dict[str, set[str]] = {}
    for c in chunks:
        ids = {
            link.ontology_id
            for link in c.entity_links
            if link.confidence >= cfg.tau_strict
        }
        chunk_entities[c.chunk_id] = ids

    # 2. Document frequency per entity (over this shelf's chunks)
    n_chunks = len(chunks)
    doc_freq: Counter[str] = Counter()
    for ents in chunk_entities.values():
        for e in ents:
            doc_freq[e] += 1

    # 3. Excluded set: ubiquitous entities + the permanent kill-list
    excluded = {
        e for e, f in doc_freq.items() if f / n_chunks > cfg.max_doc_frequency
    }
    excluded |= set(cfg.always_exclude_iris)

    # 4. Edges
    chunk_ids = [c.chunk_id for c in chunks]
    edges: list[tuple[int, int, float]] = []
    for i in range(len(chunk_ids)):
        for j in range(i + 1, len(chunk_ids)):
            shared = (
                chunk_entities[chunk_ids[i]] & chunk_entities[chunk_ids[j]]
            ) - excluded
            if len(shared) < cfg.min_shared_ids:
                continue
            w = sum(1.0 / math.log(1 + doc_freq[e]) for e in shared)
            edges.append((i, j, w))

    g.add_vertices(len(chunk_ids))
    g.vs["chunk_id"] = chunk_ids
    if edges:
        g.add_edges([(i, j) for i, j, _ in edges])
        g.es["weight"] = [w for _, _, w in edges]
    return g
