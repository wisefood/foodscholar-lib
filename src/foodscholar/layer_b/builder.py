"""Layer B orchestration.

Bottom-up the file grows as phases land:

  - Phase 1: `build_shelf_similarity_candidates(chunks, cfg)` (Pass 1)
  - Phase 2: `build_shelf_relatedness_candidates(chunks, cfg)` (Pass 2)
  - Phase 3: `build_shelf_themes(chunks, *, shelf_id, facet, cfg, llm, ...)`
    (full per-shelf pipeline through label + primary picker)
  - Phase 4: `build_layer_b(fs, *, facet, dry_run)` (top-level orchestrator)

The pure-logic graph/community/merge/label modules stay free of I/O; this
module is the only place that reads chunks from the store and writes themes
to it (the persist module handles the write).

Per layer_b_construction_brief.md §6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.layer_b.community import run_leiden
from foodscholar.layer_b.models import ThemeCandidate
from foodscholar.layer_b.relatedness_graph import build_relatedness_graph
from foodscholar.layer_b.semantic_graph import build_similarity_graph

if TYPE_CHECKING:
    from foodscholar.config import LayerBConfig
    from foodscholar.io.chunk import Chunk


def build_shelf_similarity_candidates(
    chunks: list[Chunk],
    cfg: LayerBConfig,
) -> list[ThemeCandidate]:
    """Run Pass 1 (similarity) on a single shelf's chunks.

    Chunks without an embedding are excluded from the graph (and absent
    from every output candidate) — clustering on a biased subsample is
    worse than skipping. The shelf-level embedded-fraction gate runs in
    the top-level orchestrator; this function is robust to mixed input.

    Returns Leiden communities as `ThemeCandidate(pass_name="similarity")`
    records carrying member chunk_ids, an empty `foodon_ids` set (Pass 1
    doesn't read entity_links), and a centroid (mean of L2-normalized
    member vectors — used downstream by the primary picker for similarity
    themes).
    """
    import numpy as np

    embedded = [c for c in chunks if c.embedding is not None]
    if not embedded:
        return []

    embeddings = {
        c.chunk_id: np.asarray(c.embedding, dtype=np.float32) for c in embedded
    }
    g = build_similarity_graph(embedded, embeddings, cfg.similarity)
    communities = run_leiden(g, cfg.leiden)

    if not communities:
        return []

    index_to_id: list[str] = list(g.vs["chunk_id"])
    out: list[ThemeCandidate] = []
    for members in communities:
        chunk_ids = {index_to_id[i] for i in members}
        # Centroid: L2-normalized mean — same metric the kNN graph uses.
        member_vecs = np.stack([embeddings[cid] for cid in chunk_ids])
        norms = np.linalg.norm(member_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = member_vecs / norms
        centroid = normed.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        out.append(
            ThemeCandidate(
                pass_name="similarity",
                chunk_ids=chunk_ids,
                foodon_ids=set(),
                centroid_embedding=centroid.tolist(),
                discovered_by="leiden",
            )
        )
    return out


def build_shelf_relatedness_candidates(
    chunks: list[Chunk],
    cfg: LayerBConfig,
) -> list[ThemeCandidate]:
    """Run Pass 2 (relatedness) on a single shelf's chunks.

    Builds the entity-bridge graph and runs Leiden. Unlike Pass 1, this
    pass does NOT require embeddings — entity coherence can be discovered
    on any chunk whose `entity_links` cleared the linker's confidence
    floor. The candidate's `foodon_ids` is the union of high-confidence
    ontology_ids across its member chunks; this is the entity signature
    the merge step (Phase 3) computes Jaccard against.

    Empty input / no-edges / no-communities all return [] without
    surprising the caller.
    """
    if not chunks:
        return []

    g = build_relatedness_graph(chunks, cfg.relatedness)
    communities = run_leiden(g, cfg.leiden)
    if not communities:
        return []

    index_to_id: list[str] = list(g.vs["chunk_id"])
    chunk_by_id = {c.chunk_id: c for c in chunks}

    out: list[ThemeCandidate] = []
    for members in communities:
        chunk_ids = {index_to_id[i] for i in members}
        foodon_ids: set[str] = set()
        for cid in chunk_ids:
            c = chunk_by_id.get(cid)
            if c is None:
                continue
            foodon_ids |= {
                link.ontology_id
                for link in c.entity_links
                if link.confidence >= cfg.relatedness.tau_strict
            }
        out.append(
            ThemeCandidate(
                pass_name="relatedness",
                chunk_ids=chunk_ids,
                foodon_ids=foodon_ids,
                centroid_embedding=None,  # relatedness pass has no embedding centroid
                discovered_by="leiden",
            )
        )
    return out
