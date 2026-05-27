"""Per-pass-aware primary-chunk picker for Layer B themes.

The `primary` flag on `(:Chunk)-[:THEME_OF {primary: bool}]->(:Theme)` is
used by retrieval ranking — it marks the "most representative" chunk of
the theme within a given shelf. v1 picks it per pass:

  - similarity themes: closest-to-centroid in embedding space (max cosine
    against the theme's mean-normalized centroid). Uses the centroid the
    similarity-pass builder stamped on the candidate.
  - relatedness themes: max sum-of-edge-weights to other members within
    the relatedness graph. The hub of the theme's induced subgraph.
  - merged themes: max(centroid-score, edge-degree-score) per chunk —
    take the chunk with the highest of those two maxima. Works whether
    the merged theme is more similarity-anchored or more entity-anchored.

Ties are broken by lex-first chunk_id, giving deterministic primaries
across runs (the audit-parity contract).

Per `layer_b_construction_brief.md` §14's "open decisions: primary picker"
and the Plan-agent review (lex-first alone leaks no signal — per-pass-aware
+ lex-first as tie-breaker is the v1 default).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def _centroid_scores(
    chunk_ids: list[str],
    embeddings: dict[str, np.ndarray],
    centroid: list[float] | None,
) -> dict[str, float]:
    """Cosine of each chunk's normalized vector against the centroid.
    Returns {} if the centroid isn't available."""
    if centroid is None:
        return {}
    import numpy as np

    c = np.asarray(centroid, dtype=np.float32)
    cn = np.linalg.norm(c)
    if cn == 0:
        return {}
    c = c / cn
    out: dict[str, float] = {}
    for cid in chunk_ids:
        v = embeddings.get(cid)
        if v is None:
            continue
        v = np.asarray(v, dtype=np.float32)
        vn = np.linalg.norm(v)
        if vn == 0:
            continue
        out[cid] = float((v / vn) @ c)
    return out


def _edge_degree_scores(
    chunk_ids: set[str],
    rel_graph: Any,
) -> dict[str, float]:
    """Sum-of-edge-weights to other members of the theme within rel_graph.

    Vertices outside `chunk_ids` are ignored (the rel_graph spans the
    whole shelf, but the score is theme-local)."""
    if rel_graph.vcount() == 0 or rel_graph.ecount() == 0:
        return {cid: 0.0 for cid in chunk_ids}
    cid_to_idx: dict[str, int] = {
        c: i for i, c in enumerate(rel_graph.vs["chunk_id"])
    }
    member_indices = {cid_to_idx[c] for c in chunk_ids if c in cid_to_idx}
    out: dict[str, float] = {cid: 0.0 for cid in chunk_ids}
    for edge in rel_graph.es:
        s, t = edge.source, edge.target
        if s in member_indices and t in member_indices:
            w = float(edge["weight"]) if "weight" in edge.attributes() else 1.0
            cs = rel_graph.vs[s]["chunk_id"]
            ct = rel_graph.vs[t]["chunk_id"]
            out[cs] = out.get(cs, 0.0) + w
            out[ct] = out.get(ct, 0.0) + w
    return out


def pick_primary(
    chunk_ids: set[str],
    discovery_pass: str,
    embeddings: dict[str, np.ndarray],
    centroid: list[float] | None,
    sim_graph: Any,  # reserved for future per-pass scorers
    rel_graph: Any,
) -> str:
    """Return the primary chunk_id for one theme.

    Deterministic: ties on the per-pass score break by lex-first chunk_id,
    so the same `(chunk_ids, embeddings, graphs)` always yields the same
    primary across runs. That's the audit-parity contract.
    """
    if not chunk_ids:
        raise ValueError("pick_primary requires at least one chunk")
    if len(chunk_ids) == 1:
        return next(iter(chunk_ids))

    ids_lex = sorted(chunk_ids)

    if discovery_pass == "global_similarity":
        scores = _centroid_scores(ids_lex, embeddings, centroid)
    elif discovery_pass == "relatedness":
        scores = _edge_degree_scores(chunk_ids, rel_graph)
    elif discovery_pass == "merged":
        cs = _centroid_scores(ids_lex, embeddings, centroid)
        es = _edge_degree_scores(chunk_ids, rel_graph)
        scores = {cid: max(cs.get(cid, 0.0), es.get(cid, 0.0)) for cid in ids_lex}
    else:
        raise ValueError(f"unknown discovery_pass: {discovery_pass!r}")

    # Sort: highest score first, lex-first chunk_id breaks ties.
    ranked = sorted(
        ids_lex,
        key=lambda cid: (-scores.get(cid, 0.0), cid),
    )
    return ranked[0]
