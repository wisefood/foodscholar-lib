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

import re
from collections import Counter
from typing import TYPE_CHECKING

from foodscholar.layer_b.community import run_leiden
from foodscholar.layer_b.label import label_by_keywords, label_by_llm
from foodscholar.layer_b.merge import merge_candidates
from foodscholar.layer_b.models import MergeDecision, ThemeCandidate
from foodscholar.layer_b.primary import pick_primary
from foodscholar.layer_b.relatedness_graph import build_relatedness_graph
from foodscholar.layer_b.semantic_graph import build_similarity_graph

if TYPE_CHECKING:
    from foodscholar.config import LayerBConfig
    from foodscholar.io.chunk import Chunk
    from foodscholar.io.graph import Theme
    from foodscholar.storage.protocols import LLMClient


def _slugify(text: str) -> str:
    """URL-safe lowercase slug, capped at 48 chars. Used in theme_id construction."""
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:48] or "unlabeled"


def _theme_id(facet: str, shelf_id: str, label: str, discovery_pass: str, seq: int) -> str:
    """Deterministic theme id of the form:
    `{facet}/{shelf_slug}/{label_slug}_{pass_initial}{seq}`.

    `seq` is a per-shelf-per-pass counter so two themes with the same label
    in the same shelf get `_s1`/`_s2`/etc. — no silent collision."""
    pass_initial = {"similarity": "s", "relatedness": "r", "merged": "m"}[discovery_pass]
    shelf_slug = shelf_id.split("/")[-1] if "/" in shelf_id else shelf_id
    shelf_slug = _slugify(shelf_slug)
    return f"{facet}/{shelf_slug}/{_slugify(label)}_{pass_initial}{seq}"


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


def build_shelf_themes(
    chunks: list[Chunk],
    *,
    shelf_id: str,
    facet: str,
    cfg: LayerBConfig,
    llm: LLMClient | None,
    config_hash: str,
    version: str,
) -> tuple[list[Theme], list[MergeDecision], dict[str, list[tuple[str, bool, float]]]]:
    """Run the full per-shelf pipeline.

    Returns `(themes, merge_decisions, chunk_assignments)` ready for
    `persist_themes`. The orchestrator (build_layer_b) handles the actual
    persistence so this function stays pure-logic-ish (no store I/O).

    Pipeline:
      1. Pass 1 (similarity) + Pass 2 (relatedness) candidates
      2. Greedy pair-assignment merge → labeled dicts
      3. c-TF-IDF keyword extraction over all themes
      4. Labels: LLM-polished if cfg.labeling.strategy == "llm" and llm is
         not None; otherwise top-keyword fallback
      5. Per-pass-aware primary picker per theme
      6. Construct Pydantic Theme records + chunk_assignments

    Theme IDs are deterministic: `{facet}/{shelf_slug}/{label_slug}_{p}{seq}`.
    """
    from foodscholar.io.graph import Theme

    if not chunks:
        return [], [], {}

    sim_cands = build_shelf_similarity_candidates(chunks, cfg)
    rel_cands = build_shelf_relatedness_candidates(chunks, cfg)
    merged_dicts, decisions = merge_candidates(sim_cands, rel_cands, cfg.merge)

    if not merged_dicts:
        return [], decisions, {}

    chunk_by_id = {c.chunk_id: c for c in chunks}

    # Embeddings dict + per-graph metadata for the primary picker
    import numpy as np

    embeddings: dict[str, np.ndarray] = {}
    for c in chunks:
        if c.embedding is not None:
            embeddings[c.chunk_id] = np.asarray(c.embedding, dtype=np.float32)

    # The relatedness graph is shared across all merged themes' primary
    # picking — build once.
    rel_graph = build_relatedness_graph(chunks, cfg.relatedness)
    # Sim graph isn't used by the picker today (reserved param) — saves a build.
    import igraph as ig
    sim_graph = ig.Graph()

    # Theme-idx → list[Chunk] for labeling
    theme_chunks: dict[int, list[Chunk]] = {
        i: [chunk_by_id[cid] for cid in d["chunk_ids"] if cid in chunk_by_id]
        for i, d in enumerate(merged_dicts)
    }

    # 1. Keyword terms (always computed — cheap, deterministic, fed to LLM
    # as context if labeling.strategy=='llm').
    keywords = label_by_keywords(theme_chunks, cfg.labeling)

    # 2. Labels: LLM polish or keyword-only fallback.
    if cfg.labeling.strategy == "llm" and llm is not None:
        labels = label_by_llm(theme_chunks, keywords, llm, cfg.labeling)
    else:
        # Concat top-k keywords as a free-form label fallback.
        labels = {
            i: " ".join(keywords.get(i, ["unlabeled"])[:3])
            for i in theme_chunks
        }

    themes: list[Theme] = []
    chunk_assignments: dict[str, list[tuple[str, bool, float]]] = {}
    seq_by_pass: dict[str, int] = {"similarity": 0, "relatedness": 0, "merged": 0}

    for i, d in enumerate(merged_dicts):
        pass_kind = d["discovery_pass"]
        seq_by_pass[pass_kind] += 1
        seq = seq_by_pass[pass_kind]
        label = labels.get(i, "unlabeled")
        tid = _theme_id(facet, shelf_id, label, pass_kind, seq)

        # Foodon signature: top-N most-frequent high-conf entities in this theme
        ent_counter: Counter[str] = Counter()
        for cid in d["chunk_ids"]:
            c = chunk_by_id.get(cid)
            if c is None:
                continue
            for link in c.entity_links:
                if link.confidence >= cfg.relatedness.tau_strict:
                    ent_counter[link.ontology_id] += 1
        signature = [oid for oid, _ in ent_counter.most_common(10)]

        # Pick primary chunk for this theme.
        # For sim/merged we need a centroid — recover from the source candidate.
        centroid = None
        if pass_kind in ("similarity", "merged"):
            # Find the source sim candidate (by chunk-id subset overlap).
            for sc in sim_cands:
                if sc.chunk_ids & d["chunk_ids"]:
                    centroid = sc.centroid_embedding
                    break
        primary_chunk = pick_primary(
            chunk_ids=set(d["chunk_ids"]),
            discovery_pass=pass_kind,
            embeddings=embeddings,
            centroid=centroid,
            sim_graph=sim_graph,
            rel_graph=rel_graph,
        )

        themes.append(
            Theme(
                theme_id=tid,
                label=label,
                shelf_ids=[shelf_id],
                chunk_count=len(d["chunk_ids"]),
                discovered_by=d["discovered_by"],
                discovery_version=version,
                facet=facet,  # type: ignore[arg-type]
                discovery_pass=pass_kind,  # type: ignore[arg-type]
                keyword_terms=list(keywords.get(i, [])),
                foodon_id_signature=signature,
                config_hash=config_hash,
                version=version,
            )
        )

        # Per-chunk assignments: primary=True for one chunk; weight=1.0 for
        # primary, 0.5 for others (placeholder — refined in v2 to use actual
        # centroid-cosine / edge-degree score from the picker).
        chunk_assignments[tid] = [
            (cid, cid == primary_chunk, 1.0 if cid == primary_chunk else 0.5)
            for cid in sorted(d["chunk_ids"])
        ]

    return themes, decisions, chunk_assignments
