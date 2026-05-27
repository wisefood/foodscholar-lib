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
from datetime import UTC
from typing import TYPE_CHECKING, Any

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
    pass_initial = {"similarity": "s", "relatedness": "r", "merged": "m", "global_similarity": "g"}[discovery_pass]
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


def build_global_similarity_candidates(
    chunk_ids: list[str],
    chunk_store: Any,
    cfg: LayerBConfig,
) -> list[ThemeCandidate]:
    """Run Pass 1 (similarity) across the WHOLE attached corpus."""
    import numpy as np

    from foodscholar.layer_b.community import run_leiden
    from foodscholar.layer_b.semantic_graph import build_global_similarity_graph

    if not chunk_ids:
        return []

    g = build_global_similarity_graph(chunk_ids, chunk_store, cfg.similarity)
    communities = run_leiden(g, cfg.leiden)
    if not communities:
        return []

    chunks = chunk_store.get_many(chunk_ids)
    embeddings: dict[str, np.ndarray] = {
        c.chunk_id: np.asarray(c.embedding, dtype=np.float32)
        for c in chunks
        if c.embedding is not None
    }

    index_to_id: list[str] = list(g.vs["chunk_id"])
    out: list[ThemeCandidate] = []
    for members in communities:
        member_ids = {index_to_id[i] for i in members if index_to_id[i] in embeddings}
        if not member_ids:
            continue
        member_vecs = np.stack([embeddings[cid] for cid in member_ids])
        norms = np.linalg.norm(member_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = member_vecs / norms
        centroid = normed.mean(axis=0)
        cn = np.linalg.norm(centroid)
        if cn > 0:
            centroid = centroid / cn
        out.append(
            ThemeCandidate(
                pass_name="global_similarity",
                chunk_ids=member_ids,
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


# ----------------------------------------------------------------------------
# Phase 4 — top-level orchestrator (called by fs.build_layer_b)
# ----------------------------------------------------------------------------


def _utc_iso() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()


def build_layer_b(
    fs,  # type: ignore[no-untyped-def]
    *,
    facet: str = "foods",
    dry_run: bool = False,
):
    """Top-level orchestrator — hybrid global/per-shelf design (v0.2).

    Flow:
      1. Collect all chunks attached to non-synth shelves of `facet`.
      2. Run Pass 1 (similarity) globally across ALL attached chunks — one
         Leiden on a corpus-wide kNN graph.  Safety hatch: if the corpus
         exceeds `cfg.global_similarity_max_chunks`, skip and emit [].
      3. Run Pass 2 (relatedness) per shelf (entity coherence is sharper
         inside a single shelf's chunk set).
      4. Merge global × per-shelf via `merge_global_and_local_candidates`.
      5. Backfill `shelf_ids` on unmerged global_similarity themes via
         chunk.shelf_ids filtered to this facet's non-synth shelves.
      6. Label (c-TF-IDF / LLM) + pick primary + build Theme records.
      7. Persist: `clear_themes` + single `bulk_set_theme_ids` to zero stale
         denorm + `persist_themes` for all themes in one call.

    Skips the synthetic facet root (`facet:{facet}`) — the unclassified
    bucket from iteration-8, not a coherent topic.

    Returns a `LayerBArtifact` summarising the run.
    """
    import warnings
    from collections import Counter

    import igraph as ig
    import numpy as np

    from foodscholar.io.graph import Theme
    from foodscholar.layer_b.label import label_by_keywords, label_by_llm
    from foodscholar.layer_b.merge import merge_global_and_local_candidates
    from foodscholar.layer_b.models import LayerBArtifact
    from foodscholar.layer_b.persist import persist_themes
    from foodscholar.layer_b.primary import pick_primary
    from foodscholar.layer_b.relatedness_graph import build_relatedness_graph
    from foodscholar.versioning import make_artifact_meta

    cfg = fs.config.layer_b
    meta = make_artifact_meta(phase="layer_b", config=fs.config, record_count=0)
    started = _utc_iso()

    # 1. Collect attachments — invert chunk→shelves to shelf→chunks.
    attachments = fs.graph_store.list_chunk_shelf_attachments()
    shelf_to_chunks: dict[str, list[str]] = {}
    for chunk_id, shelf_ids in attachments.items():
        for sid in shelf_ids:
            shelf_to_chunks.setdefault(sid, []).append(chunk_id)

    # Filter to facet-relevant shelves; exclude the synthetic facet root.
    facet_shelves: dict[str, object] = {
        s.shelf_id: s for s in fs.graph_store.list_shelves() if s.facet == facet
    }
    synth_root = f"facet:{facet}"

    # Sorted union of all chunks attached to non-synth facet shelves.
    attached_chunk_ids: list[str] = sorted({
        cid
        for cid, sids in attachments.items()
        if any(sid in facet_shelves and sid != synth_root for sid in sids)
    })

    # Will accumulate every chunk_id that lands in any theme this run so we
    # can issue one final bulk_set_theme_ids that clears stale denorm.
    chunks_touched_this_run: set[str] = set()

    # 2. Global Pass 1 (cross-shelf similarity).
    if len(attached_chunk_ids) > cfg.global_similarity_max_chunks:
        warnings.warn(
            f"Attached corpus ({len(attached_chunk_ids)}) exceeds "
            f"cfg.global_similarity_max_chunks ({cfg.global_similarity_max_chunks}); "
            "skipping global Pass 1.",
            stacklevel=2,
        )
        global_cands: list[ThemeCandidate] = []
    else:
        global_cands = build_global_similarity_candidates(
            chunk_ids=attached_chunk_ids,
            chunk_store=fs.chunk_store,
            cfg=cfg,
        )

    # 3. Per-shelf Pass 2 (relatedness).
    rel_cands_by_shelf: dict[str, list[ThemeCandidate]] = {}
    for shelf_id, chunk_ids in shelf_to_chunks.items():
        if shelf_id not in facet_shelves or shelf_id == synth_root:
            continue
        if len(chunk_ids) < cfg.min_chunks_per_shelf:
            continue
        chunks = fs.chunk_store.get_many(chunk_ids)
        embedded = [c for c in chunks if c.embedding is not None]
        # Embedded-fraction gate: skip if too few are embedded.
        if len(chunks) > 0 and (len(embedded) / len(chunks)) < cfg.min_embedded_fraction:
            continue
        if len(embedded) < cfg.min_chunks_per_shelf:
            continue
        rel_cands_by_shelf[shelf_id] = build_shelf_relatedness_candidates(chunks, cfg)

    # 4. Merge global × per-shelf.
    theme_dicts, _decisions = merge_global_and_local_candidates(
        global_cands, rel_cands_by_shelf, cfg.merge
    )

    # 5. Backfill shelf_ids for unmerged global_similarity themes.
    chunk_shelf_map: dict[str, list[str]] = {
        cid: [sid for sid in sids if sid in facet_shelves and sid != synth_root]
        for cid, sids in attachments.items()
    }
    for td in theme_dicts:
        if td["discovery_pass"] != "global_similarity" or td.get("shelf_ids"):
            continue
        shelf_union: set[str] = set()
        for cid in td["chunk_ids"]:
            shelf_union.update(chunk_shelf_map.get(cid, []))
        td["shelf_ids"] = sorted(shelf_union)

    # 6. Label + primary + build Theme records.
    if not theme_dicts:
        # No themes: clear+persist are no-ops but still clear ghost themes.
        if not dry_run:
            fs.graph_store.clear_themes()
        return LayerBArtifact(
            artifact_id=meta.artifact_id,
            facet=facet,  # type: ignore[arg-type]
            config_hash=meta.config_hash,
            n_shelves_themed=0,
            n_shelves_skipped=len(facet_shelves),
            n_themes_total=0,
            n_themes_by_pass={},  # type: ignore[arg-type]
            leiden_seed=cfg.leiden.random_state,
            started_at=started,
            finished_at=_utc_iso(),
        )

    all_theme_chunk_ids: list[str] = sorted({cid for td in theme_dicts for cid in td["chunk_ids"]})
    chunks_by_id = {c.chunk_id: c for c in fs.chunk_store.get_many(all_theme_chunk_ids)}

    embeddings: dict[str, np.ndarray] = {
        c.chunk_id: np.asarray(c.embedding, dtype=np.float32)
        for c in chunks_by_id.values()
        if c.embedding is not None
    }

    # c-TF-IDF labeling over all themes at once.
    theme_chunks_for_labeling: dict[int, list] = {
        i: [chunks_by_id[cid] for cid in td["chunk_ids"] if cid in chunks_by_id]
        for i, td in enumerate(theme_dicts)
    }
    keywords = label_by_keywords(theme_chunks_for_labeling, cfg.labeling)
    if cfg.labeling.strategy == "llm" and fs.llm is not None:
        labels = label_by_llm(theme_chunks_for_labeling, keywords, fs.llm, cfg.labeling)
    else:
        labels = {
            i: " ".join(keywords.get(i, ["unlabeled"])[:3]) for i in theme_chunks_for_labeling
        }

    # Shared relatedness graph across all attached chunks for primary picking.
    all_attached_chunks = [
        chunks_by_id[cid] for cid in all_theme_chunk_ids if cid in chunks_by_id
    ]
    rel_graph = build_relatedness_graph(all_attached_chunks, cfg.relatedness)
    sim_graph = ig.Graph()  # reserved — picker doesn't use it today

    themes: list[Theme] = []
    chunk_assignments: dict[str, list[tuple[str, bool, float]]] = {}
    seq_by_pass: dict[str, int] = {"global_similarity": 0, "relatedness": 0, "merged": 0}

    for i, td in enumerate(theme_dicts):
        pass_kind = td["discovery_pass"]
        seq_by_pass.setdefault(pass_kind, 0)
        seq_by_pass[pass_kind] += 1
        seq = seq_by_pass[pass_kind]
        label = labels.get(i, "unlabeled")
        slug_seed = td["shelf_ids"][0] if td.get("shelf_ids") else f"facet_{facet}"
        tid = _theme_id(facet, slug_seed, label, pass_kind, seq)

        # Foodon entity signature: top-10 most-frequent high-conf entities.
        ent_counter: Counter[str] = Counter()
        for cid in td["chunk_ids"]:
            c = chunks_by_id.get(cid)
            if c is None:
                continue
            for link in c.entity_links:
                if link.confidence >= cfg.relatedness.tau_strict:
                    ent_counter[link.ontology_id] += 1
        signature = [oid for oid, _ in ent_counter.most_common(10)]

        # Pick primary chunk — centroid comes from the global_sim candidate
        # for global_similarity/merged themes.
        centroid = None
        if pass_kind in ("global_similarity", "merged"):
            for gc in global_cands:
                if gc.chunk_ids & set(td["chunk_ids"]):
                    centroid = gc.centroid_embedding
                    break
        primary_chunk = pick_primary(
            chunk_ids=set(td["chunk_ids"]),
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
                shelf_ids=list(td.get("shelf_ids", [])),
                chunk_count=len(td["chunk_ids"]),
                discovered_by=td.get("discovered_by", "leiden"),
                discovery_version="v0.2",
                facet=facet,  # type: ignore[arg-type]
                discovery_pass=pass_kind,  # type: ignore[arg-type]
                keyword_terms=list(keywords.get(i, [])),
                foodon_id_signature=signature,
                config_hash=meta.config_hash,
                version="v0.2",
            )
        )
        chunk_assignments[tid] = [
            (cid, cid == primary_chunk, 1.0 if cid == primary_chunk else 0.5)
            for cid in sorted(td["chunk_ids"])
        ]
        chunks_touched_this_run.update(td["chunk_ids"])

    # 7. Persist — clear ghost themes, zero stale denorm, write new themes.
    if not dry_run:
        fs.graph_store.clear_themes()
        # Zero theme_ids for every chunk touched this run before persist so
        # stale denorm from prior runs doesn't bleed through (preserve the
        # single-bulk-set-theme-ids contract that the old orchestrator had).
        fs.chunk_store.bulk_set_theme_ids(
            [(cid, []) for cid in sorted(chunks_touched_this_run)]
        )
        persist_themes(themes, chunk_assignments, fs.graph_store, fs.chunk_store)

    by_pass: dict[str, int] = {}
    for t in themes:
        by_pass[t.discovery_pass] = by_pass.get(t.discovery_pass, 0) + 1

    shelves_with_themes: set[str] = {s for t in themes for s in t.shelf_ids}

    return LayerBArtifact(
        artifact_id=meta.artifact_id,
        facet=facet,  # type: ignore[arg-type]
        config_hash=meta.config_hash,
        n_shelves_themed=len(shelves_with_themes),
        n_shelves_skipped=len(facet_shelves) - len(shelves_with_themes),
        n_themes_total=len(themes),
        n_themes_by_pass=by_pass,  # type: ignore[arg-type]
        leiden_seed=cfg.leiden.random_state,
        started_at=started,
        finished_at=_utc_iso(),
    )
