"""Read-only WARN-level quality + tuning report for Layer B.

`build_quality_report(chunk_store, graph_store, cfg, *, facet)` reads the live
shelves, themes, and chunk→shelf attachments and returns a
`LayerBQualityReport` — metrics plus a list of `LayerBWarning`s. It mutates
nothing and is cleanly separate from `audit_layer_b()`, which alone gates
CRITICAL build invariants.

The metric core is `compute_quality_metrics(shelves, themes, attachments,
themed_chunk_ids, cfg, facet)` — a pure function so the tuning sweep can score a
`dry_run` build's `themes_preview` against the same logic without re-reading the
persisted store.

Theme sources map the persisted `discovery_pass` values onto the brief's
vocabulary: ``similarity_only`` := ``global_similarity``, ``relatedness_only``
:= ``relatedness``, ``merged`` := ``merged``.

See docs/superpowers/specs/2026-06-05-layer-b-quality-tuning-design.md §3.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from foodscholar.layer_b.models import LayerBQualityReport, LayerBWarning

if TYPE_CHECKING:
    from foodscholar.config import LayerBConfig
    from foodscholar.io.graph import Shelf, Theme
    from foodscholar.storage.protocols import ChunkStore, GraphStore


def _label_tokens(label: str) -> set[str]:
    """Lowercased whitespace-split token set for Jaccard comparison."""
    return {t for t in label.lower().split() if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return (len(a & b) / len(union)) if union else 0.0


def _source_of(discovery_pass: str) -> str:
    """Map a persisted discovery_pass onto the brief's source vocabulary."""
    return {
        "merged": "merged",
        "global_similarity": "similarity_only",
        "relatedness": "relatedness_only",
    }.get(discovery_pass, discovery_pass)


def compute_quality_metrics(
    shelves: list[Shelf],
    themes: list[Theme],
    attachments: dict[str, set[str]],
    themed_chunk_ids: set[str],
    cfg: LayerBConfig,
    facet: str,
) -> LayerBQualityReport:
    """Pure metric + warning computation over already-loaded inputs.

    `shelves` / `themes` must already be filtered to `facet` and exclude the
    synthetic facet root. `attachments` is chunk_id -> set(shelf_id) for the
    whole store; it's filtered to the facet's shelves here. `themed_chunk_ids`
    is the set of attached chunk_ids that landed in at least one theme (read
    from the store's `theme_ids` denorm for the live report, or derived from
    `themes_preview` for the sweep).
    """
    shelf_ids = {s.shelf_id for s in shelves}
    shelf_by_id = {s.shelf_id: s for s in shelves}
    min_size = cfg.leiden.min_community_size
    ac = cfg.audit

    # --- shelf structure ---
    depths = [s.depth for s in shelves]
    fanout: Counter[str] = Counter()
    for s in shelves:
        if s.parent_shelf_id is not None:
            fanout[s.parent_shelf_id] += 1
    sum_direct = sum(s.support_direct for s in shelves)
    sum_lifted = sum(s.support_lifted for s in shelves)
    chunk_counts = [s.chunk_count for s in shelves]

    # --- attached / themed chunks (restricted to this facet's shelves) ---
    attached = {
        cid for cid, sids in attachments.items() if sids & shelf_ids
    }
    themed = attached & themed_chunk_ids
    coverage = (len(themed) / len(attached)) if attached else 0.0

    # --- theme sources ---
    sources = Counter(_source_of(t.discovery_pass) for t in themes)

    # --- duplicate labels (same lowercased label within one shelf) ---
    labels_by_shelf: dict[str, Counter[str]] = defaultdict(Counter)
    themes_by_shelf: dict[str, list[Theme]] = defaultdict(list)
    for t in themes:
        for sid in t.shelf_ids:
            labels_by_shelf[sid][t.label.strip().lower()] += 1
            themes_by_shelf[sid].append(t)
    dup_label_themes = sum(
        n for counts in labels_by_shelf.values() for lbl, n in counts.items() if n > 1
    )

    tiny = sum(1 for t in themes if t.chunk_count < min_size)
    leakage = sum(1 for t in themes if len(t.shelf_ids) > 1)

    report = LayerBQualityReport(
        facet=facet,  # type: ignore[arg-type]
        n_shelves=len(shelves),
        max_depth=max(depths) if depths else 0,
        median_depth=statistics.median(depths) if depths else 0.0,
        max_fanout=max(fanout.values()) if fanout else 0,
        shelves_zero_direct_support=sum(1 for s in shelves if s.support_direct == 0),
        direct_to_lifted_ratio=(sum_direct / sum_lifted) if sum_lifted else 0.0,
        chunks_per_shelf_min=min(chunk_counts) if chunk_counts else 0,
        chunks_per_shelf_median=statistics.median(chunk_counts) if chunk_counts else 0.0,
        chunks_per_shelf_max=max(chunk_counts) if chunk_counts else 0,
        n_themes=len(themes),
        theme_coverage=coverage,
        n_merged=sources.get("merged", 0),
        n_similarity_only=sources.get("similarity_only", 0),
        n_relatedness_only=sources.get("relatedness_only", 0),
        n_duplicate_label_themes=dup_label_themes,
        n_tiny_themes=tiny,
        n_orphan_chunks=len(attached - themed),
        n_cross_shelf_leakage=leakage,
    )

    # ------------------------------------------------------------------ warnings
    warnings: list[LayerBWarning] = []

    # high lifted / low direct support
    for s in shelves:
        ratio = s.support_lifted / max(s.support_direct, 1)
        if s.support_direct < ac.direct_support_floor and ratio >= ac.lifted_to_direct_ratio_max:
            warnings.append(
                LayerBWarning(
                    kind="high_lifted_low_direct",
                    shelf_id=s.shelf_id,
                    message=(
                        f"lifted={s.support_lifted} dwarfs direct={s.support_direct} "
                        f"(ratio {ratio:.1f}≥{ac.lifted_to_direct_ratio_max})"
                    ),
                )
            )

    # shelf eligible but produced no themes
    for s in shelves:
        if s.chunk_count >= cfg.min_chunks_per_shelf and not themes_by_shelf.get(s.shelf_id):
            warnings.append(
                LayerBWarning(
                    kind="shelf_no_themes",
                    shelf_id=s.shelf_id,
                    message=(
                        f"{s.chunk_count} chunks (≥{cfg.min_chunks_per_shelf}) but 0 themes"
                    ),
                )
            )

    # mostly single-pass within a shelf
    for sid, shelf_themes in themes_by_shelf.items():
        if len(shelf_themes) < 2:
            continue
        sim_only = sum(1 for t in shelf_themes if t.discovery_pass == "global_similarity")
        rel_only = sum(1 for t in shelf_themes if t.discovery_pass == "relatedness")
        n = len(shelf_themes)
        for label, count in (("similarity-only", sim_only), ("relatedness-only", rel_only)):
            share = count / n
            if share > ac.single_pass_share_max:
                warnings.append(
                    LayerBWarning(
                        kind="mostly_single_pass",
                        shelf_id=sid,
                        message=f"{share:.0%} of {n} themes are {label} (>{ac.single_pass_share_max:.0%})",
                    )
                )

    # near-duplicate labels within a shelf (token-set Jaccard)
    for sid, shelf_themes in themes_by_shelf.items():
        toks = [(t, _label_tokens(t.label)) for t in shelf_themes]
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                ti, si = toks[i]
                tj, sj = toks[j]
                if _jaccard(si, sj) >= ac.dup_label_jaccard_min:
                    warnings.append(
                        LayerBWarning(
                            kind="near_duplicate_labels",
                            shelf_id=sid,
                            theme_id=ti.theme_id,
                            message=f"'{ti.label}' ≈ '{tj.label}' ({tj.theme_id})",
                        )
                    )

    # theme spans too many entities
    for t in themes:
        if len(t.foodon_id_signature) > ac.max_entity_span:
            warnings.append(
                LayerBWarning(
                    kind="theme_spans_many_entities",
                    theme_id=t.theme_id,
                    message=(
                        f"signature spans {len(t.foodon_id_signature)} FoodOn entities "
                        f"(>{ac.max_entity_span})"
                    ),
                )
            )

    # theme label duplicates its parent shelf label
    for t in themes:
        tl = t.label.strip().lower()
        for sid in t.shelf_ids:
            s = shelf_by_id.get(sid)
            if s is None:
                continue
            shelf_labels = {
                (s.label or "").strip().lower(),
                (s.display_label or "").strip().lower(),
            } - {""}
            if tl in shelf_labels:
                warnings.append(
                    LayerBWarning(
                        kind="theme_label_equals_parent",
                        shelf_id=sid,
                        theme_id=t.theme_id,
                        message=f"theme label '{t.label}' equals parent shelf label",
                    )
                )
                break

    report.warnings = warnings
    return report


def build_quality_report(
    chunk_store: ChunkStore,
    graph_store: GraphStore,
    cfg: LayerBConfig,
    *,
    facet: str = "foods",
) -> LayerBQualityReport:
    """Compute the WARN-level quality report for `facet` against live stores."""
    synth_root = f"facet:{facet}"
    shelves = [
        s
        for s in graph_store.list_shelves()
        if s.facet == facet and s.shelf_id != synth_root
    ]
    facet_shelf_ids = {s.shelf_id for s in shelves}
    themes = [
        t
        for t in graph_store.list_themes()
        if t.facet == facet and any(sid in facet_shelf_ids for sid in t.shelf_ids)
    ]
    attachments = graph_store.list_chunk_shelf_attachments()

    # Themed = attached chunks whose ES theme_ids denorm is non-empty.
    attached_ids = sorted(
        {cid for cid, sids in attachments.items() if sids & facet_shelf_ids}
    )
    themed_chunk_ids = {
        c.chunk_id for c in chunk_store.get_many(attached_ids) if c.theme_ids
    }

    return compute_quality_metrics(
        shelves, themes, attachments, themed_chunk_ids, cfg, facet
    )
