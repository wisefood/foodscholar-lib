"""Cross-store audit for Layer B (per `layer_b_construction_brief.md` §10).

`audit_layer_b(chunk_store, graph_store) -> LayerBAuditReport`:

CRITICAL invariants (any failure flips `report.passed` to False):
  - parity == 1.0  — Neo4j `(:Chunk)-[:THEME_OF]->(:Theme)` edges agree
    with the Elastic `theme_ids` denorm on each chunk
  - dangling_edges == 0  — no `theme_ids` on chunks pointing at themes
    that don't exist as `(:Theme)` nodes
  - empty_themes == 0  — no `(:Theme)` nodes with `chunk_count > 0` that
    have zero attached chunks at audit time

WARN-level reporting (informational only, doesn't fail `passed`):
  - by_pass: count per discovery_pass (similarity/relatedness/merged) —
    the brief's "≥ 1 per pass" canary (relatedness=0 means entity graph
    is mis-tuned; merged_rate=1.0 means Pass 2 isn't earning compute)
  - merged_rate: fraction of themes that are `discovery_pass="merged"`
  - n_themes / n_themed_shelves: rollout stats

The audit reads the full chunk store (via `scan()`) and all themes (via
`list_themes()` + `get_chunks_for_theme()`). For large corpora these are
cheap because each is a single round-trip; the per-theme `get_chunks_for_theme`
is N queries but N is small (≤ ~200 in v1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.layer_b.models import LayerBAuditReport

if TYPE_CHECKING:
    from foodscholar.storage.protocols import ChunkStore, GraphStore


def audit_layer_b(
    chunk_store: ChunkStore,
    graph_store: GraphStore,
) -> LayerBAuditReport:
    """Compute the §10 audit gates against live stores."""
    themes = graph_store.list_themes()
    valid_theme_ids = {t.theme_id for t in themes}

    # Graph-side (chunk_id, theme_id) pairs from THEME_OF / ATTACHED_TO edges
    graph_pairs: set[tuple[str, str]] = set()
    empty_themes = 0
    themed_shelves: set[str] = set()
    by_pass: dict[str, int] = {}
    for t in themes:
        chunks_in_theme = graph_store.get_chunks_for_theme(t.theme_id)
        if t.chunk_count > 0 and not chunks_in_theme:
            empty_themes += 1
        for cid in chunks_in_theme:
            graph_pairs.add((cid, t.theme_id))
        for sid in t.shelf_ids:
            themed_shelves.add(sid)
        by_pass[t.discovery_pass] = by_pass.get(t.discovery_pass, 0) + 1

    # Chunk-side (chunk_id, theme_id) pairs from the ES theme_ids denorm
    chunk_pairs: set[tuple[str, str]] = set()
    dangling = 0
    for chunk in chunk_store.scan():
        for tid in chunk.theme_ids:
            chunk_pairs.add((chunk.chunk_id, tid))
            if tid not in valid_theme_ids:
                dangling += 1

    union = graph_pairs | chunk_pairs
    parity = (len(graph_pairs & chunk_pairs) / len(union)) if union else 1.0

    merged = by_pass.get("merged", 0)
    total = sum(by_pass.values())
    merged_rate = (merged / total) if total else 0.0

    return LayerBAuditReport(
        parity=parity,
        dangling_edges=dangling,
        empty_themes=empty_themes,
        n_themes=len(themes),
        n_themed_shelves=len(themed_shelves),
        by_pass=by_pass,
        merged_rate=merged_rate,
    )
