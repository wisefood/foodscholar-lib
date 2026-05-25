"""Persist Layer B output: Neo4j theme nodes + chunk-theme edges +
Elastic chunk-side `theme_ids` denorm.

Per `layer_b_construction_brief.md` §7. The flow is three writes in
lockstep:

  1. `graph_store.upsert_themes(themes)` — creates/updates `(:Theme)` nodes
     and the shelf↔theme edge (via `IN_SHELF` direction in the existing
     Neo4j schema; semantically equivalent to brief's `HAS_THEME`).
  2. `graph_store.attach_chunks_to_themes_bulk(items)` — writes
     `(:Chunk)-[:THEME_OF {primary, weight}]->(:Theme)` edges.
  3. `chunk_store.bulk_set_theme_ids(items)` — denormalizes the per-chunk
     `theme_ids` list to Elastic. Uses the dedicated `bulk_set_theme_ids`
     so `shelf_ids` are untouched (Plan-agent flag — `bulk_update_attachments`
     would clobber Layer A's denorm under concurrent writes).

Caller (build_layer_b orchestrator) is responsible for calling
`graph_store.clear_themes()` + zeroing `theme_ids` on affected chunks before
re-running persist if it wants a clean slate; persist itself is purely
additive within the themes it's given.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foodscholar.io.graph import Theme
    from foodscholar.storage.protocols import ChunkStore, GraphStore


def persist_themes(
    themes: list[Theme],
    chunk_assignments: dict[str, list[tuple[str, bool, float]]],
    graph_store: GraphStore,
    chunk_store: ChunkStore,
) -> None:
    """Persist themes + chunk-theme edges + ES `theme_ids` denorm.

    `chunk_assignments[theme_id]` = list of `(chunk_id, primary, weight)`.
    Empty `themes` is a no-op.
    """
    if not themes:
        return

    # 1. Theme nodes + IN_SHELF edges
    graph_store.upsert_themes(themes)

    # 2. THEME_OF edges with (primary, weight) metadata — one bulk call
    edge_items: list[tuple[str, str, bool, float]] = []
    for theme_id, assignments in chunk_assignments.items():
        for chunk_id, primary, weight in assignments:
            edge_items.append((chunk_id, theme_id, primary, weight))
    graph_store.attach_chunks_to_themes_bulk(edge_items)

    # 3. ES theme_ids denorm — group by chunk_id, set the union per chunk.
    # Uses bulk_set_theme_ids (touches theme_ids only, leaves shelf_ids alone).
    by_chunk: dict[str, set[str]] = {}
    for theme_id, assignments in chunk_assignments.items():
        for chunk_id, _, _ in assignments:
            by_chunk.setdefault(chunk_id, set()).add(theme_id)

    bulk_items: list[tuple[str, list[str]]] = []
    for chunk_id, theme_ids in by_chunk.items():
        existing = chunk_store.get(chunk_id)
        if existing is None:
            continue
        # Merge with any pre-existing theme_ids on the chunk (a chunk attached
        # to two shelves could land in themes in both shelves; preserve the
        # other shelf's themes that this run didn't touch).
        merged = sorted(set(existing.theme_ids) | theme_ids)
        bulk_items.append((chunk_id, merged))

    chunk_store.bulk_set_theme_ids(bulk_items)
