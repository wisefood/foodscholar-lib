"""Layer A backbone orchestrator.

Multi-facet projection of the loaded ontology onto the corpus. For each facet
in `cfg.layer_a.facets`:

  1. `collect_support(...)` walks chunks, filters by confidence + facet route.
  2. `prune(...)` applies blacklist -> threshold -> depth-cap lift -> collapse.
  3. Empty support tables emit a single stub root via `facet.stub_root(...)`.

The merged shelf list is upserted via `graph_store.upsert_shelves`. No chunk
attachments here — that's `fs.attach()`, the next phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.io.artifacts import ArtifactMeta
from foodscholar.io.graph import Shelf
from foodscholar.layer_a.facet import stub_root
from foodscholar.layer_a.propagate import SupportTable, collect_support
from foodscholar.layer_a.prune import prune, shelf_id_for_foodon
from foodscholar.logging import get_logger
from foodscholar.versioning import make_artifact_meta

if TYPE_CHECKING:
    from foodscholar.config import FoodScholarConfig, LayerAConfig
    from foodscholar.io.graph import Facet
    from foodscholar.ontology import FoodOnAPI
    from foodscholar.storage.protocols import ChunkStore, GraphStore, LLMClient

_log = get_logger("foodscholar.layer_a")


def build_shelves(
    chunk_store: ChunkStore,
    ontology: FoodOnAPI,
    config: LayerAConfig,
    *,
    llm: LLMClient | None = None,
) -> list[Shelf]:
    """Build Layer A shelves across every configured facet.

    Returns a single merged, sorted list — shelves carry their own `facet`
    field, so the graph store keeps them distinct.
    """
    all_shelves: list[Shelf] = []
    for facet in config.facets:
        all_shelves.extend(_build_facet(chunk_store, ontology, config, facet, llm=llm))
    return sorted(
        all_shelves,
        key=lambda s: (s.facet, s.depth, s.label.lower(), s.shelf_id),
    )


def build_layer_a(
    chunk_store: ChunkStore,
    graph_store: GraphStore,
    ontology: FoodOnAPI,
    *,
    config: LayerAConfig,
    full_config: FoodScholarConfig,
    llm: LLMClient | None = None,
) -> ArtifactMeta:
    """Build and store Layer A shelves, returning phase metadata.

    Always starts by clearing any prior projection (`graph_store.clear_layer_a`)
    so re-runs with different thresholds / blacklists don't leave ghost
    shelves from the previous build. `upsert_shelves` uses MERGE on
    `shelf_id`, which never deletes — clearing first is the only way to make
    re-runs idempotent against the *result*, not the *cumulative history*.
    """
    shelves = build_shelves(chunk_store, ontology, config, llm=llm)
    # Aliasing pass: give jargon-labelled shelves a human-facing display_label for
    # browsable navigation. Additive — never changes labels/ids/structure.
    if llm is not None and config.alias_shelves:
        from foodscholar.layer_a.alias import alias_shelves

        shelves = alias_shelves(shelves, ontology, llm=llm)
    graph_store.clear_layer_a()
    graph_store.upsert_shelves(shelves)

    by_facet: dict[str, int] = {}
    for shelf in shelves:
        by_facet[shelf.facet] = by_facet.get(shelf.facet, 0) + 1

    meta = make_artifact_meta(
        phase="build-layer-a",
        config=full_config,
        record_count=len(shelves),
    )
    _log.info(
        "layer_a.done",
        n_shelves=len(shelves),
        by_facet=by_facet,
        min_support=config.min_support,
        max_depth=config.max_depth,
        artifact_id=meta.artifact_id,
        config_hash=meta.config_hash,
    )
    return meta


# ---------------------------------------------------------------- internals


def _build_facet(
    chunk_store: ChunkStore,
    ontology: FoodOnAPI,
    config: LayerAConfig,
    facet: Facet,
    *,
    llm: LLMClient | None = None,
) -> list[Shelf]:
    facet_config = config.resolve_facet(facet)

    # iter_chunks yields batches; pruner consumes a flat chunk iterator.
    def chunk_iter():
        for batch in chunk_store.iter_chunks():
            yield from batch

    if facet_config.bottom_up_grouping.enabled:
        # Deferred import: grouping imports from this module, so a top-level
        # import would create a cycle.
        from foodscholar.layer_a.grouping import build_grouped_shelves

        return build_grouped_shelves(
            chunk_iter(),
            ontology,
            facet_config.bottom_up_grouping,
            facet=facet,
            min_link_confidence=facet_config.min_link_confidence,
            llm=llm,
        )

    support = collect_support(
        chunk_iter(),
        ontology,
        min_link_confidence=facet_config.min_link_confidence,
        facet=facet,
        link_blocklist=facet_config.link_blocklist,
    )

    if not support:
        return [stub_root(facet)]

    if config.projection == "backbone":
        from foodscholar.layer_a.backbone import build_backbone_shelves

        shelves = build_backbone_shelves(
            support, ontology, facet_config, facet,
            max_children=config.backbone_max_children,
        )
    else:
        shelves = prune(support, ontology, facet_config, facet)
    if not shelves:
        return [stub_root(facet)]
    return _ensure_single_root(shelves, facet, support)


def _ensure_single_root(
    shelves: list[Shelf],
    facet: Facet,
    support: SupportTable,
) -> list[Shelf]:
    """Guarantee one entry point per facet.

    Layer A often produces a forest — terms whose FoodOn ancestry doesn't
    intersect with any surviving ancestor become depth-0 orphans. Users
    navigate by facet, not by ontology branch, so a forest is a worse UX than
    a single tree. Inject a synthetic facet root and re-parent every current
    root (parent_shelf_id=None) under it. After this:
      - depth 0 = 1 shelf (the facet root)
      - depth 1 = the former roots
      - depth 2+ = everything else, shifted down by 1
    Re-rooted shelves keep their previous depth + 1 (clamped to a sane max).
    """
    roots = [s for s in shelves if s.parent_shelf_id is None]
    if len(roots) <= 1:
        # Already single-rooted (or empty) — no synthetic root needed.
        return shelves

    # Build the facet root. By construction it has no foodon_id — nothing
    # in any ontology resolves to it — so no chunk can ever name it directly.
    # support_direct is 0; every chunk reaching it counts as lifted.
    #
    # Honest chunk_count = unique chunks reaching ANY former root (union of
    # their chunk-id sets from the support table). Previously this was
    # `sum(s.chunk_count for s in roots)` — but a chunk linking to multiple
    # FOODON terms in different orphan branches contributes to each root's
    # count, so summing double-counted heavily on the real corpus (6,290
    # unique chunks reported as 13,731 on the foods root).
    root_chunk_ids: set = set()
    for root_shelf_node in roots:
        if root_shelf_node.foodon_id is None:
            continue
        chunks_for_root = support.with_descendants_chunk_ids.get(
            root_shelf_node.foodon_id
        )
        if chunks_for_root:
            root_chunk_ids |= chunks_for_root
    total_count = len(root_chunk_ids)

    root_shelf = Shelf(
        shelf_id=f"facet:{facet}",
        label=_FACET_ROOT_LABELS[facet],
        facet=facet,
        depth=0,
        foodon_id=None,
        parent_shelf_id=None,
        chunk_count=total_count,
        support_direct=0,
        support_lifted=total_count,
        see_also=[],
    )

    # Re-parent every former root + shift every shelf's depth by 1.
    rerooted: list[Shelf] = [root_shelf]
    for s in shelves:
        new_parent = root_shelf.shelf_id if s.parent_shelf_id is None else s.parent_shelf_id
        rerooted.append(
            s.model_copy(update={
                "parent_shelf_id": new_parent,
                "depth": s.depth + 1,
            })
        )
    return rerooted


_FACET_ROOT_LABELS: dict[Facet, str] = {
    "foods": "Foods",
    "health": "Health",
    "sustainability": "Sustainability",
    "dietary_patterns": "Dietary patterns",
    "allergies": "Allergies",
    "nutrients": "Nutrients",
}


__all__ = ["build_layer_a", "build_shelves", "shelf_id_for_foodon"]
