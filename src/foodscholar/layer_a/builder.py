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
from foodscholar.layer_a.propagate import collect_support
from foodscholar.layer_a.prune import prune, shelf_id_for_foodon
from foodscholar.logging import get_logger
from foodscholar.versioning import make_artifact_meta

if TYPE_CHECKING:
    from foodscholar.config import FoodScholarConfig, LayerAConfig
    from foodscholar.io.graph import Facet
    from foodscholar.ontology import FoodOnAPI
    from foodscholar.storage.protocols import ChunkStore, GraphStore

_log = get_logger("foodscholar.layer_a")


def build_shelves(
    chunk_store: ChunkStore,
    ontology: FoodOnAPI,
    config: LayerAConfig,
) -> list[Shelf]:
    """Build Layer A shelves across every configured facet.

    Returns a single merged, sorted list — shelves carry their own `facet`
    field, so the graph store keeps them distinct.
    """
    all_shelves: list[Shelf] = []
    for facet in config.facets:
        all_shelves.extend(_build_facet(chunk_store, ontology, config, facet))
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
) -> ArtifactMeta:
    """Build and store Layer A shelves, returning phase metadata.

    Always starts by clearing any prior projection (`graph_store.clear_layer_a`)
    so re-runs with different thresholds / blacklists don't leave ghost
    shelves from the previous build. `upsert_shelves` uses MERGE on
    `shelf_id`, which never deletes — clearing first is the only way to make
    re-runs idempotent against the *result*, not the *cumulative history*.
    """
    shelves = build_shelves(chunk_store, ontology, config)
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
) -> list[Shelf]:
    facet_config = config.resolve_facet(facet)

    # iter_chunks yields batches; pruner consumes a flat chunk iterator.
    def chunk_iter():
        for batch in chunk_store.iter_chunks():
            yield from batch

    support = collect_support(
        chunk_iter(),
        ontology,
        min_link_confidence=facet_config.min_link_confidence,
        facet=facet,
    )

    if not support:
        return [stub_root(facet)]

    shelves = prune(support, ontology, facet_config, facet)
    if not shelves:
        return [stub_root(facet)]
    return shelves


__all__ = ["build_layer_a", "build_shelves", "shelf_id_for_foodon"]
