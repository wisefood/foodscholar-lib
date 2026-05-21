"""Minimal Layer A construction from stored chunk annotations.

This first implementation intentionally uses the current corpus/storage
contract only: chunks already have `foodon_ids`, and Layer A turns supported
FoodOn terms plus their ancestors into graph shelves. Chunk-to-shelf attachment
is still a separate phase.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from foodscholar.config import LayerAConfig
from foodscholar.io.artifacts import ArtifactMeta
from foodscholar.io.graph import Shelf, ShelfId
from foodscholar.logging import get_logger
from foodscholar.versioning import make_artifact_meta

if TYPE_CHECKING:
    from foodscholar.config import FoodScholarConfig
    from foodscholar.ontology import FoodOnAPI
    from foodscholar.storage.protocols import ChunkStore, GraphStore

_log = get_logger("foodscholar.layer_a")


def shelf_id_for_foodon(term_id: str) -> ShelfId:
    """Return the stable shelf id used for a FoodOn term."""
    return f"foodon:{term_id}"


def build_shelves(
    chunk_store: ChunkStore,
    ontology: FoodOnAPI,
    config: LayerAConfig,
) -> list[Shelf]:
    """Build Layer A shelves from chunk-level `foodon_ids`.

    Each chunk contributes at most one support count to a given FoodOn term.
    Support is propagated upward to all known ancestors so broader shelves can
    exist even when chunks mention only leaf-level foods.
    """
    if "foods" not in config.facets:
        return []

    support = _collect_support(chunk_store, ontology)
    if not support:
        return []

    depths = {term_id: _depth(ontology, term_id) for term_id in support}
    included = {
        term_id
        for term_id, count in support.items()
        if count >= config.min_support
        and depths[term_id] <= config.max_depth
        and _is_allowed(term_id, ontology.id_to_label(term_id), config)
    }

    shelves = [
        Shelf(
            shelf_id=shelf_id_for_foodon(term_id),
            label=ontology.id_to_label(term_id) or term_id,
            facet="foods",
            depth=depths[term_id],
            foodon_id=term_id,
            parent_shelf_id=_nearest_included_parent(ontology, term_id, included),
            chunk_count=support[term_id],
        )
        for term_id in included
    ]
    return sorted(shelves, key=lambda s: (s.depth, s.label.lower(), s.shelf_id))


def build_layer_a(
    chunk_store: ChunkStore,
    graph_store: GraphStore,
    ontology: FoodOnAPI,
    *,
    config: LayerAConfig,
    full_config: FoodScholarConfig,
) -> ArtifactMeta:
    """Build and store Layer A shelves, returning phase metadata."""
    shelves = build_shelves(chunk_store, ontology, config)
    graph_store.upsert_shelves(shelves)

    meta = make_artifact_meta(
        phase="build-layer-a",
        config=full_config,
        record_count=len(shelves),
    )
    _log.info(
        "layer_a.done",
        n_shelves=len(shelves),
        min_support=config.min_support,
        max_depth=config.max_depth,
        artifact_id=meta.artifact_id,
        config_hash=meta.config_hash,
    )
    return meta


def _collect_support(chunk_store: ChunkStore, ontology: FoodOnAPI) -> Counter[str]:
    support: Counter[str] = Counter()
    for batch in chunk_store.iter_chunks():
        for chunk in batch:
            for term_id in sorted(set(chunk.foodon_ids)):
                if term_id not in ontology:
                    continue
                support[term_id] += 1
                for ancestor_id in ontology.id_to_ancestors(term_id):
                    if ancestor_id in ontology:
                        support[ancestor_id] += 1
    return support


def _depth(
    ontology: FoodOnAPI,
    term_id: str,
    seen: frozenset[str] = frozenset(),
) -> int:
    if term_id in seen:
        return 0

    parents = [p for p in ontology.id_to_parents(term_id) if p in ontology]
    if not parents:
        return 0

    next_seen = seen | {term_id}
    return 1 + min(_depth(ontology, parent_id, next_seen) for parent_id in parents)


def _nearest_included_parent(
    ontology: FoodOnAPI,
    term_id: str,
    included: set[str],
) -> ShelfId | None:
    frontier = sorted(parent_id for parent_id in ontology.id_to_parents(term_id) if parent_id in ontology)
    seen: set[str] = set()

    while frontier:
        parent_id = frontier.pop(0)
        if parent_id in seen:
            continue
        if parent_id in included:
            return shelf_id_for_foodon(parent_id)
        seen.add(parent_id)
        frontier.extend(
            sorted(
                grandparent_id
                for grandparent_id in ontology.id_to_parents(parent_id)
                if grandparent_id in ontology and grandparent_id not in seen
            )
        )
    return None


def _is_allowed(
    term_id: str,
    label: str | None,
    config: LayerAConfig,
) -> bool:
    blocked = {term.lower().strip() for term in config.blacklist_terms}
    if term_id.lower() in blocked:
        return False
    return label is None or label.lower().strip() not in blocked
