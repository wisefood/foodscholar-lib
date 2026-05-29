"""Layer A attachment: wire chunks to their projected shelves.

This is the phase that runs *after* `build_layer_a` has decided which shelves
survive projection. For each chunk:

  1. Group the chunk's FOODON ids by facet (via `route_link_to_facet` on
     `entity_links`; the `foodon_ids` denorm contributes to `foods` as a cheap
     fallback for prototype-NEL data with no per-mention `entity_type`).
  2. For each (facet, foodon_id) pair, resolve to a surviving shelf by:
        a. direct       — a shelf with this exact `foodon_id`
        b. collapsed    — a shelf carrying this id in its `see_also`
        c. lifted       — the deepest surviving ancestor of this id, restricted
                          to shelves on the same facet (only matters once
                          non-foods facets have real projection support)
        d. orphan       — fall through to the synthetic facet root if one
                          exists (`facet:<facet>`); otherwise drop
  3. Record which foodon_ids resolved to each shelf as `lifted_from` on the
     `(:Chunk)-[:ATTACHED_TO]->(:Shelf)` edge. Empty list means "direct".

The resolver is a pure function over `ShelfIndex` + an `Ontology`; the
orchestrator handles iteration + writes. Tests exercise the resolver directly.

Attaches via the **nearest surviving ancestor only** — for a cow-milk chunk
with no `cow milk` shelf but surviving `dairy product` and `food product`
shelves, we attach to `dairy product` (deepest), not both. This keeps
`chunk_count` honest and mirrors the support-table semantics that drove
projection.

Performance notes
-----------------
The naive shape (one ES `_update` per chunk + one Cypher per shelf-flush, all
sequential) burns minutes on a multi-thousand-chunk corpus. This module
applies three independent fixes:

- **Bulk denorm** — `chunk_store.bulk_update_attachments(...)` ships every
  chunk's `shelf_ids` to ES in one `_bulk` call per batch. ~50ms instead of
  ~50ms per chunk.
- **Resolver cache** — `_FacetIndex.resolve_lifted(...)` memoizes the deepest
  surviving ancestor per FOODON id. Real corpora are Zipfian on FoodOn ids,
  so cache hit rates are >90% past the first batch.
- **Parallel flush** — Neo4j edge writes and Elastic denorm updates are
  independent backends; we dispatch them concurrently via a `ThreadPoolExecutor`
  so backend latency overlaps instead of serializing.

Rerun semantics
---------------
`attach()` calls `graph_store.clear_attachments()` and
`chunk_store.clear_attachments()` at the start so a re-run with a different
projection (or just different resolver knobs) produces a clean state. Shelves
themselves are untouched — those are the output of `build_layer_a`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from foodscholar.io.artifacts import ArtifactMeta
from foodscholar.layer_a.facet import route_link_to_facet
from foodscholar.logging import get_logger
from foodscholar.versioning import make_artifact_meta

if TYPE_CHECKING:
    from foodscholar.config import FoodScholarConfig
    from foodscholar.io.chunk import Chunk, ChunkId
    from foodscholar.io.graph import Facet, Shelf, ShelfId, ThemeId
    from foodscholar.ontology import FoodOnAPI
    from foodscholar.storage.protocols import ChunkStore, GraphStore

_log = get_logger("foodscholar.layer_a.attach")


# ---------------------------------------------------------------- index


@dataclass
class _FacetIndex:
    """Lookup tables for one facet's surviving shelves.

    Carries a memoization cache for the ancestor walk so popular FOODON ids
    (which dominate real corpora — Zipfian on a small head of foods) only pay
    the `id_to_ancestors` walk + depth scan once per attach() run.
    """

    by_foodon: dict[str, Shelf] = field(default_factory=dict)
    by_seealso: dict[str, Shelf] = field(default_factory=dict)
    synthetic_root: Shelf | None = None  # `facet:<facet>` shelf if present
    # Memo for `_deepest_surviving_ancestor`. None means "no surviving ancestor."
    _lifted_cache: dict[str, Shelf | None] = field(default_factory=dict)

    def resolve_lifted(
        self, foodon_id: str, ontology: FoodOnAPI
    ) -> Shelf | None:
        """Deepest surviving ancestor, with per-attach-call memoization."""
        cached = self._lifted_cache.get(foodon_id, _MISS)
        if cached is not _MISS:
            return cached  # type: ignore[return-value]
        best: Shelf | None = None
        for anc_id in ontology.id_to_ancestors(foodon_id):
            shelf = self.by_foodon.get(anc_id) or self.by_seealso.get(anc_id)
            if shelf is None:
                continue
            if best is None or shelf.depth > best.depth:
                best = shelf
        self._lifted_cache[foodon_id] = best
        return best


# Sentinel for cache misses — distinguishes "never queried" from "queried,
# resolved to None (no surviving ancestor)" so the latter is also cached.
_MISS: object = object()


@dataclass
class ShelfIndex:
    """Per-facet shelf indices, built once from a list of survivors."""

    per_facet: dict[Facet, _FacetIndex] = field(default_factory=dict)

    @classmethod
    def from_shelves(cls, shelves: Iterable[Shelf]) -> ShelfIndex:
        idx = cls()
        for shelf in shelves:
            facet_idx = idx.per_facet.setdefault(shelf.facet, _FacetIndex())
            if shelf.foodon_id is not None:
                facet_idx.by_foodon[shelf.foodon_id] = shelf
            for fid in shelf.see_also:
                facet_idx.by_seealso[fid] = shelf
            if shelf.shelf_id == f"facet:{shelf.facet}":
                facet_idx.synthetic_root = shelf
        return idx

    def facets(self) -> Iterable[Facet]:
        return self.per_facet.keys()


# ---------------------------------------------------------------- resolver


def _per_facet_foodon_ids(chunk: Chunk) -> dict[Facet, set[str]]:
    """Group a chunk's FOODON ids by the facet each routes to.

    A FOODON-prefixed `entity_link` routes to its mention's facet (or `foods`
    via the fallback for prototype `entity_type='other'` links). The
    `foodon_ids` denorm only seeds the `foods` facet — it's a cheap path for
    prototype-NEL data that has no per-mention entity_type. Non-FOODON
    `entity_links` (CHEBI / GAZ / MONDO / …) carry no FOODON id, so they
    contribute nothing here — they'd attach via their own ontology, which
    Layer A doesn't project yet.
    """
    grouped: dict[Facet, set[str]] = defaultdict(set)
    for link in chunk.entity_links:
        facet = route_link_to_facet(link)
        if facet is None:
            continue
        if not link.ontology_id.startswith("FOODON:"):
            continue  # only FOODON-prefixed ids reach FoodOn-projected shelves
        grouped[facet].add(link.ontology_id)
    for fid in chunk.foodon_ids:
        grouped["foods"].add(fid)
    return grouped


def resolve_chunk(
    chunk: Chunk,
    index: ShelfIndex,
    ontology: FoodOnAPI,
) -> dict[ShelfId, list[str]]:
    """Resolve a chunk to a `{shelf_id: lifted_from foodon_ids}` map.

    Empty `lifted_from` means the chunk had a `foodon_id` that *is* the
    shelf's `foodon_id` — a direct attachment. A non-empty list lists the
    foodon_ids that reached this shelf via collapse or ancestor lift.
    Synthetic facet roots collect every chunk that routed to that facet but
    couldn't reach any real shelf — those edges carry their orphan foodon_ids
    as `lifted_from` for traceability.

    Resolution is per (facet, foodon_id) and picks the nearest surviving
    ancestor only; one chunk-shelf pair per resolution, deduped across
    foodon_ids that resolve to the same shelf (their ids merge into the same
    `lifted_from` list).
    """
    resolutions: dict[ShelfId, set[str]] = defaultdict(set)
    direct_shelves: set[ShelfId] = set()

    for facet, foodon_ids in _per_facet_foodon_ids(chunk).items():
        facet_idx = index.per_facet.get(facet)
        if facet_idx is None:
            continue
        orphans: list[str] = []
        for fid in foodon_ids:
            shelf = facet_idx.by_foodon.get(fid)
            if shelf is not None:
                resolutions[shelf.shelf_id]  # touch
                direct_shelves.add(shelf.shelf_id)
                continue
            shelf = facet_idx.by_seealso.get(fid)
            if shelf is not None:
                resolutions[shelf.shelf_id].add(fid)
                continue
            shelf = facet_idx.resolve_lifted(fid, ontology)
            if shelf is not None:
                resolutions[shelf.shelf_id].add(fid)
                continue
            orphans.append(fid)

        # Fallback: route remaining orphans to the synthetic facet root so
        # they're still discoverable. The synthetic root carries support
        # accounting in builder._ensure_single_root; here we mirror it on the
        # edge side.
        if orphans and facet_idx.synthetic_root is not None:
            resolutions[facet_idx.synthetic_root.shelf_id].update(orphans)

    # Direct shelves get an empty list (overriding any merged-in foodon_ids
    # — direct attachment is the strongest mode and should be visually
    # distinct in audits).
    return {
        sid: [] if sid in direct_shelves else sorted(fids)
        for sid, fids in resolutions.items()
    }


# ---------------------------------------------------------------- orchestrator


def attach(
    chunk_store: ChunkStore,
    graph_store: GraphStore,
    ontology: FoodOnAPI,
    *,
    full_config: FoodScholarConfig,
    batch_size: int = 1000,
    max_workers: int = 1,
) -> ArtifactMeta:
    """Write `(:Chunk)-[:ATTACHED_TO {lifted_from}]->(:Shelf)` edges + denorm
    `shelf_ids` onto every chunk that reaches at least one shelf.

    Always starts by clearing prior attachments (both the Neo4j edges and the
    Elastic `shelf_ids` denorm) so a re-run with a different projection or
    different resolver knobs produces a clean state. Shelves themselves are
    untouched.

    The flush path runs ES and Neo4j writes in parallel via a
    `ThreadPoolExecutor` — the two backends are independent, so overlapping
    their network latency is essentially free. `max_workers=2` is the natural
    minimum; raising it further only helps if a single backend's bulk call is
    slower than batch resolution itself.

    Idempotent: re-running with the same chunks + same projection produces the
    same edges and the same denormalized lists.
    """
    graph_store.clear_attachments()
    chunk_store.clear_attachments()

    shelves = graph_store.list_shelves()
    index = ShelfIndex.from_shelves(shelves)

    # Per-shelf edge buffer: shelf_id -> {chunk_id: lifted_from}. A dict (not a
    # list) so re-resolving the same chunk to the same shelf within a batch
    # collapses to one edge — matters because chunks can arrive multiple times
    # through `iter_chunks` if ES `_doc` ordering shifts across pages mid-scan
    # under concurrent writes, and we'd otherwise inflate `n_edges` past the
    # actual count of edges in the graph (Neo4j MERGE is idempotent so the
    # graph state is right either way, but the audit metrics matter).
    pending_edges: dict[ShelfId, dict[ChunkId, list[str]]] = defaultdict(dict)
    # Per-chunk denorm buffer keyed by chunk_id, same dedupe rationale.
    pending_denorm: dict[ChunkId, tuple[list[ShelfId], list[ThemeId]]] = {}
    pending_edge_count = 0
    seen_chunks: set[ChunkId] = set()  # every chunk we examined (dedupe guard)
    attached_chunks: set[ChunkId] = set()  # subset that actually got >=1 edge
    n_edges = 0
    shelves_with_chunks: set[ShelfId] = set()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        def flush(final: bool = False) -> None:
            nonlocal pending_edge_count, n_edges
            if not pending_edges and not pending_denorm:
                return
            edges_snapshot = [
                (sid, [(cid, lf) for cid, lf in d.items()])
                for sid, d in pending_edges.items()
            ]
            denorm_snapshot = [
                (cid, sids, tids) for cid, (sids, tids) in pending_denorm.items()
            ]
            pending_edges.clear()
            pending_denorm.clear()
            pending_edge_count = 0

            futures = []
            for shelf_id, edges in edges_snapshot:
                if not edges:
                    continue
                shelves_with_chunks.add(shelf_id)
                n_edges += len(edges)
                futures.append(
                    pool.submit(graph_store.attach_chunks_to_shelf, shelf_id, edges)
                )
            if denorm_snapshot:
                futures.append(
                    pool.submit(
                        chunk_store.bulk_update_attachments,
                        denorm_snapshot,
                        wait_for_refresh=final,
                    )
                )
            # Surface backend errors immediately rather than at pool shutdown.
            for f in futures:
                f.result()

        for batch in chunk_store.iter_chunks(batch_size=batch_size):
            for chunk in batch:
                if chunk.chunk_id in seen_chunks:
                    # iter_chunks can yield the same chunk twice if ES `_doc`
                    # ordering shifts mid-scan; ignore the repeat so n_edges
                    # stays honest. (Neo4j MERGE would dedupe anyway, but the
                    # in-flight n_edges counter wouldn't.)
                    continue
                seen_chunks.add(chunk.chunk_id)

                resolutions = resolve_chunk(chunk, index, ontology)
                if not resolutions:
                    continue

                attached_chunks.add(chunk.chunk_id)
                for shelf_id, lifted_from in resolutions.items():
                    pending_edges[shelf_id][chunk.chunk_id] = lifted_from
                    pending_edge_count += 1

                # Authoritative shelf_ids set: layer-A is the sole writer now
                # that clear_attachments() runs at the start, so any old
                # shelf_ids on the chunk are stale by definition.
                new_shelf_ids = sorted(resolutions.keys())
                pending_denorm[chunk.chunk_id] = (
                    new_shelf_ids,
                    list(chunk.theme_ids),
                )

                if pending_edge_count >= batch_size:
                    flush()

        flush(final=True)

    meta = make_artifact_meta(
        phase="attach",
        config=full_config,
        record_count=n_edges,
    )
    _log.info(
        "attach.done",
        n_chunks_attached=len(attached_chunks),
        n_chunks_examined=len(seen_chunks),
        n_edges=n_edges,
        n_shelves_with_chunks=len(shelves_with_chunks),
        artifact_id=meta.artifact_id,
        config_hash=meta.config_hash,
    )
    return meta


__all__ = ["ShelfIndex", "attach", "resolve_chunk"]
