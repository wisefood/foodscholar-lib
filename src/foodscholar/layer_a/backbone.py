"""Layer A '1a+' projection: backbone-first controlled expansion.

The method validated in the bake-off, now official. Instead of the support-driven
prune cascade, we pick a **backbone** (the supported children of the facet root —
e.g. the direct children of `food product` for foods) and then **progressively
expand** each backbone node down its real FoodOn descendants, under browse caps:

  - a child is kept only if its rolled-up support clears `min_support`;
  - single-child filing tiers with no direct chunks of their own are **collapsed**
    (e.g. `Corn -> maize kernel(0) -> corn kernel(83)` shows `corn kernel` under
    `Corn`) — faithful, since the kept node is still a real is-a descendant;
  - each node is placed under exactly **one** parent (FoodOn is a DAG), so a node
    never appears twice;
  - empty dead-ends (no descendants, no direct chunks) are **pruned**;
  - fan-out and depth are capped.

All displayed nodes are real FoodOn ids; the tree is is-a faithful. Counts use the
support table: `chunk_count` = rolled-up distinct chunks, with `support_direct` /
`support_lifted` split for the D/L badge.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from foodscholar.layer_a.prune import shelf_id_for_foodon

if TYPE_CHECKING:
    from foodscholar.config import _ResolvedFacetConfig
    from foodscholar.io.graph import Facet, Shelf
    from foodscholar.layer_a.propagate import SupportTable
    from foodscholar.ontology import FoodOnAPI

# FoodOn anchor each facet's backbone hangs under. Foods = "food product".
_FACET_ROOT_ID = {"foods": "FOODON:00001002"}


def _resolve_root(ontology: FoodOnAPI, facet: Facet, configured: str | None) -> str | None:
    if configured:
        return configured if configured in ontology else None
    rid = _FACET_ROOT_ID.get(facet)
    return rid if rid and rid in ontology else None


def build_backbone_shelves(
    support: SupportTable,
    ontology: FoodOnAPI,
    config: _ResolvedFacetConfig,
    facet: Facet,
    *,
    root_id: str | None = None,
    max_children: int = 12,
) -> list[Shelf]:
    """Project `support` into shelves via backbone-first controlled expansion.

    Returns [] when no backbone clears `min_support` (caller falls back to a stub).
    """
    from foodscholar.io.graph import Shelf

    min_support = config.min_support
    max_depth = config.max_depth
    blocked = {t.lower().strip() for t in config.blacklist_terms}

    def node_support(fid: str) -> int:
        return support.with_descendants.get(fid, 0)

    def direct(fid: str) -> int:
        return support.direct.get(fid, 0)

    def allowed(fid: str) -> bool:
        lbl = (ontology.id_to_label(fid) or "").lower().strip()
        return lbl not in blocked

    def supported_children(fid: str) -> list[str]:
        return [
            c for c in ontology.id_to_children(fid)
            if c in ontology and allowed(c) and node_support(c) >= min_support
        ]

    def resolve_filing_tier(fid: str) -> str:
        seen: set[str] = set()
        while fid not in seen and direct(fid) == 0 and len(supported_children(fid)) == 1:
            seen.add(fid)
            fid = supported_children(fid)[0]
        return fid

    def display_children(fid: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for child in supported_children(fid):
            resolved = resolve_filing_tier(child)
            if resolved not in seen:
                seen.add(resolved)
                out.append(resolved)
        out.sort(key=node_support, reverse=True)
        return out[:max_children]

    # Backbone = supported children of the facet root; else top-level supported
    # terms (those with no supported strict ancestor) so non-foods facets still work.
    root = _resolve_root(ontology, facet, root_id)
    if root is not None:
        backbone = sorted(supported_children(root), key=node_support, reverse=True)
    else:
        supported = {t for t in support.with_descendants
                     if node_support(t) >= min_support and allowed(t)}
        backbone = sorted(
            (t for t in supported
             if not any(a in supported for a in ontology.id_to_ancestors(t))),
            key=node_support, reverse=True,
        )
    if not backbone:
        return []

    # Single-parent DFS: first placement wins (FoodOn DAG -> tree).
    placed: set[str] = set()
    parent_of: dict[str, str | None] = {}
    order: list[str] = []

    def expand(parent: str, depth: int) -> None:
        if depth >= max_depth:
            return
        for child in display_children(parent):
            if child in placed:
                continue
            placed.add(child)
            parent_of[child] = parent
            order.append(child)
            expand(child, depth + 1)

    for b in backbone:
        if b in placed:
            continue
        placed.add(b)
        parent_of[b] = None
        order.append(b)
        expand(b, 1)

    # Prune empty dead-ends (no kept descendants AND no direct chunks), cascading.
    children_of: dict[str, list[str]] = defaultdict(list)
    for n in order:
        p = parent_of[n]
        if p is not None:
            children_of[p].append(n)
    kept = set(order)
    changed = True
    while changed:
        changed = False
        for n in list(kept):
            if not children_of.get(n) and direct(n) == 0:
                kept.discard(n)
                p = parent_of.get(n)
                if p is not None and n in children_of.get(p, []):
                    children_of[p].remove(n)
                changed = True

    # Projection depth: backbone roots = 0, children = parent+1 (clamped).
    depth_of: dict[str, int] = {}
    for n in order:  # order is parent-before-child
        if n not in kept:
            continue
        p = parent_of[n]
        depth_of[n] = 0 if p is None else min(depth_of.get(p, 0) + 1, max_depth)

    shelves: list[Shelf] = []
    for n in order:
        if n not in kept:
            continue
        wd = node_support(n)
        d = direct(n)
        p = parent_of[n]
        shelves.append(Shelf(
            shelf_id=shelf_id_for_foodon(n),
            label=ontology.id_to_label(n) or n,
            facet=facet,
            depth=depth_of[n],
            foodon_id=n,
            parent_shelf_id=shelf_id_for_foodon(p) if p else None,
            chunk_count=wd,
            support_direct=d,
            support_lifted=max(wd - d, 0),
            see_also=[],
        ))
    return sorted(shelves, key=lambda s: (s.depth, s.label.lower(), s.shelf_id))
