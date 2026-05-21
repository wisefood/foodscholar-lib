"""Pruning passes for Layer A.

Order of operations:

  1. blacklist            (before threshold so chunks lift to surviving ancestors)
  2. umbrella rule        (structural: direct-share < X AND lifted-share > Y →
                           FoodOn organizational classifiers caught by data,
                           not by name)
  3. whitelist exception  (marks terms immune to steps 1, 2, AND 4)
  4. threshold            (with-descendants metric)
  5. depth cap            (LIFT to nearest surviving ancestor at depth <= cap)
  6. single-child collapse (iterate to fixed point; record see_also on survivor)

Each step assumes earlier ones have run. Reversing 1<->4 leaks chunks under
blacklisted intermediates and inflates parents. Reversing 5<->6 creates
collapses depth-cap would've prevented. The umbrella rule sits between
blacklist and whitelist because both are categorical "drop unless explicitly
kept" passes — putting them adjacent keeps the keep/drop logic in one band.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.io.graph import Shelf, ShelfId

if TYPE_CHECKING:
    from foodscholar.config import _ResolvedFacetConfig
    from foodscholar.io.graph import Facet
    from foodscholar.layer_a.propagate import SupportTable
    from foodscholar.ontology import FoodOnAPI


def shelf_id_for_foodon(term_id: str) -> ShelfId:
    """Stable shelf id for a FOODON term. Public so tests + attach can reuse."""
    return f"foodon:{term_id}"


def prune(
    support: SupportTable,
    ontology: FoodOnAPI,
    config: _ResolvedFacetConfig,
    facet: Facet,
) -> list[Shelf]:
    """Run the full prune cascade and produce final Shelf records."""
    blocked_labels = {term.lower().strip() for term in config.blacklist_terms}
    whitelist = set(config.whitelist)

    # Step 1+2+3+4: figure out which term ids survive blacklist + umbrella +
    # threshold. Whitelist bypasses all three.
    direct_share_max = config.umbrella_direct_share_max
    lifted_share_min = config.umbrella_lifted_share_min
    umbrella_enabled = direct_share_max > 0.0

    survivors: set[str] = set()
    for term_id, count_wd in support.with_descendants.items():
        if term_id in whitelist:
            # Whitelisted terms bypass blacklist, umbrella, AND threshold.
            survivors.add(term_id)
            continue
        if not _is_allowed(term_id, ontology.id_to_label(term_id), blocked_labels):
            continue
        if umbrella_enabled and count_wd > 0:
            direct = support.direct.get(term_id, 0)
            lifted = max(count_wd - direct, 0)
            direct_share = direct / count_wd
            lifted_share = lifted / count_wd
            if direct_share < direct_share_max and lifted_share > lifted_share_min:
                # Umbrella class — nobody mentions it directly enough relative
                # to its descendant-aggregated support to be worth a shelf.
                continue
        if count_wd < config.min_support:
            continue
        survivors.add(term_id)

    # Whitelist additions that weren't in the support table at all — keep them
    # with 0 chunk_count (they're scaffolding).
    for wl_id in whitelist:
        if wl_id in ontology and wl_id not in survivors:
            label = ontology.id_to_label(wl_id)
            if _is_allowed(wl_id, label, blocked_labels):
                survivors.add(wl_id)

    if not survivors:
        return []

    cap = config.max_depth

    # Build initial shelves keyed by term_id with parent_shelf_id set;
    # projection-relative depth is assigned in a second pass below.
    shelf_by_id: dict[str, Shelf] = {}
    parent_term_by_id: dict[str, str | None] = {}
    for term_id in sorted(survivors):
        direct = support.direct.get(term_id, 0)
        with_descendants = support.with_descendants.get(term_id, 0)
        lifted = max(with_descendants - direct, 0)
        parent_term = _nearest_included_ancestor(
            ontology, term_id, survivors, max_depth=cap
        )
        parent_term_by_id[term_id] = parent_term
        shelf_by_id[term_id] = Shelf(
            shelf_id=shelf_id_for_foodon(term_id),
            label=ontology.id_to_label(term_id) or term_id,
            facet=facet,
            depth=0,  # provisional — set in projection BFS below.
            foodon_id=term_id,
            parent_shelf_id=shelf_id_for_foodon(parent_term) if parent_term else None,
            chunk_count=with_descendants,
            support_direct=direct,
            support_lifted=lifted,
            see_also=[],
        )

    # Assign projection-relative depth (not ontology depth): roots get 0,
    # children get parent.depth + 1, capped at `cap`. This is what users
    # navigate — "how many clicks from a root am I" — and it's stable under
    # the umbrella rule (which kills ontology-intermediate ancestors).
    shelf_by_id = _assign_projection_depth(shelf_by_id, parent_term_by_id, cap)

    # Step 5: single-child collapse, iterated to fixed point.
    if config.collapse_single_child_chains:
        shelf_by_id = _collapse_single_children(shelf_by_id, ontology)

    return sorted(
        shelf_by_id.values(),
        key=lambda s: (s.depth, s.label.lower(), s.shelf_id),
    )


# ---------------------------------------------------------------- helpers


def _assign_projection_depth(
    shelf_by_id: dict[str, Shelf],
    parent_term_by_id: dict[str, str | None],
    cap: int,
) -> dict[str, Shelf]:
    """Set each shelf's `depth` to its projection-relative depth.

    Roots (no parent in projection) get 0; everyone else gets `parent.depth + 1`,
    clamped at `cap`. Topological — handled by iterating until depths stabilize.
    Cycle detection via a visit counter (FoodOn shouldn't have cycles, but if
    a survivor set produces one via the lifted-ancestor walk, we cap iteration
    at 2*|shelves| and treat anything still unresolved as a root).
    """
    depth_by_id: dict[str, int] = {}
    max_iters = 2 * len(shelf_by_id) + 1
    pending = set(shelf_by_id.keys())
    for _ in range(max_iters):
        if not pending:
            break
        resolved_this_pass: set[str] = set()
        for tid in pending:
            parent_term = parent_term_by_id.get(tid)
            if parent_term is None:
                depth_by_id[tid] = 0
                resolved_this_pass.add(tid)
            elif parent_term in depth_by_id:
                depth_by_id[tid] = min(depth_by_id[parent_term] + 1, cap)
                resolved_this_pass.add(tid)
        if not resolved_this_pass:
            # Unresolved cycle / dangling — treat remaining as roots.
            for tid in pending:
                depth_by_id[tid] = 0
            break
        pending -= resolved_this_pass

    return {
        tid: shelf.model_copy(update={"depth": depth_by_id.get(tid, 0)})
        for tid, shelf in shelf_by_id.items()
    }


def _is_allowed(term_id: str, label: str | None, blocked: set[str]) -> bool:
    if term_id.lower() in blocked:
        return False
    return not (label is not None and label.lower().strip() in blocked)


def _depth(
    ontology: FoodOnAPI,
    term_id: str,
    seen: frozenset[str] = frozenset(),
) -> int:
    """Minimum path length from any root. Cycle-safe (FoodOn shouldn't have
    cycles but defensive)."""
    if term_id in seen:
        return 0
    parents = [p for p in ontology.id_to_parents(term_id) if p in ontology]
    if not parents:
        return 0
    next_seen = seen | {term_id}
    return 1 + min(_depth(ontology, p, next_seen) for p in parents)


def _nearest_included_ancestor(
    ontology: FoodOnAPI,
    term_id: str,
    included: set[str],
    *,
    max_depth: int,
) -> str | None:
    """Pick the deepest surviving ancestor at ontology-depth <= max_depth.

    Why ancestors, not BFS-via-parents: FoodOn is a DAG and `id_to_parents`
    only returns direct is_a parents — equivalent-class axioms and parents
    from related ontologies (UBERON, NCBITaxon) can be on the parent list but
    get filtered out by `prefix_filter=[FOODON:]`. The resulting parent index
    is too sparse to find a real ancestor via BFS in cases like cow milk
    (parents=[animal milk → UBERON dead-end] and one cow-substance branch that
    doesn't reach food product). `id_to_ancestors` carries the closed
    transitive set from pronto's `t.superclasses()`, which captures every
    is_a / subClassOf path — so it sees `food product` as an ancestor of cow
    milk even when no parent chain in the loaded subset does.

    Among the surviving ancestors at depth <= max_depth, return the deepest
    (the one closest to the term being parented). Ties broken by id for
    determinism.
    """
    candidates: list[tuple[int, str]] = []
    for ancestor_id in ontology.id_to_ancestors(term_id):
        if ancestor_id not in ontology or ancestor_id not in included:
            continue
        d = _depth(ontology, ancestor_id)
        if d <= max_depth:
            candidates.append((d, ancestor_id))
    if not candidates:
        return None
    # Deepest first, then alphabetical id for tie-break.
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def _collapse_single_children(
    shelf_by_id: dict[str, Shelf],
    ontology: FoodOnAPI,
) -> dict[str, Shelf]:
    """Collapse shelves whose only surviving child carries all their meaning.

    A shelf is collapsed when (a) it has exactly one surviving child in the
    current shelf set AND (b) the child's parent_shelf_id points back to this
    shelf (so we're collapsing within a real edge, not a sibling-of-uncle
    situation). The shelf's `foodon_id` is recorded on the child's `see_also`,
    and the child's parent edge is rewired to the shelf's parent.

    Iterates to fixed point.
    """
    current = dict(shelf_by_id)

    while True:
        # Build parent -> set(child term_id) over the current shelf set.
        children_by_parent: dict[str, list[str]] = {}
        for tid, shelf in current.items():
            if shelf.parent_shelf_id is None:
                continue
            parent_term = _term_from_shelf_id(shelf.parent_shelf_id)
            if parent_term is None:
                continue
            children_by_parent.setdefault(parent_term, []).append(tid)

        collapsed_any = False
        # Walk shelves in deterministic order.
        for parent_term in sorted(children_by_parent.keys()):
            children = children_by_parent[parent_term]
            if len(children) != 1:
                continue
            if parent_term not in current:
                continue
            parent_shelf = current[parent_term]
            child_term = children[0]
            child_shelf = current[child_term]

            # Collapse: child inherits parent's parent edge + see_also.
            new_see_also = list(child_shelf.see_also)
            if parent_shelf.foodon_id and parent_shelf.foodon_id not in new_see_also:
                new_see_also.append(parent_shelf.foodon_id)
            new_see_also.extend(
                fid for fid in parent_shelf.see_also if fid not in new_see_also
            )

            current[child_term] = child_shelf.model_copy(
                update={
                    "parent_shelf_id": parent_shelf.parent_shelf_id,
                    "depth": max(parent_shelf.depth, 0),
                    "see_also": new_see_also,
                }
            )
            del current[parent_term]
            collapsed_any = True
            break  # restart loop — set of children changes

        if not collapsed_any:
            break

    return current


def _term_from_shelf_id(shelf_id: str) -> str | None:
    """Reverse `shelf_id_for_foodon` — returns the foodon term id or None for
    non-foodon shelves (stub roots etc.)."""
    if shelf_id.startswith("foodon:"):
        return shelf_id[len("foodon:") :]
    return None
