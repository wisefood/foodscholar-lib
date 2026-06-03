"""LLM aliasing pass over Layer A shelves.

Adds a human-friendly ``display_label`` to shelves whose FoodOn ``label`` reads
as jargon, so faceted navigation lands on recognizable names. Additive only: the
shelf's ``label``, ``foodon_id``, and place in the tree are never changed — the
projection stays exactly as faithful as before, only the displayed name improves.

The LLM sees a read-only lens (parent, siblings, children with support, and —
when a relation index is supplied — FoodOn object-property + cross-ontology
relation bridges like ``has defining ingredient`` / ``derives from``) so it can
name a node well without altering structure. Shelves that already carry a
``display_label`` (e.g. from bottom-up grouping) and synthetic facet roots
(no ``foodon_id``) are left untouched.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.io.graph import Shelf
    from foodscholar.ontology import FoodOnAPI

_log = get_logger("foodscholar.layer_a.alias")

_ALIAS_SCHEMA = {
    "type": "object",
    "properties": {
        "alias": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["alias"],
}

# FoodOn object properties most useful for naming a node, best first.
_FOOD_RELATION_RANK = (
    "has defining ingredient",
    "has ingredient",
    "derives from",
    "member of",
    "has food substance analog",
    "has part",
    "has substance added",
    "has quality",
)


def _rank_relations(rels, *, limit: int = 5):
    def key(r):
        name = r[1]
        rank = _FOOD_RELATION_RANK.index(name) if name in _FOOD_RELATION_RANK else len(
            _FOOD_RELATION_RANK
        )
        return (rank, name)

    return sorted(rels, key=key)[:limit]


def _lens(shelf, parent, siblings, children, ontology, relation_index):
    def line(s):
        return f"  - {s.display_label or s.label} ({s.chunk_count} chunks)"

    kid_lines = "\n".join(line(c) for c in children) or "  (none)"
    sib_lines = "\n".join(line(s) for s in siblings) or "  (none)"
    bridge_lines = "  (none)"
    if relation_index is not None and shelf.foodon_id is not None:
        bridges = _rank_relations(relation_index.get(shelf.foodon_id, []))
        bridge_lines = "\n".join(
            f"  - {rn} -> {ontology.id_to_label(t) or t}" for _, rn, t in bridges
        ) or "  (none)"
    parent_name = (parent.display_label or parent.label) if parent else "(facet root)"
    return (
        f"NODE: {shelf.label}\n"
        f"PARENT: {parent_name}\n"
        f"SUPPORT: {shelf.chunk_count} chunks "
        f"(direct {shelf.support_direct} / lifted {shelf.support_lifted})\n"
        f"CHILDREN:\n{kid_lines}\n"
        f"SIBLINGS:\n{sib_lines}\n"
        f"RELATIONS:\n{bridge_lines}\n\n"
        "This is a FIXED category in a browsable food tree — its id, place in the "
        "hierarchy, and chunks will NOT change. Your only job is naming: if the "
        "label is technical / jargon, give a short everyday name for THE SAME "
        "category (use the context above, but do not generalize or narrow the "
        "meaning). If the label is already a recognizable food or food group, "
        'return null.\nReturn JSON {"alias": "..."|null, "reason": "..."}.'
    )


def alias_shelves(
    shelves: list[Shelf],
    ontology: FoodOnAPI,
    *,
    llm,  # type: ignore[no-untyped-def]
    relation_index: dict | None = None,
) -> list[Shelf]:
    """Fill ``display_label`` on jargon-labelled shelves via the LLM lens.

    Mutates and returns ``shelves``. Skips synthetic facet roots (no
    ``foodon_id``) and shelves that already have a ``display_label``. Structure,
    labels, ids, and chunk placement are never touched.
    """
    by_id = {s.shelf_id: s for s in shelves}
    children: dict[str, list[str]] = defaultdict(list)
    for s in shelves:
        if s.parent_shelf_id:
            children[s.parent_shelf_id].append(s.shelf_id)

    aliased = 0
    for shelf in shelves:
        if shelf.foodon_id is None or shelf.display_label:
            continue
        parent = by_id.get(shelf.parent_shelf_id) if shelf.parent_shelf_id else None
        siblings = [
            by_id[c] for c in children.get(shelf.parent_shelf_id, [])
            if c != shelf.shelf_id and c in by_id
        ] if shelf.parent_shelf_id else []
        kids = [by_id[c] for c in children.get(shelf.shelf_id, []) if c in by_id]
        try:
            obj = llm.generate_json(
                _lens(shelf, parent, siblings, kids, ontology, relation_index),
                _ALIAS_SCHEMA,
                max_tokens=256,
            ) or {}
        except Exception as exc:  # a broken judge must not corrupt labels
            _log.warning("alias.judge_failed", shelf=shelf.shelf_id, error=str(exc))
            continue
        alias = obj.get("alias")
        if isinstance(alias, str) and alias.strip() and alias.strip() != shelf.label:
            shelf.display_label = alias.strip()
            aliased += 1

    _log.info("layer_a.alias.done", n_aliased=aliased, n_shelves=len(shelves))
    return shelves
