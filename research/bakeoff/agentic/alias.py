"""Aliasing-only pass over a frozen, FoodOn-faithful backbone.

The agentic method does ONE thing: add layperson **aliases** to node labels. It
does **not** edit structure and does **not** reparent anything — not nodes, not
chunks. The backbone's edges, ids, AND its chunk homing (``leaf_home``) are
copied through verbatim from the input (e.g. 1a+), so the result scores
identically on coverage / faithfulness / specificity / findability and differs
only on **nameability**. Aliases are additive (the FOODON id and original
``labels[nid]`` are never touched), so the tree is exactly as faithful as the
backbone while reading more nameably.

The agent keeps its read-only lens (support, supported-children, and FoodOn
object-property relation bridges — ``has defining ingredient``, ``has ingredient``,
``derives from``, ``member of``, …) so it can name a node well — e.g. seeing what
a node derives from or is defined by helps pick a recognizable alias. The lens is
strictly informational: relations shape the *wording*, never the *placement*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bakeoff.result import MethodResult
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from bakeoff.agentic.tools import GraphTools

_log = get_logger("bakeoff.agentic.alias")

_ALIAS_SCHEMA = {
    "type": "object",
    "properties": {
        "alias": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["alias"],
}

# FoodOn object properties most useful for understanding a food node, best first.
# (Relations whose targets leave FoodOn — `in taxon`→NCBITaxon, `has quality`→PATO
# — never reach the lens because the relation index only keeps FOODON→FOODON edges.)
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


def _rank_relations(
    rels: list[tuple[str, str, str]], *, limit: int = 5
) -> list[tuple[str, str, str]]:
    def key(r: tuple[str, str, str]) -> tuple[int, str]:
        name = r[1]
        rank = _FOOD_RELATION_RANK.index(name) if name in _FOOD_RELATION_RANK else len(
            _FOOD_RELATION_RANK
        )
        return (rank, name)

    return sorted(rels, key=key)[:limit]


def _lens(
    tools: GraphTools,
    node: str,
    parent: str | None,
    child_ids: list[str],
    sibling_ids: list[str],
) -> str:
    kid_lines = "\n".join(
        f"  - {tools.label(c)} ({tools.support(c)} chunks)" for c in child_ids
    ) or "  (none)"
    sib_lines = "\n".join(
        f"  - {tools.label(s)} ({tools.support(s)} chunks)" for s in sibling_ids
    ) or "  (none)"
    # Relations include FoodOn object properties AND cross-ontology references
    # (CHEBI, NCBITaxon, PATO, …) — context for naming, never structure.
    bridges = _rank_relations(tools.relation_targets(node))
    bridge_lines = "\n".join(
        f"  - {rn} -> {tools.label(t)}" for _, rn, t in bridges
    ) or "  (none)"
    return (
        f"NODE: {tools.label(node)}\n"
        f"PARENT: {tools.label(parent) if parent else '(root)'}\n"
        f"SUPPORT: {tools.support(node)} chunks\n"
        f"CHILDREN:\n{kid_lines}\n"
        f"SIBLINGS:\n{sib_lines}\n"
        f"RELATIONS:\n{bridge_lines}\n\n"
        "This node is a FIXED category in a browsable food tree — its FoodOn id, "
        "place in the hierarchy, and chunks will NOT change. Your ONLY job is "
        "naming: if the label is technical / jargon, give a short everyday name "
        "for THE SAME category (use the relations above for context, but do not "
        "generalize or narrow the meaning). If the label is already a recognizable "
        "food or food group, return null.\n"
        'Return JSON {"alias": "..."|null, "reason": "..."}.'
    )


def build_aliased_result(
    base: MethodResult,
    *,
    tools: GraphTools,
    llm,  # type: ignore[no-untyped-def]
) -> MethodResult:
    """Return ``base`` with layperson aliases added — nothing else changed.

    Structure (``edges``, ``labels``), chunk homing (``leaf_home``,
    ``home_edge_type``, ``home_distance``), and ``counts`` are copied verbatim;
    only ``aliases`` is populated. No node or chunk is ever moved.
    """
    tree_nodes = {base.root, *base.edges.keys()}
    for kids in base.edges.values():
        tree_nodes.update(kids)
    parent_of: dict[str, str] = {}
    for parent, kids in base.edges.items():
        for child in kids:
            parent_of.setdefault(child, parent)

    aliases: dict[str, str] = {}
    audit: list[dict] = []
    calls = 0
    for node in sorted(tree_nodes):
        if node == base.root:
            continue
        calls += 1
        parent = parent_of.get(node)
        siblings = [c for c in base.edges.get(parent, []) if c != node] if parent else []
        try:
            obj = llm.generate_json(
                _lens(tools, node, parent, list(base.edges.get(node, [])), siblings),
                _ALIAS_SCHEMA,
                max_tokens=256,
            ) or {}
        except Exception as exc:  # a broken judge must not corrupt labels
            _log.warning("alias.judge_failed", node=node, error=str(exc))
            obj = {"alias": None}
        original = base.labels.get(node, node)
        alias = obj.get("alias")
        if isinstance(alias, str) and alias.strip() and alias.strip() != original:
            aliases[node] = alias.strip()
        audit.append({
            "node": node, "label": original,
            "alias": aliases.get(node), "reason": obj.get("reason", ""),
        })

    return MethodResult(
        name="agentic",
        root=base.root,
        edges=base.edges,
        labels=dict(base.labels),
        counts=dict(base.counts),
        leaf_home=dict(base.leaf_home),
        home_edge_type=dict(base.home_edge_type),
        home_distance=dict(base.home_distance),
        aliases=aliases,
        llm_calls=calls,
        audit=audit,
    )
