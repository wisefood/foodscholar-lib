"""The agentic DFS editor: LLM makes local KEEP/COLLAPSE/REPARENT decisions over
the real FoodOn support DAG, emitting a MethodResult for the bake-off scorecard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.layer_a.bakeoff.agentic.support import rollup_support
from foodscholar.layer_a.bakeoff.agentic.tools import GraphTools
from foodscholar.layer_a.bakeoff.result import MethodResult, home_distance
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.ontology import FoodOnAPI

_log = get_logger("foodscholar.layer_a.bakeoff.agentic")

_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["KEEP", "COLLAPSE", "REPARENT"]},
        "reason": {"type": "string"},
    },
    "required": ["action"],
}


def _lens(tools: GraphTools, node: str, parent: str | None) -> str:
    kids = tools.supported_children(node)
    kid_lines = "\n".join(f"  - {tools.label(c)} ({tools.support(c)} chunks)" for c in kids)
    bridges = tools.relation_targets(node)[:5]
    bridge_lines = "\n".join(f"  - {rn} -> {tools.label(t)}" for _, rn, t in bridges)
    return (
        f"NODE: {tools.label(node)}\n"
        f"PARENT: {tools.label(parent) if parent else '(root)'}\n"
        f"SUPPORT: {tools.support(node)} chunks\n"
        f"CHILDREN:\n{kid_lines or '  (none)'}\n"
        f"RELATIONS:\n{bridge_lines or '  (none)'}\n\n"
        "You are curating a browsable food category tree from FoodOn. Decide this "
        "node's role:\n"
        "- KEEP: it's a recognizable food category worth a shelf.\n"
        "- COLLAPSE: it's redundant with its parent; lift its children up.\n"
        "- REPARENT: it's an organizational artifact; lift its children to the parent.\n"
        'Return JSON {"action": "KEEP|COLLAPSE|REPARENT", "reason": "..."}.'
    )


def build_agentic_result(
    leaf_chunks: dict[str, set[str]],
    ontology: FoodOnAPI,
    *,
    relation_index: dict,
    llm,
    root: str,
    min_support: int = 25,
    max_depth: int = 6,
    max_children: int = 12,
    retriever=None,
) -> MethodResult:
    """Run the DFS editor and return a MethodResult."""
    node_support = {n: len(cs) for n, cs in rollup_support(leaf_chunks, ontology, root=root).items()}
    tools = GraphTools(ontology, relation_index, node_support=node_support,
                       min_support=min_support, retriever=retriever)

    edges: dict[str, list[str]] = {}
    labels: dict[str, str] = {root: tools.label(root)}
    counts: dict[str, int] = {root: node_support.get(root, 0)}
    audit: list[dict] = []
    calls = [0]

    def ask(node: str, parent: str | None) -> str:
        calls[0] += 1
        try:
            obj = llm.generate_json(_lens(tools, node, parent), _ACTION_SCHEMA, max_tokens=256)
        except Exception as exc:
            _log.warning("agentic.action_failed", node=node, error=str(exc))
            obj = {"action": "KEEP"}
        action = (obj or {}).get("action", "KEEP")
        if action not in {"KEEP", "COLLAPSE", "REPARENT"}:
            action = "KEEP"
        audit.append({"node": node, "label": tools.label(node), "action": action,
                      "reason": (obj or {}).get("reason", "")})
        return action

    def visit(node: str, kept_parent: str, depth: int) -> None:
        for child in tools.supported_children(node)[:max_children]:
            action = ask(child, node) if depth < max_depth else "KEEP"
            if action == "KEEP":
                edges.setdefault(kept_parent, []).append(child)
                labels[child] = tools.label(child)
                counts[child] = tools.support(child)
                if depth + 1 < max_depth:
                    visit(child, child, depth + 1)
            else:  # COLLAPSE / REPARENT: skip child as a shelf, lift its children
                if depth + 1 < max_depth:
                    visit(child, kept_parent, depth + 1)

    visit(root, root, 0)

    kept = {root, *(c for kids in edges.values() for c in kids)}
    leaf_home: dict[str, str] = {}
    home_edge_type: dict[str, str] = {}
    home_dist: dict[str, int] = {}
    for leaf in leaf_chunks:
        if leaf not in ontology:
            continue
        cands = [a for a in [leaf, *ontology.id_to_ancestors(leaf)] if a in kept]
        if cands:
            home = max(cands, key=lambda a: len(ontology.id_to_ancestors(a)))
            leaf_home[leaf] = home
            home_edge_type[leaf] = "is-a"
            home_dist[leaf] = home_distance(leaf, home, ontology)
    return MethodResult(
        name="agentic", root=root, edges=edges, labels=labels, counts=counts,
        leaf_home=leaf_home, home_edge_type=home_edge_type, home_distance=home_dist,
        llm_calls=calls[0], audit=audit,
    )
