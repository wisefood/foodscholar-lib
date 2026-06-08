"""Export the Layer A/B/C graph (shelves + themes + cards) to GraphML.

GraphML node/edge attributes must be scalars (str/int/float/bool) — list fields
(e.g. a theme's `keyword_terms`, a card's `cited_chunk_ids`) are flattened to
comma-joined strings. Typed via `node_type` (shelf|theme|card) and `edge_type`
(parent_of|has_theme|has_card). `networkx` is lazy-imported (it ships the
GraphML writer); install via the `[viz]` extra or `pip install networkx`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foodscholar.graph_view import GraphView


def _scalar(value: Any) -> str | int | float | bool:
    """Coerce a value to a GraphML-safe scalar (lists → comma-joined string)."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return "" if value is None else value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    return str(value)


def export_graphml(
    graph: GraphView,
    output: str | Path,
    *,
    facet: str | None = "foods",
) -> Path:
    """Write `graph`'s shelves + themes + cards to `output` as GraphML.

    Nodes: shelves (`node_type="shelf"`), their themes (`"theme"`), and the
    cards on shelves/themes (`"card"`). Edges: shelf→child (`parent_of`),
    shelf→theme (`has_theme`), shelf/theme→card (`has_card`). When `facet` is
    given, only that facet's shelves (and their themes/cards) are exported.
    Returns the output `Path`.
    """
    import networkx as nx

    g = nx.DiGraph()
    out = Path(output)

    shelves = graph.shelves(facet=facet) if facet else graph.shelves()

    def _add_card(owner_id: str, owner) -> None:
        card = owner.card()
        if card is None:
            return
        m = card.model
        g.add_node(
            m.card_id, node_type="card", label=_scalar(m.title),
            target_id=_scalar(m.target_id), target_type=_scalar(m.target_type),
            summary=_scalar(m.summary), evidence_quality=_scalar(m.evidence_quality),
            llm_model=_scalar(m.llm_model), prompt_version=_scalar(m.prompt_version),
            cited_chunk_ids=_scalar(m.cited_chunk_ids),
            safety_flagged=_scalar(m.safety_flagged),
            embedding_model=_scalar(getattr(m, "embedding_model", None)),
        )
        g.add_edge(owner_id, m.card_id, edge_type="has_card")

    for sh in shelves:
        sm = sh.model
        g.add_node(
            sm.shelf_id, node_type="shelf", label=_scalar(sm.label),
            facet=_scalar(sm.facet), depth=_scalar(sm.depth),
            foodon_id=_scalar(sm.foodon_id), chunk_count=_scalar(sm.chunk_count),
            support_direct=_scalar(sm.support_direct),
            support_lifted=_scalar(sm.support_lifted),
        )
        if sm.parent_shelf_id:
            g.add_edge(sm.parent_shelf_id, sm.shelf_id, edge_type="parent_of")
        _add_card(sm.shelf_id, sh)

        for th in sh.themes():
            tm = th.model
            g.add_node(
                tm.theme_id, node_type="theme", label=_scalar(tm.label),
                facet=_scalar(tm.facet), chunk_count=_scalar(tm.chunk_count),
                discovered_by=_scalar(tm.discovered_by),
                discovery_pass=_scalar(tm.discovery_pass),
                keyword_terms=_scalar(tm.keyword_terms),
            )
            g.add_edge(sm.shelf_id, tm.theme_id, edge_type="has_theme")
            _add_card(tm.theme_id, th)

    out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(g, out)
    return out
