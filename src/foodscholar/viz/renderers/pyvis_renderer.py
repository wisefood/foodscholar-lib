"""Interactive HTML renderer via pyvis (vis.js under the hood).

Pyvis writes a self-contained HTML file with embedded JS so the graph is
panable/zoomable. Returns the HTML as a string when `output=None`, otherwise
writes to disk and returns the path.

Gated by the `[viz]` extra: `pip install 'foodscholar[viz]'`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from foodscholar.viz.model import VizGraph
from foodscholar.viz.renderers.base import Renderer, color_for


class PyvisRenderer(Renderer):
    """Render any `VizGraph` (L1-L4) to an interactive HTML graph."""

    name = "pyvis"

    def __init__(
        self,
        *,
        height: str = "650px",
        width: str = "100%",
        physics: bool = True,
        notebook: bool = True,
    ) -> None:
        try:
            from pyvis.network import Network  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "the 'pyvis' package is required for PyvisRenderer. "
                "Install with: pip install 'foodscholar[viz]'"
            ) from e
        self._Network = Network
        self._height = height
        self._width = width
        self._physics = physics
        self._notebook = notebook

    def render(self, graph: VizGraph, *, output: str | Path | None = None) -> Any:
        if graph.attrs.get("empty_state"):
            # Render a one-node placeholder rather than a broken HTML.
            return self._render_placeholder(graph, output)

        net = self._Network(
            height=self._height,
            width=self._width,
            directed=True,
            notebook=self._notebook,
            cdn_resources="in_line",  # self-contained HTML
            bgcolor="#FAFAFA",
            font_color="#111827",
        )
        # Per-level physics / layout. vis-network has a hierarchical layout
        # built in that's the right tool for L4 (ontology is_a) and L3
        # (shelf/theme backbone) — top-down, no overlap. For L1 / L2 the
        # default force-directed solver with stronger repulsion gives a
        # decent spread without nodes stacking.
        if graph.level in ("L3", "L4"):
            net.set_options('{'
                '"layout": {"hierarchical": {"enabled": true, '
                '   "direction": "UD", "sortMethod": "directed", '
                '   "levelSeparation": 120, "nodeSpacing": 140}},'
                '"physics": {"enabled": false},'
                '"edges": {"smooth": {"type": "cubicBezier"}}}')
        else:
            net.set_options('{'
                '"physics": {"barnesHut": {'
                '   "gravitationalConstant": -8000, '
                '   "springLength": 140, '
                '   "avoidOverlap": 0.5}, "minVelocity": 0.5},'
                '"interaction": {"tooltipDelay": 100}}')

        for node in graph.nodes:
            net.add_node(
                node.id,
                label=node.label[:80],
                title=_tooltip(node),
                color=color_for(node),
                size=_size_for(node),
                shape=_shape_for(node),
            )
        for edge in graph.edges:
            net.add_edge(
                edge.source,
                edge.target,
                title=f"{edge.kind} (weight={edge.weight:.2f})",
                value=edge.weight,
                color={"color": _edge_color(edge.kind), "opacity": 0.7},
                arrows="to",
            )

        # pyvis's `generate_html` is the modern, side-effect-free path; older
        # versions only have `show`/`save_graph` which we shim around.
        if hasattr(net, "generate_html"):
            html = net.generate_html(notebook=self._notebook)
        else:  # pragma: no cover
            html = net.html  # type: ignore[attr-defined]

        if output is None:
            return html
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return out

    def _render_placeholder(
        self, graph: VizGraph, output: str | Path | None
    ) -> Any:
        reason = graph.attrs.get("reason", "no data yet")
        html = (
            f"<div style='padding:24px;font-family:system-ui;"
            f"border:1px dashed #9CA3AF;color:#374151;background:#F9FAFB'>"
            f"<h3 style='margin-top:0'>{graph.title}</h3>"
            f"<p>{reason}</p>"
            f"</div>"
        )
        if output is None:
            return html
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return out


# -------------------------------------------------------------- styling


def _shape_for(node) -> str:  # type: ignore[no-untyped-def]
    return {
        "entity": "dot",
        "chunk": "box",
        "shelf": "diamond",
        "theme": "triangle",
        "card": "star",
        "ontology_term": "ellipse",
        "anchor": "hexagon",
    }.get(node.kind, "dot")


def _size_for(node) -> float:  # type: ignore[no-untyped-def]
    # Scale weight (chunk_count) into [10, 60] roughly. Bigger ≠ better; just
    # informative. log-ish via sqrt so the head doesn't dwarf the tail.
    base = 12.0
    import math

    return min(60.0, base + 4.0 * math.sqrt(max(0.0, node.weight)))


def _tooltip(node) -> str:  # type: ignore[no-untyped-def]
    """HTML-formatted hover tooltip — pyvis renders it inline."""
    lines = [f"<b>{node.label}</b>", f"<i>{node.kind}</i>"]
    if node.facet:
        lines.append(f"facet: {node.facet}")
    for key, value in node.attrs.items():
        if key.endswith("_preview") and isinstance(value, str):
            lines.append(f"{key}: {value[:160]}")
        elif isinstance(value, list):
            lines.append(f"{key}: {', '.join(str(v) for v in value[:5])}")
        elif isinstance(value, (str, int, float, bool)):
            lines.append(f"{key}: {value}")
    return "<br>".join(lines)


def _edge_color(kind: str) -> str:
    return {
        "mentions": "#94A3B8",
        "attached_to": "#16A34A",
        "has_theme": "#EAB308",
        "parent_of": "#22C55E",
        "describes": "#F97316",
        "is_a": "#9333EA",
        "cites": "#0EA5E9",
    }.get(kind, "#94A3B8")
