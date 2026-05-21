"""Static graph renderer via Graphviz `dot`.

Emits SVG / PNG / dot source for inclusion in docs and papers. Needs the
`graphviz` Python wrapper plus the `dot` binary on `$PATH` (apt-get install
graphviz).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from foodscholar.viz.model import VizGraph
from foodscholar.viz.renderers.base import Renderer, color_for


class GraphvizRenderer(Renderer):
    """Render `VizGraph` to a static image via Graphviz `dot`."""

    name = "graphviz"

    def __init__(self, *, format: str = "svg", engine: str = "dot") -> None:
        try:
            import graphviz  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "the 'graphviz' python package is required for GraphvizRenderer. "
                "Install with: pip install 'foodscholar[viz]'  "
                "(and `apt-get install graphviz` for the dot binary)."
            ) from e
        self._gv = graphviz
        self._format = format
        self._engine = engine

    def render(self, graph: VizGraph, *, output: str | Path | None = None) -> Any:
        dot = self._gv.Digraph(
            comment=graph.title,
            engine=self._engine,
            format=self._format,
        )
        dot.attr("graph", rankdir="LR", bgcolor="#FAFAFA", fontname="Helvetica")
        dot.attr("node", fontname="Helvetica", fontsize="10", style="filled")
        dot.attr("edge", fontname="Helvetica", fontsize="8")

        for n in graph.nodes:
            anchor = bool(n.attrs.get("is_anchor"))
            dot.node(
                n.id,
                label=n.label[:60],
                fillcolor=color_for(n),
                fontcolor=_text_color_for(color_for(n)),
                shape=_shape_for(n.kind),
                penwidth=("3" if anchor else "1"),
                tooltip=_tooltip(n),
            )
        for e in graph.edges:
            dot.edge(
                e.source,
                e.target,
                label=e.kind,
                color=_edge_color(e.kind),
                penwidth=str(min(4.0, 1.0 + e.weight * 0.4)),
            )

        if output is None:
            return dot.pipe(format=self._format)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        rendered = dot.render(filename=str(out.with_suffix("")), cleanup=True)
        return Path(rendered)


# -------------------------------------------------------------- styling


def _shape_for(kind: str) -> str:
    return {
        "entity": "circle",
        "chunk": "box",
        "shelf": "diamond",
        "theme": "triangle",
        "card": "doubleoctagon",
        "ontology_term": "ellipse",
        "anchor": "hexagon",
    }.get(kind, "circle")


def _tooltip(node) -> str:  # type: ignore[no-untyped-def]
    lines = [node.label, f"({node.kind})"]
    if node.facet:
        lines.append(f"facet: {node.facet}")
    return " | ".join(lines)


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


def _text_color_for(bg_hex: str) -> str:
    """Pick black or white text for legibility against `bg_hex` (#RRGGBB)."""
    try:
        r = int(bg_hex[1:3], 16)
        g = int(bg_hex[3:5], 16)
        b = int(bg_hex[5:7], 16)
    except (ValueError, IndexError):
        return "#000000"
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return "#000000" if luminance > 0.55 else "#FFFFFF"
