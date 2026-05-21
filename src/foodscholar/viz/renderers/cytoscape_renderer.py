"""Interactive HTML renderer via Cytoscape.js.

Emits a self-contained HTML file that embeds Cytoscape.js from a CDN. No
Python dependencies beyond the standard library — the renderer ships the JS
template as a string and inlines the graph data as JSON. Suitable for the
notebook (via `IPython.display.HTML`) or as a standalone artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from foodscholar.viz.model import VizGraph
from foodscholar.viz.renderers.base import Renderer, color_for

_CYTOSCAPE_CDN = "https://unpkg.com/cytoscape@3.30.0/dist/cytoscape.min.js"
_LAYOUT_CDN = "https://unpkg.com/cytoscape-cose-bilkent@4.1.0/cytoscape-cose-bilkent.js"


_HTML_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; font-family: system-ui; }}
  #cy {{ width: 100%; height: 92vh; background: #FAFAFA; }}
  header {{ padding: 8px 16px; background: #F3F4F6; border-bottom: 1px solid #E5E7EB; }}
  header h3 {{ margin: 0; font-size: 14px; color: #111827; }}
  header span {{ font-size: 12px; color: #6B7280; margin-left: 8px; }}
</style>
<script src="{cyto_js}"></script>
<script src="{layout_js}"></script>
</head><body>
<header><h3>{title}</h3><span>{n_nodes} nodes · {n_edges} edges · level {level}</span></header>
<div id="cy"></div>
<script>
const elements = {elements_json};
const cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: elements,
  style: [
    {{ selector: 'node', style: {{
        'background-color': 'data(color)',
        'label': 'data(label)',
        'color': '#111827',
        'font-size': 10,
        'text-valign': 'bottom',
        'text-margin-y': 6,
        'text-wrap': 'wrap',
        'text-max-width': 100,
        'width': 'data(size)',
        'height': 'data(size)',
        'border-color': '#1F2937',
        'border-width': 1,
    }}}},
    {{ selector: 'node[?anchor]', style: {{ 'border-width': 3, 'border-color': '#DC2626' }} }},
    {{ selector: 'edge', style: {{
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'line-color': 'data(color)',
        'target-arrow-color': 'data(color)',
        'width': 'data(width)',
        'opacity': 0.7,
    }}}},
  ],
  layout: {{ name: 'cose-bilkent', animate: false, randomize: true, nodeRepulsion: 7000, idealEdgeLength: 90 }},
}});
cy.on('tap', 'node', evt => {{
  const d = evt.target.data();
  console.log(d.tooltip);
  alert(d.tooltip || d.label);
}});
</script>
</body></html>
"""


class CytoscapeRenderer(Renderer):
    """Render any `VizGraph` (L1-L4) to a self-contained Cytoscape.js page."""

    name = "cytoscape"

    def render(self, graph: VizGraph, *, output: str | Path | None = None) -> Any:
        elements = self._to_elements(graph)
        html = _HTML_TEMPLATE.format(
            title=_html_escape(graph.title),
            cyto_js=_CYTOSCAPE_CDN,
            layout_js=_LAYOUT_CDN,
            elements_json=json.dumps(elements),
            n_nodes=len(graph.nodes),
            n_edges=len(graph.edges),
            level=graph.level,
        )
        if output is None:
            return html
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return out

    @staticmethod
    def _to_elements(graph: VizGraph) -> list[dict[str, Any]]:
        elements: list[dict[str, Any]] = []
        for n in graph.nodes:
            import math

            size = min(60.0, 12.0 + 4.0 * math.sqrt(max(0.0, n.weight)))
            elements.append({
                "data": {
                    "id": n.id,
                    "label": n.label[:80],
                    "kind": n.kind,
                    "color": color_for(n),
                    "size": size,
                    "anchor": bool(n.attrs.get("is_anchor")),
                    "tooltip": _tooltip(n),
                },
            })
        for e in graph.edges:
            elements.append({
                "data": {
                    "id": f"{e.source}->{e.target}::{e.kind}",
                    "source": e.source,
                    "target": e.target,
                    "kind": e.kind,
                    "weight": e.weight,
                    "width": max(1.0, min(6.0, e.weight)),
                    "color": _edge_color(e.kind),
                },
            })
        return elements


# -------------------------------------------------------------- helpers


def _html_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _tooltip(node) -> str:  # type: ignore[no-untyped-def]
    lines = [f"{node.label} ({node.kind})"]
    if node.facet:
        lines.append(f"facet: {node.facet}")
    for key, value in node.attrs.items():
        if key.endswith("_preview") and isinstance(value, str):
            lines.append(f"{key}: {value[:160]}")
        elif isinstance(value, list):
            lines.append(f"{key}: {', '.join(str(v) for v in value[:5])}")
        elif isinstance(value, (str, int, float, bool)):
            lines.append(f"{key}: {value}")
    return "\\n".join(lines)


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
