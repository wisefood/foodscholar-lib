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


# Two-stage layout: every page first runs the level-specific layout, then a
# noOverlap-like pass via cytoscape's `boundingBox`-aware `preset` re-layout
# is unnecessary — cose / concentric / breadthfirst all accept overlap
# parameters and respect them. We additionally call `cy.nodes().forEach` to
# nudge any pair that's still on top of one another by half a node size after
# layoutstop. All cytoscape-core layouts, no extensions.
_HTML_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; font-family: system-ui; }}
  #cy {{ width: 100%; height: calc(100vh - 40px); background: #FAFAFA; }}
  header {{ padding: 8px 16px; background: #F3F4F6; border-bottom: 1px solid #E5E7EB; }}
  header h3 {{ margin: 0; font-size: 14px; color: #111827; }}
  header span {{ font-size: 12px; color: #6B7280; margin-left: 8px; }}
  #cy-error {{ padding: 24px; font-family: system-ui; color: #B91C1C; display: none; }}
</style>
<script src="{cyto_js}"></script>
</head><body>
<header><h3>{title}</h3><span>{n_nodes} nodes · {n_edges} edges · level {level}</span></header>
<div id="cy"></div>
<div id="cy-error"></div>
<script>
const elements = {elements_json};
const layoutConfig = {layout_json};

// concentric() takes a node and returns a numeric "rank" — higher = closer
// to center. Cytoscape's `concentric` layout doesn't accept a string here,
// only a function, so we ship one that reads from the data the Python side
// stamps on each node.
if (layoutConfig.name === 'concentric') {{
  layoutConfig.concentric = function(node) {{
    return node.data('concentric_rank') || 0;
  }};
  layoutConfig.levelWidth = function() {{ return 1; }};
}}

function initCy() {{
  if (typeof cytoscape === 'undefined') {{
    document.getElementById('cy-error').style.display = 'block';
    document.getElementById('cy-error').textContent =
      'cytoscape.js failed to load — check the network or your iframe sandbox.';
    return;
  }}
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
    layout: layoutConfig,
  }});

  // Post-layout: fit viewport to all nodes, then do one O(n^2) overlap-nudge
  // sweep so concentric / breadthfirst rings with tied ranks don't stack.
  cy.on('layoutstop', () => {{
    nudgeOverlaps(cy);
    cy.fit(undefined, 30);
  }});

  cy.on('tap', 'node', evt => {{
    const d = evt.target.data();
    alert(d.tooltip || d.label);
  }});
}}

function nudgeOverlaps(cy) {{
  // Push pairs of nodes that share the same position. Cheap O(n^2) sweep
  // — fine for graphs up to a few hundred nodes which is all our builder
  // produces.
  const nodes = cy.nodes();
  const minGap = 12;
  for (let i = 0; i < nodes.length; i++) {{
    for (let j = i + 1; j < nodes.length; j++) {{
      const a = nodes[i].position();
      const b = nodes[j].position();
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const wanted = (nodes[i].width() + nodes[j].width()) / 2 + minGap;
      if (dist < wanted && dist > 0) {{
        const push = (wanted - dist) / 2;
        const ux = dx / dist;
        const uy = dy / dist;
        nodes[i].position({{ x: a.x - ux * push, y: a.y - uy * push }});
        nodes[j].position({{ x: b.x + ux * push, y: b.y + uy * push }});
      }} else if (dist === 0) {{
        // Two nodes exactly co-located — separate horizontally by `wanted`.
        nodes[j].position({{ x: a.x + wanted, y: a.y }});
      }}
    }}
  }}
}}

if (document.readyState === 'loading') {{
  document.addEventListener('DOMContentLoaded', initCy);
}} else {{
  initCy();
}}
</script>
</body></html>
"""


# How to translate a `VizGraph.level` into a cytoscape-core layout config.
# Each level picks the layout that best surfaces its structure:
#   L0  grid          — disconnected nodes, no need for force-directed
#   L1  concentric    — anchor centered, then chunks ring, then co-entity ring
#   L2  cose (tuned)  — small mixed-shape graph, force-directed
#   L3  breadthfirst  — Layer A/B/C hierarchy, roots at top
#   L4  breadthfirst  — ontology is_a tree, root at top
def _layout_for_level(level: str) -> dict[str, Any]:
    base_cose = {
        "name": "cose",
        "animate": False,
        "randomize": True,
        "nodeOverlap": 25,            # was 12 — push nodes apart harder
        "nodeRepulsion": 20000,       # the actual force; default 2048
        "idealEdgeLength": 110,
        "padding": 40,
        "componentSpacing": 80,
        "fit": True,
    }
    if level == "L0":
        return {
            "name": "grid",
            "rows": 0,                # auto rows
            "padding": 30,
            "fit": True,
            "avoidOverlap": True,
            "avoidOverlapPadding": 20,
        }
    if level == "L1":
        return {
            "name": "concentric",
            "padding": 40,
            "fit": True,
            "minNodeSpacing": 40,
            "spacingFactor": 1.5,
            "avoidOverlap": True,
        }
    if level == "L4" or level == "L3":
        return {
            "name": "breadthfirst",
            "directed": True,
            "padding": 40,
            "fit": True,
            "spacingFactor": 1.4,
            "avoidOverlap": True,
            "grid": False,
        }
    # L2 (or anything unrecognized) → tuned cose
    return base_cose


class CytoscapeRenderer(Renderer):
    """Render any `VizGraph` (L1-L4) to a self-contained Cytoscape.js page."""

    name = "cytoscape"

    def render(self, graph: VizGraph, *, output: str | Path | None = None) -> Any:
        elements = self._to_elements(graph)
        html = _HTML_TEMPLATE.format(
            title=_html_escape(graph.title),
            cyto_js=_CYTOSCAPE_CDN,
            elements_json=json.dumps(elements),
            layout_json=json.dumps(_layout_for_level(graph.level)),
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
                    # `concentric` layout (L1) reads this — higher = closer to
                    # the center. Anchor entity at the heart, chunks in the
                    # first ring, co-entities outermost. Layouts that don't
                    # use it ignore the field.
                    "concentric_rank": _concentric_rank(n),
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


def _concentric_rank(node) -> int:  # type: ignore[no-untyped-def]
    """Numeric rank used by L1's `concentric` layout. Higher = inner ring.

    - Anchor entity (is_anchor=True)  → 3 (center)
    - Chunks                          → 2 (middle ring)
    - Other entities (co-mentioned)   → 1 (outer ring)
    - Anything else                   → 0
    """
    if node.kind == "entity" and node.attrs.get("is_anchor"):
        return 3
    if node.kind == "chunk":
        return 2
    if node.kind == "entity":
        return 1
    return 0


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
