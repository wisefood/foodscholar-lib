"""Visualization layer for foodscholar.

Three pieces:

  - `model.VizGraph` — renderer-agnostic intermediate representation of a
    graph slice (nodes + edges + layout/styling hints).
  - `builder` — functions that turn the foodscholar stores into a `VizGraph`
    at five abstraction levels (L0 stats → L4 ontology subtree).
  - `renderers` — pluggable backends (pyvis / cytoscape / graphviz /
    matplotlib) that turn a `VizGraph` into HTML / SVG / a `Figure`.

Usage via the facade:

    fs.viz.entity_neighborhood("FOODON:03309927").render("pyvis", output="olive.html")
    fs.viz.entity_histogram(prefix="FOODON", k=20).render("matplotlib")

Or directly:

    from foodscholar.viz import builder, renderers
    graph = builder.entity_histogram(fs, k=20)
    renderers.matplotlib().render(graph, output="entities.png")
"""

from foodscholar.viz.model import VizEdge, VizGraph, VizNode
from foodscholar.viz.view import VizView

__all__ = ["VizEdge", "VizGraph", "VizNode", "VizView"]
