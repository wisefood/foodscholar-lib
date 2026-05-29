"""Renderer abstraction: take a `VizGraph`, emit some output."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from foodscholar.viz.model import VizGraph


class Renderer(ABC):
    """Common base for everything that turns a `VizGraph` into an artifact.

    `render(graph, output=None)` returns whatever the renderer naturally
    produces:

      - HTML renderers (cytoscape, pyvis): return the HTML string when
        `output` is None; write it to disk and return the path otherwise.
      - Graphviz: returns an SVG/PNG bytes blob or writes to `output`.
      - Matplotlib: returns the `Figure` when `output` is None; saves
        otherwise.

    Renderers must NOT raise just because a particular level doesn't make
    sense for them (e.g. matplotlib doesn't draw graphs). They emit a
    coherent best-effort artifact and document the limitation in the
    docstring.
    """

    name: str = "base"

    @abstractmethod
    def render(self, graph: VizGraph, *, output: str | Path | None = None) -> Any: ...


# Default visual styling — a small lookup the renderers share so the four
# backends produce comparable images.

KIND_COLORS: dict[str, str] = {
    "entity": "#4F46E5",        # indigo
    "chunk": "#0EA5E9",         # sky
    "shelf": "#16A34A",         # green
    "theme": "#EAB308",         # yellow
    "card": "#F97316",          # orange
    "ontology_term": "#9333EA", # purple
    "anchor": "#6B7280",        # gray
}

FACET_COLORS: dict[str, str] = {
    "foods": "#16A34A",
    "nutrients": "#0EA5E9",
    "health": "#DC2626",
    "dietary_patterns": "#9333EA",
    "allergies": "#F97316",
    "sustainability": "#84CC16",
}


def color_for(node) -> str:  # type: ignore[no-untyped-def]
    """Best-effort node color: facet wins if known, kind otherwise."""
    if node.facet and node.facet in FACET_COLORS:
        return FACET_COLORS[node.facet]
    return KIND_COLORS.get(node.kind, "#6B7280")
