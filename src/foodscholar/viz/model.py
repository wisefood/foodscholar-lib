"""Renderer-agnostic graph data model.

The viz builder produces a `VizGraph` (typed nodes + edges + hints); the
renderers consume it. Adding a new renderer never changes the builder; adding
a new view never changes the renderers.

Why not just hand renderers our Pydantic `Shelf`/`Theme`/`Entity` directly?
Two reasons:
  - Layout / styling concerns (node color per kind, edge weight visibility)
    don't belong on the domain models.
  - The builder may include synthetic nodes (e.g. a "query" anchor) that
    aren't first-class Pydantic objects.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Visual node "kinds" the renderers know about. Each kind gets a distinct
# color / shape / size in the default style.
NodeKind = Literal[
    "entity",
    "chunk",
    "shelf",
    "theme",
    "card",
    "ontology_term",
    "anchor",
]


class VizNode(BaseModel):
    """A single node in a `VizGraph`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    """Globally unique within the VizGraph (e.g. an OntologyId, ChunkId,
    or 'q::<query>' for the anchor node)."""
    label: str
    """Human-readable text the renderer puts on the node."""
    kind: NodeKind
    weight: float = 1.0
    """Per-kind metric the renderer can map to size — chunk_count for an
    entity, len(chunks) for a shelf, etc."""
    facet: str | None = None
    """Layer A facet hint when relevant (foods / nutrients / …). Renderers
    may map facet → color band."""
    attrs: dict[str, Any] = Field(default_factory=dict)
    """Free-form payload the builder wants the renderer to surface in
    tooltips / titles (ontology_id, chunk text snippet, mention_count, …).
    Must be JSON-serializable."""


class VizEdge(BaseModel):
    """A directed edge between two `VizNode`s."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    target: str
    kind: str
    """Edge relation name. Common values: 'mentions', 'attached_to',
    'has_theme', 'parent_of', 'is_a', 'cites'. Renderers may map kind → color."""
    weight: float = 1.0
    """Per-edge magnitude (confidence, co-mention count) for line thickness."""
    attrs: dict[str, Any] = Field(default_factory=dict)


class VizGraph(BaseModel):
    """A typed graph the renderers consume.

    Layout is the renderer's job — `VizGraph` only describes structure +
    styling hints. Build it via `foodscholar.viz.builder.*` functions, render
    it via `foodscholar.viz.renderers.*`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    nodes: list[VizNode]
    edges: list[VizEdge]
    level: Literal["L0", "L1", "L2", "L3", "L4"]
    """Which abstraction tier the builder produced.

      - L0  corpus-wide entity statistics (no graph, just nodes + per-kind aggregates)
      - L1  entity neighborhood (one entity + 1-hop chunks + co-mentioned entities)
      - L2  shelf / theme view (Layer A/B; populated once those builders land)
      - L3  full backbone (shelves + themes + cards)
      - L4  ontology subtree (FoodOn ancestors / descendants)
    """
    attrs: dict[str, Any] = Field(default_factory=dict)
    """Builder-side annotations (counts, the query that built it, etc.).
    Renderers may show these in a title bar / caption."""

    def __len__(self) -> int:
        """Number of nodes — handy in tests."""
        return len(self.nodes)

    def neighbors(self, node_id: str) -> list[str]:
        """Both-direction adjacency lookup."""
        out: list[str] = []
        for e in self.edges:
            if e.source == node_id:
                out.append(e.target)
            elif e.target == node_id:
                out.append(e.source)
        return out
