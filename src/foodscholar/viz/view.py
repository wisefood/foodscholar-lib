"""User-facing handle exposed as `fs.viz` on the facade.

Two pieces:

  - `RenderableGraph` wraps a `VizGraph` and adds a fluent `.render(backend,
    output=...)` so users can chain `fs.viz.entity_neighborhood(id).render("pyvis", ...)`.
  - `VizView` exposes the five builder functions as methods that return
    `RenderableGraph` instances.

Renderers themselves are constructed lazily inside `RenderableGraph.render`
so importing `fs.viz` carries no `[viz]` dependency cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from foodscholar.viz import builder
from foodscholar.viz.model import VizGraph

if TYPE_CHECKING:
    from foodscholar.facade import FoodScholar


RendererName = Literal["pyvis", "cytoscape", "graphviz", "matplotlib", "tree"]


class RenderableGraph:
    """A `VizGraph` plus a `.render(backend, ...)` shortcut.

    The graph is the value most callers want — it's a serializable Pydantic
    model that round-trips through JSON. Use `.graph` to get the underlying
    `VizGraph`; use `.render("pyvis", output=...)` to emit an artifact.
    """

    def __init__(self, graph: VizGraph) -> None:
        self.graph = graph

    @property
    def nodes(self):  # type: ignore[no-untyped-def]
        return self.graph.nodes

    @property
    def edges(self):  # type: ignore[no-untyped-def]
        return self.graph.edges

    @property
    def title(self) -> str:
        return self.graph.title

    @property
    def level(self) -> str:
        return self.graph.level

    def __len__(self) -> int:
        return len(self.graph)

    def __repr__(self) -> str:
        return (
            f"RenderableGraph(level={self.graph.level}, "
            f"nodes={len(self.graph.nodes)}, edges={len(self.graph.edges)}, "
            f"title={self.graph.title!r})"
        )

    def render(
        self,
        backend: RendererName = "pyvis",
        *,
        output: str | Path | None = None,
        **renderer_kwargs: Any,
    ) -> Any:
        """Render with one of the four backends.

        - `"pyvis"`   → interactive HTML (string or file).
        - `"cytoscape"` → self-contained Cytoscape.js HTML.
        - `"graphviz"` → SVG / PNG bytes (or file when `output` is given).
        - `"matplotlib"` → `Figure` (or file when `output` is given). Edges
          are ignored; this is a stats-bars renderer.

        `output` semantics match the renderer — for HTML backends, a `Path`
        is written and returned; for matplotlib, the figure is saved and
        the path returned; for None, the natural in-memory artifact comes
        back.
        """
        from foodscholar.viz import renderers as _renderers

        factory = {
            "pyvis": _renderers.pyvis,
            "cytoscape": _renderers.cytoscape,
            "graphviz": _renderers.graphviz,
            "matplotlib": _renderers.matplotlib,
            "tree": _renderers.tree,
        }
        if backend not in factory:
            raise ValueError(
                f"unknown viz backend {backend!r}; choose from {sorted(factory)}"
            )
        renderer = factory[backend](**renderer_kwargs) if backend != "cytoscape" \
            else factory[backend]()
        return renderer.render(self.graph, output=output)


class VizView:
    """User-facing visualization handle. Exposed as `fs.viz`.

    Each method builds a `VizGraph` for one abstraction level and wraps it in
    a `RenderableGraph` so the caller can immediately chain `.render(...)`.
    """

    def __init__(self, fs: FoodScholar) -> None:
        self._fs = fs

    # ------------------------------------------------------------- L0

    def entity_histogram(
        self,
        *,
        prefix: str | None = None,
        k: int = 30,
    ) -> RenderableGraph:
        """Top-`k` entities by `chunk_count`. Best rendered with matplotlib."""
        return RenderableGraph(builder.entity_histogram(self._fs, prefix=prefix, k=k))

    # ------------------------------------------------------------- L1

    def entity_neighborhood(
        self,
        ontology_id: str,
        *,
        max_chunks: int = 12,
        max_co_entities: int = 25,
    ) -> RenderableGraph:
        """Anchor entity + mentioning chunks + co-mentioned entities."""
        return RenderableGraph(
            builder.entity_neighborhood(
                self._fs,
                ontology_id,
                max_chunks=max_chunks,
                max_co_entities=max_co_entities,
            )
        )

    # ------------------------------------------------------------- L2

    def shelf(
        self,
        shelf_id: str,
        *,
        max_chunks: int = 12,
    ) -> RenderableGraph:
        """One shelf + its themes + chunks (requires Layer A)."""
        return RenderableGraph(
            builder.shelf_view(self._fs, shelf_id, max_chunks=max_chunks)
        )

    # ------------------------------------------------------------- L3

    def backbone(
        self,
        *,
        facet: str | None = None,
        include_cards: bool = True,
    ) -> RenderableGraph:
        """Full Layer A/B/C backbone — shelves, themes, cards."""
        return RenderableGraph(
            builder.backbone(self._fs, facet=facet, include_cards=include_cards)
        )

    def layer_a_tree(self, facet: str = "foods") -> RenderableGraph:
        """Full Layer A shelf tree for a facet, themes grouped by origin.

        Best rendered with the `"tree"` backend:
        `fs.viz.layer_a_tree("foods").render("tree", output="tree.html")`.
        """
        return RenderableGraph(builder.layer_a_tree(self._fs, facet))

    # ------------------------------------------------------------- L4

    def ontology_subtree(
        self,
        ontology_id: str,
        *,
        max_descendants: int = 30,
        include_ancestors: bool = True,
    ) -> RenderableGraph:
        """FoodOn ancestors + descendants of `ontology_id`."""
        return RenderableGraph(
            builder.ontology_subtree(
                self._fs.ontology,
                ontology_id,
                max_descendants=max_descendants,
                include_ancestors=include_ancestors,
            )
        )
