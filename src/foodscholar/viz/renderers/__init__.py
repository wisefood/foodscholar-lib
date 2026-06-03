"""Pluggable renderers for `VizGraph`s.

All backends are lazy-imported behind the `[viz]` extra. Each is a thin
adapter that takes a `VizGraph` and emits HTML / SVG / a `Figure`.
"""

from foodscholar.viz.renderers.base import Renderer

__all__ = ["Renderer"]


def cytoscape() -> Renderer:
    """Lazy constructor for `CytoscapeRenderer`. Self-contained HTML, no deps."""
    from foodscholar.viz.renderers.cytoscape_renderer import CytoscapeRenderer

    return CytoscapeRenderer()


def pyvis(**kwargs) -> Renderer:  # type: ignore[no-untyped-def]
    """Lazy constructor for `PyvisRenderer`. Needs `[viz]` extra."""
    from foodscholar.viz.renderers.pyvis_renderer import PyvisRenderer

    return PyvisRenderer(**kwargs)


def graphviz(**kwargs) -> Renderer:  # type: ignore[no-untyped-def]
    """Lazy constructor for `GraphvizRenderer`. Needs `[viz]` extra + `dot` binary."""
    from foodscholar.viz.renderers.graphviz_renderer import GraphvizRenderer

    return GraphvizRenderer(**kwargs)


def matplotlib(**kwargs) -> Renderer:  # type: ignore[no-untyped-def]
    """Lazy constructor for `MatplotlibRenderer`. Needs `[viz]` extra."""
    from foodscholar.viz.renderers.matplotlib_renderer import MatplotlibRenderer

    return MatplotlibRenderer(**kwargs)


def tree(**kwargs) -> Renderer:  # type: ignore[no-untyped-def]
    """Lazy constructor for `TreeRenderer`. Self-contained HTML, no deps."""
    from foodscholar.viz.renderers.tree_renderer import TreeRenderer

    return TreeRenderer()
