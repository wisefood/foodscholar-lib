"""Stats renderer via matplotlib.

Renders the *node-only* projection of a `VizGraph` as a horizontal bar
chart sorted by `weight`. Edges are ignored — this is the right tool for
L0 (entity histograms) and for getting a quick magnitude picture of any
other level. For real graph drawing, use the Pyvis / Cytoscape / Graphviz
renderers instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from foodscholar.viz.model import VizGraph
from foodscholar.viz.renderers.base import Renderer, color_for


class MatplotlibRenderer(Renderer):
    """Bar chart over `VizGraph.nodes` sorted by `weight`."""

    name = "matplotlib"

    def __init__(self, *, max_bars: int = 25, figsize: tuple[float, float] = (8, 6)) -> None:
        try:
            import matplotlib  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "the 'matplotlib' package is required for MatplotlibRenderer. "
                "Install with: pip install 'foodscholar[viz]'"
            ) from e
        self._max_bars = max_bars
        self._figsize = figsize

    def render(self, graph: VizGraph, *, output: str | Path | None = None) -> Any:
        import matplotlib.pyplot as plt

        # Sort by weight desc and trim to max_bars; pick a reasonable label.
        bars = sorted(graph.nodes, key=lambda n: n.weight, reverse=True)[: self._max_bars]
        if not bars:
            fig, ax = plt.subplots(figsize=self._figsize)
            ax.text(0.5, 0.5, graph.title + "\n(no data)", ha="center", va="center")
            ax.set_axis_off()
            return self._save_or_return(fig, output)

        labels = [_truncate(b.label, 30) + f"\n{b.id}" for b in bars]
        weights = [b.weight for b in bars]
        colors = [color_for(b) for b in bars]

        fig, ax = plt.subplots(figsize=self._figsize)
        bars_obj = ax.barh(range(len(bars)), weights, color=colors)
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()  # largest at top
        ax.set_xlabel("weight (e.g. chunk_count)")
        ax.set_title(graph.title)

        # Annotate weights at the end of each bar.
        for bar, w in zip(bars_obj, weights, strict=True):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {int(w) if w == int(w) else round(w, 2)}",
                va="center",
                fontsize=7,
            )
        fig.tight_layout()
        return self._save_or_return(fig, output)

    @staticmethod
    def _save_or_return(fig, output: str | Path | None) -> Any:  # type: ignore[no-untyped-def]
        if output is None:
            return fig
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
        return out


def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"
