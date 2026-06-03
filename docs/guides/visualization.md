# Visualization

`fs.viz` turns a slice of the graph into a renderable artifact. The headline view is the
**interactive Layer A tree**: the full shelf hierarchy on the left, and — when you click
a shelf — its Layer B themes grouped by origin on the right.

## The Layer A tree

```python
out = fs.viz.layer_a_tree("foods").render(
    "tree", output="data/viz/layer_a_tree_foods.html"
)
```

This writes a **self-contained** HTML file (data baked in as JSON, vanilla JS — no
external assets, works offline). The left pane shows every shelf with a
`Chunks: total (D: direct | L: lifted)` badge and a `[theme count]`; sub-threshold
shelves are greyed. Click a shelf to see its themes split into **Merged / Similarity /
Relatedness** with a per-origin filter — the three [discovery passes](../concepts/layer-b-themes.md).

Show it inline in a notebook with an IFrame (the file is written under `data/viz/`,
relative to the notebook in `notebooks/`):

```python
from IPython.display import IFrame
IFrame(src="../data/viz/layer_a_tree_foods.html", width="100%", height=700)
```

```{note}
The tree reads themes straight from the graph, so it reflects whatever the last
`build_layer_b` produced. After a per-shelf rebuild, a shelf shows only the themes
actually built from its chunks. If you see a shelf swamped with off-topic themes, you're
looking at an older global-Pass-1 build — rebuild with `pass1_mode="per_shelf"`.
```

## Other renderers

`fs.viz` also exposes graph-style views over other slices (entity neighbourhoods, the
shelf/theme/card backbone, ontology subtrees), rendered with pluggable backends
(`pyvis`, `cytoscape`, `graphviz`, `matplotlib`). The renderers behind the `[viz]` extra
need their packages installed:

```python
fs.viz.backbone(facet="foods").render("cytoscape", output="backbone.html")
fs.viz.entity_neighborhood("FOODON:03309927").render("pyvis", output="olive.html")
```

Each `render(backend, output=...)` returns the HTML string when `output` is omitted, or
writes the file and returns its path otherwise.
