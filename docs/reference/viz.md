# Visualization

`fs.viz` is a `VizView` — it builds renderable slices of the graph (the interactive Layer
A tree, entity neighbourhoods, the backbone, ontology subtrees) and hands them to a
pluggable renderer. See the [visualization guide](../guides/visualization.md) for usage.

## VizView

```{autoclass} foodscholar.viz.view.VizView
:members:
:member-order: bysource
```

## Renderable graph model

The renderer-agnostic intermediate representation a view produces.

```{autopydantic_model} foodscholar.viz.model.VizGraph
```

```{autopydantic_model} foodscholar.viz.model.VizNode
```

```{autopydantic_model} foodscholar.viz.model.VizEdge
```
