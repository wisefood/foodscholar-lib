# Graph view

`fs.graph` is a `GraphView` — the fluent read/write surface over the graph. Reads return
**handles** that wrap the underlying models ([`Shelf`](data-model.md),
[`Theme`](data-model.md), [`Card`](data-model.md)) and add navigation methods. See the
[exploring-the-graph guide](../guides/exploring-the-graph.md) for task-oriented usage.

## GraphView

```{autoclass} foodscholar.graph_view.GraphView
:members:
:member-order: bysource
```

## Handles

```{autoclass} foodscholar.graph_view.ShelfHandle
:members:
:member-order: bysource
```

```{autoclass} foodscholar.graph_view.ThemeHandle
:members:
:member-order: bysource
```

```{autoclass} foodscholar.graph_view.CardHandle
:members:
:member-order: bysource
```
