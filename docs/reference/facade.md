# `FoodScholar` — the facade

`FoodScholar` is the single object you work through. It owns the configured stores, the
ontology, the LLM client, and the embedder, and exposes every pipeline phase as a
method. Construct it with [`from_config`](../getting-started/configuration.md) for real
work, or [`in_memory`](../getting-started/quickstart.md) for a zero-setup instance.

The sub-surfaces hang off it as attributes:

- `fs.graph` → [GraphView](graph-view.md) (read/write the graph)
- `fs.ontology` → [FoodOnAPI](ontology.md) (FoodOn lookup)
- `fs.relations` → Layer 0 relation reads (see [Layer 0](../concepts/layer-0-relations.md))
- `fs.viz` → [VizView](viz.md) (renderable views)
- `fs.config` → [FoodScholarConfig](config.md) (the live, mutable config)

```{autoclass} foodscholar.facade.FoodScholar
:members:
:member-order: bysource
```
