# API reference

This section is generated from the source docstrings. It documents the **public
surface** — the objects you import and call directly. For the *why* behind them, read
the [Concepts](../concepts/architecture.md); for task recipes, the
[Guides](../guides/building-the-graph.md).

The surface is small and layered:

- **[`FoodScholar`](facade.md)** — the facade. One object that owns the stores, the
  ontology, the LLM, and every build phase (`ingest`, `build_layer_a`, …). Almost
  everything you do starts here.
- **[Configuration](config.md)** — the `FoodScholarConfig` model and its sections,
  with every field, type, and default.
- **[Data model](data-model.md)** — the Pydantic contracts that flow through the
  pipeline: `Chunk`, `Mention`, `EntityLink`, `Shelf`, `Theme`, `Card`, `Entity`.
- **[Graph view](graph-view.md)** — `fs.graph`, the fluent read/write surface and its
  navigation handles.
- **[Ontology](ontology.md)** — `fs.ontology`, the FoodOn lookup API.
- **[Visualization](viz.md)** — `fs.viz`, the renderable graph views.
- **[Storage protocols](storage.md)** — the interfaces every store backend implements.

```{toctree}
:hidden:

facade
config
data-model
graph-view
ontology
viz
storage
```
