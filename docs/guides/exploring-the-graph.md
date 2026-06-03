# Exploring the graph

`fs.graph` is the fluent read/write surface over the graph. Reads return **handles**
that wrap the underlying Pydantic models and add navigation methods, so you can hop
around the structure without writing queries.

## Reading

```python
fs.graph.shelves(facet="dietary_patterns")     # list[ShelfHandle]
fs.graph.shelf("s-med").themes()               # list[ThemeHandle]
fs.graph.shelf("s-med").chunks()               # list[Chunk]
fs.graph.shelf("s-med").parent()               # ShelfHandle | None
fs.graph.shelf("s-med").children()             # list[ShelfHandle]

fs.graph.theme("t-olive").shelves()            # back-references to shelves
fs.graph.theme("t-olive").card().cited_chunks()

fs.graph.summary()                             # {"shelves": ..., "themes": ...}
```

A handle exposes the model's fields directly (`shelf.label`, `shelf.chunk_count`,
`theme.discovery_pass`) and adds traversal methods (`.parent()`, `.children()`,
`.themes()`, `.chunks()`, `.card()`). Reach the raw Pydantic model any time via
`handle.model`.

## Writing

The same surface builds the graph by hand — handy in tests and notebooks:

```python
fs.graph.add_shelf(shelf_id="s-med", label="Mediterranean diet",
                   facet="dietary_patterns", depth=1)
fs.graph.attach_chunks(["c1", "c2"], shelf="s-med")   # auto-denormalizes shelf_ids
fs.graph.add_theme(theme_id="t-olive", label="Olive oil", shelf_ids=["s-med"],
                   discovered_by="leiden", discovery_version="v0",
                   facet="dietary_patterns", discovery_pass="global_similarity")
```

`attach_chunks` requires exactly one target — `shelf=` **or** `theme=`, not both — and
keeps the Elasticsearch denormalization (`shelf_ids` / `theme_ids`) in lockstep with the
Neo4j edges automatically.

## Retrieval

```python
fs.graph.search("olive oil", shelf="s-med", k=5)   # hybrid BM25 + kNN, shelf-filtered
```

This is the retrieval path from [](../concepts/architecture.md): hybrid search over
Elasticsearch, optionally scoped to a shelf or theme.

```{tip}
Everything here works identically on the in-memory backend, so you can prototype graph
walks against `FoodScholar.in_memory()` with no services running.
```
