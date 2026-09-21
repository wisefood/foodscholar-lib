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

### Entities and relations

Two sibling namespaces sit beside `fs.graph`, one per layer of the entity graph.
They return plain Pydantic records rather than handles:

```python
fs.entities.get("FOODON:03301710")             # Entity | None
fs.entities.chunks_for("FOODON:03301710")      # chunks that mention it
fs.entities.search("olive")                    # BM25-ish over label + synonyms

fs.relations.for_entity("FOODON:03301710")     # Layer 0 edges touching it (both directions)
fs.relations.for_entity("FOODON:03301710", direction="out")
fs.relations.for_chunks(["tb_0421"])           # what one passage asserts — one round-trip
fs.relations.by_predicate("reduces")
fs.relations.grounded()                        # only edges with both endpoints real ids
fs.relations.predicates()                      # [(predicate, count), ...] — watch this sprawl
fs.relations.summary()                         # {"relations", "fully_grounded", ...}
```

`fs.relations` is empty until `fs.build_relations()` has run — see
[Layer 0](../concepts/layer-0-relations.md).

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

## Scoped chunk search

```python
fs.graph.search("olive oil", shelf="s-med", k=5)   # BM25 + kNN, shelf-filtered
```

This is chunk-store search — BM25 and kNN over Elasticsearch, optionally scoped to one
shelf or theme. It is the tool for *exploring* a branch of the graph, and it is **not**
`fs.retrieve()`: it does not read Layer 0, so it has no triple-similarity or PageRank
branch, and it takes no advantage of the entity graph.

Reach for `fs.graph.search()` when you want to look inside a known shelf, and for
[`fs.retrieve()`](../concepts/retrieval.md) when you have a question and want the best
evidence across the corpus.

```{tip}
Everything here works identically on the in-memory backend, so you can prototype graph
walks against `FoodScholar.in_memory()` with no services running.
```
