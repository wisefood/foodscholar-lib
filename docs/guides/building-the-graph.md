# Building the graph

This is the end-to-end pipeline that turns a corpus into the three-layer graph. Each
step is a method on the facade; each writes to the configured stores.

```python
from foodscholar import FoodScholar

fs = FoodScholar.from_config("config.yaml")

fs.init()                          # 1. provision stores (idempotent)
fs.ingest("data/corpus", nel_dir="data/ner")   # 2. load chunks + attach NEL annotations
fs.embed()                         # 3. chunk-text embeddings (Layer B Pass 1 + kNN)
fs.build_entities()                # 4. dedupe entity links into first-class entities
fs.build_layer_a()                 # 5. FoodOn-projected backbone shelves (+ aliasing)
fs.attach()                        # 6. attach chunks to shelves (writes shelf_ids)
fs.build_layer_b(facet="foods")    # 7. per-shelf theme discovery
fs.build_layer_c()                 # 8. cited write-up cards
```

| Step | Method | Produces | Concept |
|---|---|---|---|
| 1 | `init` | empty indices + constraints | — |
| 2 | `ingest` | chunks + mentions + entity links | [Corpus input](../concepts/corpus-input.md) |
| 3 | `embed` | 768-d chunk vectors | — |
| 4 | `build_entities` | `(:Entity)` nodes | [Annotation](../concepts/annotation.md) |
| 5 | `build_layer_a` | `(:Shelf)` hierarchy | [Layer A](../concepts/layer-a-backbone.md) |
| 6 | `attach` | `(:Chunk)-[:ATTACHED_TO]->(:Shelf)` | [Layer A](../concepts/layer-a-backbone.md) |
| 7 | `build_layer_b` | `(:Theme)` + `THEME_OF` | [Layer B](../concepts/layer-b-themes.md) |
| 8 | `build_layer_c` | `(:Card)` | [Layer C](../concepts/layer-c-cards.md) |

After a Layer A or Layer B build, a cross-store **audit** runs the consistency
invariants (see [](../concepts/architecture.md)); a failing critical invariant fails the
build.

## The reference notebook

[`notebooks/graph_build.ipynb`](https://github.com/wisefood/foodscholar-lib/blob/main/notebooks/graph_build.ipynb)
is a clean, phase-by-phase version of exactly this pipeline, with a `BACKEND` toggle:

- `BACKEND = "memory"` — fully offline: loads `data/annotated.parquet` (build it once
  with `scripts/make_annotated_parquet.py`), no Elasticsearch or Neo4j required.
- `BACKEND = "elastic"` — the real stores at `localhost:9200` / `localhost:7687`.

It ends by rendering the interactive Layer A tree — see [](visualization.md).

```{tip}
Run tests and builds in the `foodscholar` conda env (Python 3.11). See
[](../getting-started/installation.md).
```
