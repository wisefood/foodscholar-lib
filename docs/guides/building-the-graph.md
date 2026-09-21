# Building the graph

This is the end-to-end pipeline that turns a directory of documents into a queryable
graph. Each step is a method on the facade; each writes to the configured stores.

```python
from foodscholar import FoodScholar

fs = FoodScholar.from_config("config.yaml")

# 0. Produce the corpus from PDFs. Runs before ingest, not part of build().
fs.chunk_documents("data/pdfs/guides", out_dir="data/corpus", source_type="guide")

fs.init()                          # 1. provision stores (idempotent)
fs.ingest("data/corpus", nel_dir="data/ner")   # 2. load chunks + attach NEL annotations
fs.embed()                         # 3. chunk-text embeddings (Layer B Pass 1 + kNN)
fs.build_entities()                # 4. dedupe entity links into first-class entities
fs.build_relations()               # 5. Layer 0 typed relations — OPT-IN, see below
fs.build_layer_a()                 # 6. FoodOn-projected backbone shelves (+ aliasing)
fs.attach()                        # 7. attach chunks to shelves (writes shelf_ids)
fs.build_layer_b(facet="foods")    # 8. per-shelf theme discovery
fs.build_layer_c()                 # 9. cited write-up cards

hits, trace = fs.retrieve("Is olive oil heart-healthy?", k=5)   # 10. query it
```

| Step | Method | Produces | Concept |
|---|---|---|---|
| 0 | `chunk_documents` | corpus CSVs from PDFs | [Chunking a corpus](chunking-a-corpus.md) |
| 1 | `init` | empty indices + constraints | — |
| 2 | `ingest` | chunks + mentions + entity links | [Corpus input](../concepts/corpus-input.md) |
| 3 | `embed` | 768-d chunk vectors | — |
| 4 | `build_entities` | `(:Entity)` nodes | [Annotation](../concepts/annotation.md) |
| 5 | `build_relations` | `(:Entity)-[:RELATED]->(:Entity)` | [Layer 0](../concepts/layer-0-relations.md) |
| 6 | `build_layer_a` | `(:Shelf)` hierarchy | [Layer A](../concepts/layer-a-backbone.md) |
| 7 | `attach` | `(:Chunk)-[:ATTACHED_TO]->(:Shelf)` | [Layer A](../concepts/layer-a-backbone.md) |
| 8 | `build_layer_b` | `(:Theme)` + `THEME_OF` | [Layer B](../concepts/layer-b-themes.md) |
| 9 | `build_layer_c` | `(:Card)` | [Layer C](../concepts/layer-c-cards.md) |
| 10 | `retrieve` | ranked passages + branch scores | [Retrieval](../concepts/retrieval.md) |

## Step 5 is opt-in, and skipping it costs you two of three retrieval branches

`build_relations()` is gated behind `relations.enabled: true` because it costs an LLM
pass over the whole corpus. It is the only optional step, and the consequence of
skipping it is easy to miss: [retrieval](../concepts/retrieval.md) scores a passage on
its text (0.3), on its extracted triples (0.3) and on Personalized PageRank over the
entity graph (0.4). **Without Layer 0 the last two branches have nothing to read**, stay
silent, and `fs.retrieve()` degrades to plain kNN.

That degradation is quiet by design — a corpus mid-build should answer rather than fail —
so it will not announce itself. `trace.branches_used` reports which branches actually
contributed, and `trace.relations` is 0 when Layer 0 is missing.

```{warning}
Check `relations.store.backend` before running it. It defaults to `memory`, which means
a build on defaults writes Layer 0 into the builder's process and loses it at exit. Set
it to `elastic` for anything a service will later read.
```

## Steps 0 and 5 sit outside `build()`

`fs.build()` runs steps 1–9, and calls `build_relations()` only when
`relations.enabled` is true. `chunk_documents()` is never part of it: it produces the
corpus that `ingest()` consumes, so it runs first and separately.

After a Layer A or Layer B build, a cross-store **audit** (`fs.audit()`) runs the
consistency invariants (see [](../concepts/architecture.md)); a failing critical
invariant fails the build.

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
