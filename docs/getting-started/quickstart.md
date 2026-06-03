# Quickstart

## Zero-config, in-memory

The fastest way to see the shape of the API. `FoodScholar.in_memory()` wires up
in-memory stores, a deterministic mock embedder, and a mock LLM — no services, no
model downloads, no API keys.

```python
from foodscholar import FoodScholar
from foodscholar.io.chunk import Chunk

fs = FoodScholar.in_memory()

fs.upsert_chunks([
    Chunk(
        chunk_id="c1",
        text="Mediterranean diet reduces cardiovascular risk.",
        source_doc_id="d1",
        source_type="abstract",
        section_type="abstract",
    ),
])

fs.info()
# {'foodscholar': '0.1.0', 'config_hash': '...', 'chunk_store': 'memory', ...}
```

Everything you do against `fs.graph`, `fs.ontology`, `fs.viz`, and the build phases
works the same way regardless of which stores back them — the in-memory backend is
just the zero-setup default.

## A real build from a config

For a real corpus and persistent stores, drive everything from a YAML config (see
[](configuration.md)) and run the phases:

```python
fs = FoodScholar.from_config("config.yaml")

fs.init()                 # provision the stores (idempotent)
fs.ingest("data/corpus", nel_dir="data/ner")   # load corpus + annotations
fs.embed()                # chunk-text embeddings (for Layer B Pass 1 + kNN)
fs.build_entities()       # dedupe entity links into first-class entities
fs.build_layer_a()        # FoodOn-projected backbone shelves
fs.attach()               # attach chunks to shelves
fs.build_layer_b(facet="foods")   # per-shelf theme discovery
fs.build_layer_c()        # cited write-up cards

answer = fs.query("Is olive oil heart-healthy?")
```

```{tip}
[`notebooks/graph_build.ipynb`](https://github.com/wisefood/foodscholar-lib/blob/main/notebooks/graph_build.ipynb)
is a clean, phase-by-phase walk-through of exactly this build, with a `BACKEND`
toggle to run it fully offline (in-memory + a parquet snapshot) or against real
Elasticsearch + Neo4j.
```

## What's next

- Understand *why* it's built this way — the three layers and two stores (Concepts, coming soon).
- Explore the result — `fs.graph` handles and the interactive Layer A tree (Guides, coming soon).
- Tune Layer B coverage — the per-shelf Pass-1 knobs (Guides, coming soon).
