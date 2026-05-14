# foodscholar

Hierarchical knowledge graph over a corpus of nutrition literature.

FoodScholar builds and serves a **three-layer hierarchical graph** over a
chunked corpus of dietary guides, textbooks, and ~100k scientific abstracts,
and exposes a retrieval API on top:

- **Layer A — Backbone.** Curated, multi-facet semantic menu projected
  from FoodOn (foods, health, sustainability, dietary patterns, allergies,
  nutrients).
- **Layer B — Themes.** Topic communities discovered per shelf via
  embedding-based community detection.
- **Layer C — Write-ups.** LLM-generated cards attached to every shelf and
  theme, with every claim cited back to source chunks.

See [`BRIEF.md`](BRIEF.md) for the full design.

## Status

**v0.1.0 — foundation only.** The public surface is complete; phase
implementations land milestone-by-milestone per BRIEF §12.

## Install

```bash
conda create -n foodscholar python=3.11 -y
conda activate foodscholar
pip install -e '.[dev]'        # add [all] when you need ES/Neo4j/etc.
```

## Quickstart

```python
from foodscholar import FoodScholar
from foodscholar.io.chunk import Chunk

# Zero-config — backed by in-memory stores + mock embedder + mock LLM
fs = FoodScholar.in_memory()

fs.upsert_chunks([
    Chunk(chunk_id="c1", text="Mediterranean diet reduces cardiovascular risk.",
          source_doc_id="d1", source_type="abstract", section_type="abstract"),
])

fs.info()
# {'foodscholar': '0.1.0', 'config_hash': '...', 'chunk_store': 'memory', ...}

# Phase methods raise a clear NotImplementedError until their milestone lands:
# fs.annotate(), fs.build_layer_a(), fs.attach(),
# fs.build_layer_b(), fs.build_layer_c(), fs.build(), fs.query("...")
```

For production use, drive everything from a YAML config:

```python
fs = FoodScholar.from_config("config.yaml")
fs.load_chunks("data/chunks.parquet")
fs.build()
answer = fs.query("Is olive oil heart-healthy?")
```

## Loading the ontology

`fs.ontology` is the FoodOn lookup API used by the linker, the layer_a
backbone projection, and layer_c prompts. First access lazily loads
`cfg.ontology.foodon_path` via pronto (caching to Parquet alongside it):

```python
fs.ontology.name_to_id("olive oil")              # "FOODON:..." | None
fs.ontology.id_to_label("FOODON:03309927")
fs.ontology.id_to_synonyms("FOODON:03309927", include_related=True)
fs.ontology.id_to_ancestors("FOODON:03309927")   # closed transitive set
fs.ontology.id_to_descendants("FOODON:00001002")
fs.ontology.search("olive", limit=25)
```

For tests and notebooks you can skip the loader and pass an in-memory API
directly: `fs.attach_ontology(FoodOnAPI(terms))`.

## Exploring the graph

`fs.graph` is the fluent read/write surface over the graph. Reads return
**handles** that wrap the underlying Pydantic models and add navigation
methods:

```python
fs.graph.add_shelf(shelf_id="s-med", label="Mediterranean diet",
                   facet="dietary_patterns", depth=1)
fs.graph.attach_chunks(["c1", "c2"], shelf="s-med")   # auto-denormalizes

fs.graph.shelves(facet="dietary_patterns")            # list[ShelfHandle]
fs.graph.shelf("s-med").themes()                      # list[ThemeHandle]
fs.graph.shelf("s-med").chunks()                      # list[Chunk]
fs.graph.shelf("s-med").parent()                      # ShelfHandle | None
fs.graph.theme("t-olive").shelves()                   # back-references
fs.graph.theme("t-olive").card().cited_chunks()
fs.graph.search("olive oil", shelf="s-med", k=5)      # hybrid retrieval
fs.graph.summary()                                    # {"shelves": ..., "themes": ...}
```

See [`notebooks/build_graph.ipynb`](notebooks/build_graph.ipynb) for a
phase-by-phase walk-through driven entirely from `fs.graph`.

## CLI

Every CLI command wraps the same facade method:

```bash
foodscholar info        --config config.yaml
foodscholar init        --config config.yaml
foodscholar build-all   --config config.yaml
foodscholar query "..." --config config.yaml
foodscholar version
```

## Testing

```bash
pytest                       # unit tests only
pytest -m integration        # requires docker-compose: ES + Neo4j
ruff check src tests
```

## Layout

```
foodscholar/
├── pyproject.toml
├── config.example.yaml
├── notebooks/build_graph.ipynb     # phase-by-phase walk-through
├── examples/                       # phase walk-throughs
├── tests/
│   ├── unit/                       # run against in-memory stores
│   └── integration/                # real ES + Neo4j (skipped by default)
└── src/foodscholar/
    ├── __init__.py                 # FoodScholar, GraphView, types
    ├── facade.py                   # the FoodScholar facade
    ├── graph_view.py               # fs.graph + Shelf/Theme/Card handles
    ├── config.py                   # Pydantic config + YAML loader
    ├── versioning.py               # config_hash + ArtifactMeta
    ├── logging.py                  # structlog setup
    ├── io/                         # Pydantic data contracts
    ├── corpus/                     # chunk loading
    ├── annotate/                   # NER + linking + embeddings   (stub)
    ├── ontology/                   # FoodOn loader + lookup       (M1 ✓)
    ├── layer_a/                    # backbone builder             (stub)
    ├── layer_b/                    # theme discovery              (stub)
    ├── layer_c/                    # write-up cards               (stub)
    ├── retrieval/                  # public query API             (stub)
    ├── storage/                    # protocols + adapters
    ├── cli/                        # typer entry point (one-line per command)
    └── evaluation/                 # gates + scorers              (stub)
```
