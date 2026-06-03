# foodscholar

**A hierarchical knowledge graph over a corpus of nutrition literature — built for grounded, citable answers.**

FoodScholar ingests dietary guides, textbooks, and scientific abstracts, builds a
**three-layer hierarchical graph** over the chunked corpus, and serves a retrieval API
on top. Every answer traces back to the source chunks that support it.

- **Layer A — Backbone.** A curated, multi-facet menu of *shelves* projected from the
  [FoodOn](https://foodon.org) ontology (foods, health, nutrients, dietary patterns,
  allergies, sustainability).
- **Layer B — Themes.** Fine-grained topic communities discovered **per shelf** by two
  complementary passes — embedding similarity and entity relatedness — then merged.
- **Layer C — Cards.** LLM-generated write-ups for each shelf and theme, with **every
  claim cited back** to source chunks.

📖 **[Full documentation →](https://foodscholar-lib.readthedocs.io)**

## Install

```bash
conda create -n foodscholar python=3.11 -y
conda activate foodscholar
pip install -e '.[dev]'        # add extras (llm, elastic, neo4j, clustering, viz) as needed
```

See [docs: Installation](docs/getting-started/installation.md) for the extras matrix and
local services.

## Quickstart

```python
from foodscholar import FoodScholar
from foodscholar.io.chunk import Chunk

# Zero-config: in-memory stores + mock embedder + mock LLM. No services, no keys.
fs = FoodScholar.in_memory()

fs.upsert_chunks([
    Chunk(chunk_id="c1", text="Mediterranean diet reduces cardiovascular risk.",
          source_doc_id="d1", source_type="abstract", section_type="abstract"),
])
fs.info()
```

For a real build, drive everything from a YAML config and run the phases:

```python
fs = FoodScholar.from_config("config.yaml")
fs.init(); fs.ingest("data/corpus", nel_dir="data/ner"); fs.embed()
fs.build_layer_a(); fs.attach(); fs.build_layer_b(facet="foods"); fs.build_layer_c()
answer = fs.query("Is olive oil heart-healthy?")
```

[`notebooks/graph_build.ipynb`](notebooks/graph_build.ipynb) is a clean, phase-by-phase
walk-through with an offline (`memory`) and a real (`elastic` + `neo4j`) mode.

## Documentation

| | |
|---|---|
| [Quickstart](docs/getting-started/quickstart.md) · [Configuration](docs/getting-started/configuration.md) | get going, then configure stores/LLM/layers |
| [Architecture](docs/concepts/architecture.md) · [Layers A](docs/concepts/layer-a-backbone.md)/[B](docs/concepts/layer-b-themes.md)/[C](docs/concepts/layer-c-cards.md) | the design and the three layers |
| [Corpus input](docs/concepts/corpus-input.md) · [Annotation](docs/concepts/annotation.md) | the input format and the NER/linking pipeline |
| [Building](docs/guides/building-the-graph.md) · [Exploring](docs/guides/exploring-the-graph.md) · [Visualization](docs/guides/visualization.md) · [Tuning Layer B](docs/guides/tuning-layer-b.md) | task guides |
| [API reference](docs/reference/index.md) | the public surface, from docstrings |

`config.example.yaml` documents every config field; [`BRIEF.md`](BRIEF.md) is the
original design brief.

## Testing

Run in the `foodscholar` conda env (Python 3.11):

```bash
conda activate foodscholar
pytest                       # unit tests
pytest -m integration        # requires docker-compose: ES + Neo4j
ruff check src tests
```

> The `base` env's older NumPy can be incompatible with newer Pythons — always use the `foodscholar` env.

Method-selection provenance (the Layer A bake-off harness) lives under `research/` and
is not shipped; run it with `pytest research/`.

## Layout

```
src/foodscholar/
├── facade.py        # the FoodScholar facade (entry point)
├── graph_view.py    # fs.graph + Shelf/Theme/Card handles
├── config.py        # Pydantic config + YAML loader
├── io/              # data contracts (Chunk, Shelf, Theme, Card, Entity)
├── corpus/          # chunk + NEL loading
├── annotate/        # GLiNER NER + dense HNSW linking + embeddings
├── ontology/        # FoodOn loader + lookup (FoodOnAPI)
├── llm/             # provider-agnostic LLM client + fallback chain
├── layer_a/         # backbone projection + aliasing
├── layer_b/         # per-shelf theme discovery (two passes + merge)
├── layer_c/         # cited write-up cards
├── retrieval/       # query API
├── storage/         # protocols + memory / elastic / neo4j adapters
├── viz/             # renderable graph views (incl. the interactive tree)
├── cli/             # typer entry point
└── evaluation/      # gates + scorers
```
```
notebooks/graph_build.ipynb   # phase-by-phase build + interactive tree
docs/                         # Sphinx docs (published on Read the Docs)
research/                     # archived method bake-off (not shipped)
```
