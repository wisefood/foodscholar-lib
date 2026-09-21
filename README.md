# foodscholar

**A hierarchical knowledge graph over a corpus of nutrition literature — built for grounded, citable answers.**

FoodScholar chunks dietary guides, textbooks, and scientific abstracts, builds a
**three-layer hierarchical graph** over that corpus, and serves a retrieval API on top.
Every answer traces back to the source chunks that support it.

- **Layer A — Backbone.** A curated, multi-facet menu of *shelves* projected from the
  [FoodOn](https://foodon.org) ontology (foods, health, nutrients, dietary patterns,
  allergies, sustainability).
- **Layer B — Themes.** Fine-grained topic communities discovered **per shelf** by two
  complementary passes — embedding similarity and entity relatedness — then merged.
- **Layer C — Cards.** LLM-generated write-ups for each shelf and theme, with **every
  claim cited back** to source chunks.

Underneath all three sits **Layer 0 — Relations**: typed, ontology-grounded edges
(`olive oil --reduces--> LDL cholesterol`) extracted from chunk text, each carrying the
passages it came from. Opt-in, because extraction costs an LLM pass over the corpus.

📖 **[Full documentation →](https://foodscholar-lib.readthedocs.io)**

## Install

```bash
conda create -n foodscholar python=3.11 -y
conda activate foodscholar
pip install -e '.[dev]'        # extras: llm, elastic, neo4j, clustering, viz,
                               #         annotate, ontology, relations, chunking
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

# Optional: produce the corpus from source PDFs (needs the [chunking] extra).
fs.chunk_documents("data/pdfs/guides", out_dir="data/corpus", source_type="guide")

fs.init(); fs.ingest("data/corpus", nel_dir="data/ner"); fs.embed()
fs.build_entities()
fs.build_relations()              # Layer 0 — opt-in, needs a real LLM
fs.build_layer_a(); fs.attach(); fs.build_layer_b(facet="foods"); fs.build_layer_c()

fs.relations.for_entity("FOODON:03301710")   # what the corpus asserts about olive oil
answer = fs.query("Is olive oil heart-healthy?")
```

[`notebooks/graph_build.ipynb`](notebooks/graph_build.ipynb) is a clean, phase-by-phase
walk-through with an offline (`memory`) and a real (`elastic` + `neo4j`) mode.

## Documentation

| | |
|---|---|
| [Quickstart](docs/getting-started/quickstart.md) · [Configuration](docs/getting-started/configuration.md) | get going, then configure stores/LLM/layers |
| [Architecture](docs/concepts/architecture.md) · [Layers A](docs/concepts/layer-a-backbone.md)/[B](docs/concepts/layer-b-themes.md)/[C](docs/concepts/layer-c-cards.md) | the design and the three layers |
| [Layer 0 — Relations](docs/concepts/layer-0-relations.md) | typed edges under the entity graph |
| [Corpus input](docs/concepts/corpus-input.md) · [Annotation](docs/concepts/annotation.md) | the input format and the NER/linking pipeline |
| [Chunking a corpus](docs/guides/chunking-a-corpus.md) · [Building](docs/guides/building-the-graph.md) · [Exploring](docs/guides/exploring-the-graph.md) · [Visualization](docs/guides/visualization.md) · [Tuning Layer B](docs/guides/tuning-layer-b.md) | task guides |
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

Method-selection provenance lives under `research/` and is not shipped — the Layer A
bake-off (`pytest research/`) and the NER/NEL bake-off harness
(`research/ner_nel_bakeoff/`, which gates any change to the NER or linker defaults).

## Layout

```
src/foodscholar/
├── facade.py        # the FoodScholar facade (entry point)
├── graph_view.py    # fs.graph + Shelf/Theme/Card handles
├── config.py        # Pydantic config + YAML loader
├── io/              # data contracts (Chunk, Shelf, Theme, Card, Entity, Relation)
├── corpus/          # chunker (PDFs/text -> corpus CSVs) + chunk & NEL loading
├── annotate/        # GLiNER / GLiNER2 NER + dense HNSW linking + embeddings
├── relations/       # Layer 0: extract -> dedupe -> ground -> aggregate
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
research/                     # archived method bake-offs (not shipped)
scripts/corpus/               # corpus-prep tools that run upstream of the library
kggen/                        # provenance: the source pipeline this integrates
```
