# foodscholar

[![PyPI](https://img.shields.io/pypi/v/foodscholar.svg)](https://pypi.org/project/foodscholar/)
[![Python](https://img.shields.io/pypi/pyversions/foodscholar.svg)](https://pypi.org/project/foodscholar/)
[![Documentation](https://readthedocs.org/projects/foodscholar-lib/badge/?version=latest)](https://foodscholar-lib.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/wisefood/foodscholar-lib/blob/main/LICENSE)

**The whole pipeline from nutrition PDFs to ranked, citable evidence — chunking, ontology linking, knowledge-graph construction, and hybrid retrieval.**

FoodScholar takes dietary guides, textbooks and scientific abstracts and runs the whole
pipeline: **chunk** the PDFs, **link** every mention to the [FoodOn](https://foodon.org)
ontology, **construct** a three-layer hierarchical graph over the result, and **retrieve**
over it. Every hit traces back to the source chunk it came from.

The library retrieves and stops there — ranked, scored evidence. Formulating an answer
means owning a model, a prompt registry, a citation format and an editorial policy, which
belong to the service asking the question.

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

**Retrieval** (`fs.retrieve()`) is the Extended KG-Gen hybrid: a passage is scored on how
it reads (0.3), on what its extracted triples assert (0.3), and on where it sits in the
entity graph by Personalized PageRank (0.4). The graph branches make it more than vector
search — a passage on "sodium and hypertension" surfaces for "salt and blood pressure"
because the graph connects them, not because the words match. It reads the stores
directly, so there is no index artifact to build, mount or keep in step with the graph.

📖 **[Full documentation →](https://foodscholar-lib.readthedocs.io)**

## Install

```bash
conda create -n foodscholar python=3.11 -y
conda activate foodscholar
pip install -e '.[dev]'        # extras: llm, elastic, neo4j, clustering, viz,
                               #         annotate, ontology, relations, chunking
```

See [docs: Installation](https://foodscholar-lib.readthedocs.io/en/latest/getting-started/installation.html) for the extras matrix and
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

# Retrieval: ranked passages + the branch scores that ranked them.
# The library retrieves; formulating an answer is your pipeline's job.
hits, trace = fs.retrieve("Is olive oil heart-healthy?", k=5)
```

[`notebooks/graph_build.ipynb`](https://github.com/wisefood/foodscholar-lib/blob/main/notebooks/graph_build.ipynb) is a clean, phase-by-phase
walk-through with an offline (`memory`) and a real (`elastic` + `neo4j`) mode.

## Documentation

| | |
|---|---|
| [Quickstart](https://foodscholar-lib.readthedocs.io/en/latest/getting-started/quickstart.html) · [Configuration](https://foodscholar-lib.readthedocs.io/en/latest/getting-started/configuration.html) | get going, then configure stores/LLM/layers |
| [Architecture](https://foodscholar-lib.readthedocs.io/en/latest/concepts/architecture.html) · [Layers A](https://foodscholar-lib.readthedocs.io/en/latest/concepts/layer-a-backbone.html)/[B](https://foodscholar-lib.readthedocs.io/en/latest/concepts/layer-b-themes.html)/[C](https://foodscholar-lib.readthedocs.io/en/latest/concepts/layer-c-cards.html) | the design and the three layers |
| **[Extended KG-Gen](https://foodscholar-lib.readthedocs.io/en/latest/concepts/extended-kg-gen.html)** | **the method: chunk → link → extract → ground → retrieve** |
| [Layer 0 — Relations](https://foodscholar-lib.readthedocs.io/en/latest/concepts/layer-0-relations.html) | typed edges under the entity graph |
| [Retrieval](https://foodscholar-lib.readthedocs.io/en/latest/concepts/retrieval.html) | the three scoring branches, the cost model and its bounds |
| [Corpus input](https://foodscholar-lib.readthedocs.io/en/latest/concepts/corpus-input.html) · [Annotation](https://foodscholar-lib.readthedocs.io/en/latest/concepts/annotation.html) | the input format and the NER/linking pipeline |
| [Chunking a corpus](https://foodscholar-lib.readthedocs.io/en/latest/guides/chunking-a-corpus.html) · [Building](https://foodscholar-lib.readthedocs.io/en/latest/guides/building-the-graph.html) · [Exploring](https://foodscholar-lib.readthedocs.io/en/latest/guides/exploring-the-graph.html) · [Visualization](https://foodscholar-lib.readthedocs.io/en/latest/guides/visualization.html) · [Tuning Layer B](https://foodscholar-lib.readthedocs.io/en/latest/guides/tuning-layer-b.html) | task guides |
| [API reference](https://foodscholar-lib.readthedocs.io/en/latest/reference/index.html) | the public surface, from docstrings |

`config.example.yaml` documents every config field; [`BRIEF.md`](https://github.com/wisefood/foodscholar-lib/blob/main/BRIEF.md) is the
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
├── retrieval/       # Extended KG-Gen hybrid retrieval (fs.retrieve)
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
```
