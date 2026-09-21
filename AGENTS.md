# AGENTS.md

Guidance for AI agents and skills working with **foodscholar**. Keep this current; it's
the fast path to being productive without reading the whole codebase.

## What this is

A Python library that builds a **three-layer hierarchical knowledge graph** over a
chunked corpus of nutrition literature and serves a retrieval API. Every answer is
traceable to the source chunks that support it.

- **Layer 0 — Relations:** typed `(:Entity)-[:RELATED {predicate}]->(:Entity)` edges
  extracted from chunk text and grounded through the linker. Opt-in.
- **Layer A — Backbone:** FoodOn-projected *shelves* (the browsable menu).
- **Layer B — Themes:** per-shelf topic communities (two passes: embedding similarity +
  entity relatedness, then merged).
- **Layer C — Cards:** cited LLM write-ups for shelves/themes.

Two stores: **Elasticsearch** (retrieval — BM25 + HNSW kNN) and **Neo4j** (navigation —
shelf hierarchy, theme membership). A `memory` backend implements the same protocols, so
everything runs in-process with no services.

## Environment & commands

- **Always use the `foodscholar` conda env (Python 3.11).** The `base` env's older NumPy
  is incompatible with newer Pythons and fails to import — running tests there gives
  misleading `numpy ... source directory` errors. This trips people up constantly.

```bash
conda activate foodscholar            # Python 3.11 — REQUIRED for tests/builds
pip install -e '.[dev]'               # core dev; add extras as needed (see below)

pytest tests/unit -q                  # unit tests — the gate (run in foodscholar env)
pytest -m integration                 # needs docker-compose: Elasticsearch + Neo4j
pytest research/                       # archived bake-off tests (not the gate)
ruff check src tests                  # lint gate (line-length 100, py311)
```

Extras: `llm` (anthropic/openai/groq/gemini/ollama), `elastic`, `neo4j`, `clustering`
(leidenalg/igraph/sklearn), `viz` (pyvis/graphviz/matplotlib), `ontology` (pronto),
`relations` (semhash/inflect), `chunking` (docling/nltk).
Local services: `docker compose up -d elasticsearch neo4j`.

Docs build (Sphinx, `foodscholar` env): `pip install -r docs/requirements.txt && sphinx-build -b html -W docs docs/_build/html`.

## Using the library (API cheat-sheet)

```python
from foodscholar import FoodScholar

fs = FoodScholar.in_memory()          # in-memory stores + MOCK embedder + MOCK LLM
fs = FoodScholar.from_config("config.yaml")   # or a dict, or a FoodScholarConfig

# Build phases (run in order; each writes to the configured stores):
fs.chunk_documents("data/pdfs", out_dir="data/corpus", source_type="guide")  # optional
fs.init(); fs.ingest("data/corpus", nel_dir="data/ner"); fs.embed()
fs.build_entities(); fs.build_relations()      # Layer 0 — opt-in, needs a real LLM
fs.build_layer_a(); fs.attach()
fs.build_layer_b(facet="foods"); fs.build_layer_c()
fs.query("Is olive oil heart-healthy?")

# Sub-surfaces:
fs.graph.shelves(facet="foods"); fs.graph.shelf(id).themes()    # read/write graph (handles)
fs.relations.for_entity("FOODON:03301710")                       # Layer 0 edges
fs.ontology.id_to_ancestors("FOODON:03310029")                  # FoodOn lookup
fs.viz.layer_a_tree("foods").render("tree", output="tree.html") # interactive tree
fs.config.layer_b.pass1_mode = "per_shelf"                      # config is live + mutable
```

CLI mirrors the phases: `foodscholar {init|ingest|chunk-corpus|build-relations|build-layer-a|attach|build-layer-b|build-layer-c|build-all|query|info|version} --config config.yaml`.

Config: `from_config` takes a YAML path / dict / `FoodScholarConfig`; `${ENV}` substitution
runs over all forms; `extra="forbid"` (a typo'd key raises). Only `corpus` is required.

## Non-obvious facts (read before changing things)

- **NER = GLiNER; linker = single-tier *dense* (HNSW + BioLORD).** There is no
  lexical/fuzzy/LLM tier — the old 4-tier linker and `keyword`/`agentic` NER were removed.
  Don't reintroduce "tier 1/2" language.
- **Two embedders on purpose:** chunk *text* → BGE-base (retrieval, Layer B Pass 1);
  entity *surfaces* → BioLORD (linking). Not interchangeable.
- **Layer A method:** `projection="backbone"` (the "1a+" backbone projection) is the
  production default + an LLM aliasing pass. `prune` is a kept fallback.
- **Layer B production mode is `pass1_mode="per_shelf"`.** Themes attach to their
  **origin shelf** (not the union of member chunks' shelves). The `discovery_pass` value
  `global_similarity` is a historical name — in per-shelf mode it means *per-shelf*.
- **Only the `foods` facet is populated** (FoodOn is food-only); the other five facets are
  scaffolded.
- **`in_memory()` uses a MOCK LLM + mock embedder.** Don't trust theme labels / cards /
  Pass-1 themes from it — wire a real provider via config for meaningful output.
- **ES 9.4 strips `dense_vector` from `_source`** even with a correct mapping; embeddings
  are read back via the `fields` API. Validate a read round-trip, not just the index.
- **Groq reasoning models (`openai/gpt-oss-*`) return empty** via `GroqClient.generate`.
  Use `llama-3.1-8b-instant` or `llama-3.3-70b-versatile` for labels.
- **Chunk ids are fresh UUIDs.** Re-chunking a document orphans every relation,
  attachment and card citing the old ids — `chunk_documents` refuses to overwrite an
  existing corpus without `force=True`. Set `chunking.chunk_id_strategy="content_hash"`
  on new corpora to make re-chunking idempotent.
- **`chunk_metadata` differs by source type.** Guides carry 9 keys, textbooks only
  `file`/`heading`/`page_number`, abstracts bibliographic ones. Read it via
  `chunk.provenance` (typed, normalizes the `DOI`/`doi` split), not raw `.get()`.
  `page_number` is the chunk's **first** page, not its span.
- **Layer 0 relations are opt-in** (`relations.enabled`) — two LLM calls per chunk.
  `build_relations` refuses to persist under the mock LLM; use `dry_run=True`.
  Relation ids are content-addressed, so re-runs are true upserts, and the store is
  the resume log (no savepoint files).
- **The relation extractor constrains endpoints to an enum** of the step-1 entities.
  Do not "simplify" that schema to a plain string — it is what keeps triples aligned
  with the extracted entities.
- **GLiNER2 (`annotate.ner: "gliner2"`) is opt-in, not better by default.** It trades
  ~12% recall for ~40% precision (~28% fewer mentions), and Layer A support counts key
  off mention volume. Measure downstream before flipping.
- **`research/`** holds the archived Layer A method bake-off — provenance, not shipped, not
  in the main test gate (it imports as `bakeoff`, with its own `conftest.py`).

## Layout

```
src/foodscholar/
  facade.py        FoodScholar (entry point)      graph_view.py  fs.graph + handles
  config.py        Pydantic config + YAML         io/            data contracts
  corpus/          chunk + NEL loading + chunker  annotate/      GLiNER(2) + dense HNSW + embed
  relations/       Layer 0 extract/ground/dedupe   io/relation.py Relation contract
  ontology/        FoodOn loader + FoodOnAPI       llm/           provider-agnostic client
  layer_a/         backbone projection + alias     layer_b/       themes (2 passes + merge)
  layer_c/         cited cards                     retrieval/     query API
  storage/         protocols + memory/elastic/neo4j adapters      viz/  renderable views
  cli/             typer commands                  evaluation/    gates + scorers
tests/{unit,integration}/   research/  notebooks/   docs/  config.example.yaml
```

## Conventions

- **Tests are the gate.** Make `pytest tests/unit` (foodscholar env) and `ruff check src
  tests` pass before claiming done. Add a unit test for new behavior (TDD where practical).
- **In-memory first.** New behavior should work against `FoodScholar.in_memory()` so the
  unit suite can exercise it without services.
- **Config is `extra="forbid"`** — add a field to the right Pydantic model, don't pass
  loose kwargs.
- **Don't edit `research/`** as if it were library code; it's archived.
- **Conventional Commits**, factual messages; branch off `main`, PR to merge.

## Where to look

- **Docs:** `docs/` (Sphinx, published on Read the Docs). Start with
  [docs/index.md](docs/index.md); [concepts/architecture.md](docs/concepts/architecture.md)
  for the design; [concepts/worked-example.md](docs/concepts/worked-example.md) for one
  chunk traced through every layer; [concepts/glossary.md](docs/concepts/glossary.md) for
  the vocabulary; [reference/](docs/reference/index.md) for the API.
- **Config:** `config.example.yaml` documents every field with its default.
- **Notebook:** `notebooks/graph_build.ipynb` is the phase-by-phase build (offline or real).
- **Design brief:** [BRIEF.md](BRIEF.md).
