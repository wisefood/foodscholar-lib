# FoodScholar — Python Package Implementation Brief

A self-contained briefing for implementing the `foodscholar` Python package. Hand this to an implementer (human or AI) and they should be able to scaffold the project end-to-end without further clarification.

---

## 1. Project context

FoodScholar is a hierarchical knowledge graph over a corpus of nutrition literature. The corpus comprises a few dietary guides, a few textbooks, and roughly 100,000 scientific abstracts. It has already been collected, extracted, and **chunked** into thousands of section-aware chunks. The chunks are the starting point for this package.

The package builds and serves a **three-layer hierarchical graph** over that corpus, and exposes a retrieval API on top.

- **Layer A — Backbone.** A curated, multi-facet semantic menu. Facets: `foods`, `health`, `sustainability`, `dietary_patterns`, `allergies`, `nutrients`. Built by projecting **FoodOn** down to a corpus-adaptive subset (frequency-weighted ancestor propagation, pruning, depth cap, blacklist, plus curated top-level facets the ontology can't supply alone).
- **Layer B — Themes.** Topic communities discovered **per shelf** via embedding-based community detection (Leiden primary, HDBSCAN fallback). Themes can attach to multiple shelves (multi-label).
- **Layer C — Write-ups.** LLM-generated cards attached to every shelf and every theme. Each card carries a summary, a tip, an evidence-quality note, optional controversy / confidence indicators, and citations to the chunks it draws from. **Every claim in a card must trace back to a cited chunk.**

Chunks attach to multiple Layer A shelves (multi-label, propagated through ontology ancestors) and to multiple Layer B themes.

---

## 2. Architecture decisions (already settled — do not re-litigate)

| Concern | Decision |
|---|---|
| Chunk store | **Elasticsearch 8.x** (BM25 + dense_vector kNN, hybrid via RRF, keyword-array filtering on `shelf_ids` / `theme_ids`) |
| Graph store | **Neo4j** (hierarchy, edges, card nodes, chunk stub nodes) |
| Bridge key | `chunk_id` exists in both stores; full chunk body lives only in Elastic |
| Ontology v1 | **FoodOn only**, loaded with `pronto` and `import_depth=0`. MONDO and ChEBI deferred to v2. |
| Food NER | **SciFoodNER** (already chosen by the project) |
| Entity linking | Lexical (exact + fuzzy against FoodOn names + synonyms) then dense fallback (**SapBERT**), with semantic-type gate |
| Embeddings | **SPECTER2** for scientific abstract chunks, **BGE-large** (`BAAI/bge-large-en-v1.5`) for textbook / guide chunks |
| Clustering | **Leiden / hierarchical Leiden** primary, **HDBSCAN** fallback. BERTopic acceptable for fast prototyping. |
| LLM (Layer C) | Pluggable. Default: Claude Sonnet class. Tests use a mock client. |
| Package mgmt | `pyproject.toml` with `hatch`. Optional extras for stage-specific deps. |
| Data contracts | **Pydantic v2** models throughout. |
| Backend abstraction | `typing.Protocol` classes for `ChunkStore`, `GraphStore`, `Embedder`, `LLMClient`, etc. Concrete adapters slot in. |
| Reproducibility | Every artifact stamped with a **config hash**. Phases produce versioned outputs that can be re-loaded to resume a downstream phase. |

---

## 3. Package layout

```
foodscholar/
├── pyproject.toml
├── README.md
├── config.example.yaml
├── examples/
│   ├── 01_load_chunks.py
│   ├── 02_run_annotation.py
│   ├── 03_build_layer_a.py
│   └── 04_query.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   └── integration/         # docker-compose for ES + Neo4j
└── src/
    └── foodscholar/
        ├── __init__.py
        ├── config.py        # Pydantic config models + YAML loader
        ├── versioning.py    # config-hash, artifact metadata
        ├── io/              # data contracts
        │   ├── __init__.py
        │   ├── chunk.py     # Chunk, Mention, EntityLink
        │   ├── graph.py     # Shelf, Theme, Card
        │   └── artifacts.py
        ├── corpus/
        │   ├── __init__.py
        │   ├── loader.py    # read chunks.parquet → list[Chunk]
        │   └── sources.py   # adapters per source_type
        ├── annotate/
        │   ├── __init__.py
        │   ├── ner.py       # SciFoodNER wrapper → list[Mention]
        │   ├── linker.py    # mention → ontology_id (lexical + dense)
        │   └── embedder.py  # SPECTER2 / BGE wrappers
        ├── ontology/
        │   ├── __init__.py
        │   ├── foodon.py    # pronto loader + cache
        │   └── api.py       # name_to_id, id_to_ancestors, id_to_synonyms, id_to_label
        ├── layer_a/
        │   ├── __init__.py
        │   ├── propagate.py # ancestor propagation + frequency aggregation
        │   ├── prune.py     # min-support, single-child collapse, depth cap, blacklist
        │   ├── facet.py     # curated facet definitions
        │   └── builder.py   # orchestrator: chunks + ontology → list[Shelf]
        ├── layer_b/
        │   ├── __init__.py
        │   ├── semantic_graph.py
        │   ├── community.py # Leiden / HDBSCAN
        │   └── builder.py
        ├── layer_c/
        │   ├── __init__.py
        │   ├── prompts.py   # versioned prompt templates
        │   ├── generator.py
        │   └── grounding.py # claim-to-chunk grounding check
        ├── retrieval/
        │   ├── __init__.py
        │   ├── resolve.py
        │   ├── traverse.py
        │   ├── hybrid.py
        │   └── synthesize.py
        ├── storage/
        │   ├── __init__.py
        │   ├── protocols.py # Protocol classes
        │   ├── elastic.py   # ElasticChunkStore
        │   ├── neo4j.py     # Neo4jGraphStore
        │   └── memory.py    # InMemoryChunkStore / InMemoryGraphStore (tests)
        ├── cli/
        │   ├── __init__.py
        │   └── main.py      # typer
        └── evaluation/
            ├── __init__.py
            ├── gold.py
            └── scorers.py
```

---

## 3.5 Python API surface

The internal module layout in §3 is the implementer's map. **Users should not have to learn it.** The public Python surface is intentionally small: one facade class (`FoodScholar`) plus a graph view object (`fs.graph`). Every CLI command in §9 is a thin wrapper around the same facade method so the two surfaces stay in lockstep.

### The facade

```python
from foodscholar import FoodScholar

# Production
fs = FoodScholar.from_config("config.yaml")
fs.load_chunks("data/chunks.parquet")
fs.build()
answer = fs.query("Is olive oil heart-healthy?")

# Notebooks / tests — zero config
fs = FoodScholar.in_memory()
```

`FoodScholar` owns four pluggable backends: `chunk_store`, `graph_store`, `embedder`, `llm`. Stores are constructed from `cfg.storage`; embedder/LLM default to mocks for the in-memory case and are pluggable via keyword args to either factory. Methods:

| Method | Maps to |
|---|---|
| `fs.info()` | dict of version + backends + active models |
| `fs.load_chunks(path)` | corpus loader → `chunk_store.upsert` |
| `fs.upsert_chunks(chunks)` | direct upsert (notebooks/tests) |
| `fs.init()` | provisions backing stores (ES index, Neo4j constraints) |
| `fs.annotate()` | NER + 3-tier linker + embeddings (returns ArtifactMeta) |
| `fs.ner` / `fs.linker` | lazy-built NER + Linker objects, individually probeable |
| `fs.build_layer_a()` | backbone phase |
| `fs.attach()` | chunk→shelf attachments + denormalize |
| `fs.build_layer_b()` | theme discovery phase |
| `fs.build_layer_c()` | write-up cards phase |
| `fs.build()` | annotate → layer_a → attach → layer_b → layer_c |
| `fs.query(text)` | retrieval pipeline (§14) |

Phase methods that aren't implemented yet raise `NotImplementedError` with a clear message ("phase 'X' is not implemented yet in foodscholar v0.1.0; see BRIEF.md §12"). This keeps the surface complete and discoverable from day one.

### Graph view (`fs.graph`)

The graph is the thing users want to explore most. `fs.graph` is a `GraphView` that exposes both read and write operations against the chunk + graph stores. **No method on `fs.graph` requires the user to touch the underlying protocols.**

```python
# Read — fluent navigation via handles
fs.graph.shelves(facet="dietary_patterns")        # list[ShelfHandle]
fs.graph.roots()                                  # top-level shelves per facet
fs.graph.shelf("s-med").label                     # passthrough to Pydantic Shelf
fs.graph.shelf("s-med").themes()                  # list[ThemeHandle]
fs.graph.shelf("s-med").chunks()                  # list[Chunk]
fs.graph.shelf("s-med").parent()                  # ShelfHandle | None
fs.graph.shelf("s-med").children()
fs.graph.shelf("s-med").neighbors(hops=2)
fs.graph.shelf("s-med").card()                    # CardHandle | None
fs.graph.theme("t-olive").shelves()               # back-references
fs.graph.theme("t-olive").chunks()
fs.graph.theme("t-olive").card()
fs.graph.card("s-med", "shelf").cited_chunks()
fs.graph.search("olive oil", shelf="s-med", k=5)  # hybrid retrieval, scoped

# Write — explicit mutation, no manual denormalization
fs.graph.add_shelf(shelf_id="s-med", label="Mediterranean diet",
                   facet="dietary_patterns", depth=1)
fs.graph.add_theme(theme_id="t-olive", label="Olive oil",
                   shelf_ids=["s-med"],
                   discovered_by="leiden", discovery_version="v1")
fs.graph.add_card(card)  # accepts a Pydantic Card or kwargs
fs.graph.attach_chunks(["c1","c2"], shelf="s-med")   # auto-denormalizes shelf_ids
fs.graph.attach_chunks(["c1"], theme="t-olive")
fs.graph.summary()                                # {"shelves": N, "themes": M, "roots": K}
```

`ShelfHandle`, `ThemeHandle`, `CardHandle` **wrap** their Pydantic model rather than subclass it — the underlying `Shelf` / `Theme` / `Card` stay serializable and frozen-friendly, while navigation methods live on the handle. The Pydantic model is always reachable via `handle.model`. Handles are lazy: each navigation method routes through the stores, so `fs.graph` stays in lockstep with mutations made by phase modules.

### Ontology view (`fs.ontology`)

The ontology backs the linker, the layer_a backbone projection, and layer_c card prompts. `fs.ontology` is a lazily-loaded `FoodOnAPI` (read-only) over the FoodOn terms declared in `cfg.ontology`:

```python
fs.ontology.name_to_id("olive oil")          # "FOODON:03309927" | None
fs.ontology.id_to_label("FOODON:03309927")
fs.ontology.id_to_synonyms("FOODON:03309927", include_related=False)
fs.ontology.id_to_ancestors("FOODON:03309927")    # closed transitive set
fs.ontology.id_to_parents("FOODON:03309927")      # direct only
fs.ontology.id_to_descendants("FOODON:00001002")
fs.ontology.is_subclass_of(child, ancestor)
fs.ontology.search("olive", limit=25)             # substring prefilter for SapBERT fallback
```

First access triggers `load_ontology(path, cache_path=...)` which uses pronto with `import_depth=0` (FoodOn only — MONDO and ChEBI deferred to v2 per §2). Results cache to a Parquet file alongside the source, keyed on `(source_size, source_mtime)` so the cache invalidates when FoodOn is updated. Tests bypass the loader with `fs.attach_ontology(api)`.

### Annotate (`fs.ner` / `fs.linker` / `fs.annotate()`)

The annotate phase is owned by three pluggable pieces, each with a default that works against the in-memory facade:

```python
fs.ner.extract("Mediterranean diet rich in olive oil.")  # list[Mention]
fs.linker.link(mention)                                  # EntityLink | None
fs.linker.dry_run("evo")                                 # convenience: text -> EntityLink
fs.annotate()                                            # full phase, returns ArtifactMeta
```

Defaults:

- **NER** — `KeywordNER.from_ontology(fs.ontology)`. Word-boundary regex over every label + exact synonym in the ontology (obsolete terms excluded). Real `SciFoodNERAdapter` is gated by the `[annotate]` extra and runs behind `@pytest.mark.slow`.
- **Linker** — `ThreeTierLinker(fs.ontology, ...)`. Tries in order: `lexical_exact` (exact case-insensitive match) → `lexical_fuzzy` (rapidfuzz `token_set_ratio`, threshold from `cfg.annotate.linker.lexical_threshold`) → `dense` (cosine over precomputed term embeddings, opt-in via the dense embedder). Each link records `method` and `confidence` so the post-hoc audit in §17 is mechanical.
- **Embedder** — `HashEmbedder` for in-memory; `HFEmbedder("allenai/specter2_base")` or `SourceTypeRouter(scientific=SPECTER2, general=BGE-large)` for production. The router dispatches per `chunk.source_type` — `abstract` → scientific; `textbook`/`guide` → general — per BRIEF §2.

Override defaults with `fs.attach_ner(...)`, `fs.attach_linker(...)`, or by passing `embedder=...` to either factory.

### Evaluation gates

`foodscholar.evaluation.evaluate_linker(linker, gold)` drives a JSONL gold set against any `Linker` and returns a `LinkerEvalReport` with `coverage`, `accuracy`, per-tier breakdown, and a list of misses. The §17 gate ("entity-linking coverage ≥ 70%") is a unit test in `tests/unit/test_evaluation_linker.py` and fails CI if the linker regresses below threshold.

Design principles, in priority order:

1. **One way to do common things.** A user should not have to choose between `fs.graph.attach_chunks(...)` and `fs.graph_store.attach_chunks_to_shelf(...)` + `fs.chunk_store.update_attachments(...)`. The convenience method does both.
2. **No silent state mismatches.** `attach_chunks` updates both stores in one call so denormalization can't drift. Phase modules call the same `GraphView` methods.
3. **Models stay pure.** Navigation lives on handles, not on Pydantic models. Models remain JSON-serializable and free of hidden store references.
4. **Stores stay protocol-only.** `GraphView` is a layer *above* the protocols. Adapters (`ElasticChunkStore`, `Neo4jGraphStore`) only need to implement the protocol — the fluent API comes for free.

The CLI commands listed in §9 should be implemented as one-line wrappers over the corresponding facade methods. No business logic in the CLI module.

---

## 4. Data contracts (Pydantic v2)

Every phase consumes and produces these types. Place under `src/foodscholar/io/`.

### `io/chunk.py`

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

ChunkId = str
SectionType = Literal[
    "abstract", "results", "discussion", "methods",
    "introduction", "conclusion", "guideline", "textbook", "other"
]
SourceType = Literal["abstract", "textbook", "guide"]

class Mention(BaseModel):
    text: str
    start: int
    end: int
    score: float
    ner_model_version: str

class EntityLink(BaseModel):
    mention: Mention
    ontology_id: str          # e.g. "FOODON:03309927"
    confidence: float
    method: Literal["lexical_exact", "lexical_fuzzy", "dense"]
    linker_version: str

class Chunk(BaseModel):
    chunk_id: ChunkId
    text: str
    source_doc_id: str
    source_type: SourceType
    section_type: SectionType
    year: int | None = None

    embedding: list[float] | None = None
    embedding_model: str | None = None

    mentions: list[Mention] = Field(default_factory=list)
    entity_links: list[EntityLink] = Field(default_factory=list)
    foodon_ids: list[str] = Field(default_factory=list)

    # denormalized from Neo4j for fast ES filtering
    shelf_ids: list[str] = Field(default_factory=list)
    theme_ids: list[str] = Field(default_factory=list)

    enrichment_version: str = "v0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### `io/graph.py`

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

ShelfId = str
ThemeId = str
CardId = str
Facet = Literal[
    "foods", "health", "sustainability",
    "dietary_patterns", "allergies", "nutrients"
]
EvidenceQuality = Literal["high", "medium", "low", "debated", "unclear"]

class Shelf(BaseModel):
    shelf_id: ShelfId
    label: str
    facet: Facet
    depth: int
    foodon_id: str | None = None
    parent_shelf_id: ShelfId | None = None
    chunk_count: int = 0

class Theme(BaseModel):
    theme_id: ThemeId
    label: str
    parent_theme_id: ThemeId | None = None
    shelf_ids: list[ShelfId]
    chunk_count: int = 0
    discovered_by: Literal["leiden", "hdbscan", "bertopic"]
    discovery_version: str

class Card(BaseModel):
    card_id: CardId
    target_id: str                     # shelf_id or theme_id
    target_type: Literal["shelf", "theme"]
    title: str
    summary: str
    tip: str | None = None
    evidence_quality: EvidenceQuality
    controversy_note: str | None = None
    confidence_note: str | None = None
    cited_chunk_ids: list[str]
    llm_model: str
    prompt_version: str
    safety_flagged: bool = False
    generated_at: datetime = Field(default_factory=datetime.utcnow)
```

### `io/artifacts.py`

```python
from datetime import datetime
from pydantic import BaseModel

class ArtifactMeta(BaseModel):
    artifact_id: str
    phase: str
    config_hash: str
    upstream_artifact_ids: list[str]
    record_count: int
    schema_version: str
    created_at: datetime
```

---

## 5. Storage protocols

### `storage/protocols.py`

```python
from typing import Protocol, Iterable
from foodscholar.io.chunk import Chunk, ChunkId
from foodscholar.io.graph import Shelf, Theme, Card, ShelfId, ThemeId

class ChunkStore(Protocol):
    def upsert(self, chunks: Iterable[Chunk]) -> None: ...
    def get(self, chunk_id: ChunkId) -> Chunk | None: ...
    def get_many(self, chunk_ids: list[ChunkId]) -> list[Chunk]: ...
    def search(
        self,
        query: str,
        theme_ids: list[ThemeId] | None = None,
        shelf_ids: list[ShelfId] | None = None,
        k: int = 10,
        use_vector: bool = True,
        use_bm25: bool = True,
    ) -> list[Chunk]: ...
    def update_attachments(
        self,
        chunk_id: ChunkId,
        shelf_ids: list[ShelfId],
        theme_ids: list[ThemeId],
    ) -> None: ...
    def scan(self) -> list[Chunk]: ...               # full enumeration (notebooks / GraphView)

class GraphStore(Protocol):
    def upsert_shelves(self, shelves: list[Shelf]) -> None: ...
    def upsert_themes(self, themes: list[Theme]) -> None: ...
    def upsert_cards(self, cards: list[Card]) -> None: ...
    def attach_chunks_to_shelf(self, shelf_id: ShelfId, chunk_ids: list[ChunkId]) -> None: ...
    def attach_chunks_to_theme(self, theme_id: ThemeId, chunk_ids: list[ChunkId]) -> None: ...
    def get_shelf(self, shelf_id: ShelfId) -> Shelf | None: ...
    def get_themes_for_shelf(self, shelf_id: ShelfId) -> list[Theme]: ...
    def get_chunks_for_theme(self, theme_id: ThemeId) -> list[ChunkId]: ...
    def get_neighbors(self, shelf_id: ShelfId, hops: int = 1) -> list[ShelfId]: ...
    def get_card(self, target_id: str, target_type: str) -> Card | None: ...
    def list_shelves(self) -> list[Shelf]: ...       # full enumeration (GraphView)
    def list_themes(self) -> list[Theme]: ...

class Embedder(Protocol):
    model_id: str
    @property
    def dim(self) -> int: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class LLMClient(Protocol):
    model_id: str
    def generate(self, prompt: str, max_tokens: int = 1024) -> str: ...
```

`InMemoryChunkStore` and `InMemoryGraphStore` must be implemented from day one — every unit test runs against them.

---

## 6. Neo4j model

```cypher
// Constraints — create on first run
CREATE CONSTRAINT shelf_id  IF NOT EXISTS FOR (s:Shelf)  REQUIRE s.shelf_id  IS UNIQUE;
CREATE CONSTRAINT theme_id  IF NOT EXISTS FOR (t:Theme)  REQUIRE t.theme_id  IS UNIQUE;
CREATE CONSTRAINT card_id   IF NOT EXISTS FOR (c:Card)   REQUIRE c.card_id   IS UNIQUE;
CREATE CONSTRAINT chunk_id  IF NOT EXISTS FOR (n:Chunk)  REQUIRE n.chunk_id  IS UNIQUE;

// Nodes
(:Shelf  {shelf_id, label, facet, depth, foodon_id, chunk_count})
(:Theme  {theme_id, label, parent_theme_id, chunk_count, discovered_by, discovery_version})
(:Card   {card_id, title, summary, tip, evidence_quality, llm_model, prompt_version, generated_at})
(:Chunk  {chunk_id})                       // stub only — body lives in Elastic

// Edges
(:Shelf)-[:PARENT_OF]->(:Shelf)
(:Shelf)-[:HAS_THEME]->(:Theme)
(:Theme)-[:HAS_SUBTHEME]->(:Theme)
(:Theme)-[:HAS_CHUNK {weight}]->(:Chunk)
(:Shelf)-[:HAS_CHUNK {weight}]->(:Chunk)
(:Card)-[:DESCRIBES]->(:Shelf|:Theme)
(:Card)-[:CITES]->(:Chunk)
```

A theme can be attached to multiple shelves via `HAS_THEME` from each.

---

## 7. Elasticsearch chunk index mapping

```json
{
  "settings": { "index": { "number_of_shards": 1, "number_of_replicas": 0 } },
  "mappings": {
    "properties": {
      "chunk_id":      { "type": "keyword" },
      "text":          { "type": "text",    "analyzer": "english" },
      "source_doc_id": { "type": "keyword" },
      "source_type":   { "type": "keyword" },
      "section_type":  { "type": "keyword" },
      "year":          { "type": "integer" },
      "embedding":     { "type": "dense_vector", "dims": 768, "index": true, "similarity": "cosine" },
      "embedding_model": { "type": "keyword" },
      "foodon_ids":    { "type": "keyword" },
      "shelf_ids":     { "type": "keyword" },
      "theme_ids":     { "type": "keyword" },
      "enrichment_version": { "type": "keyword" }
    }
  }
}
```

Dim defaults to 768 (SPECTER2). If using BGE-large, dim is 1024 — split into two indexes or pick one. Recommendation: one index per embedder, alias the active one as `foodscholar_chunks`.

Hybrid retrieval uses Elastic's RRF combining BM25 and kNN, with `shelf_ids` / `theme_ids` as a `filter` clause when those are provided.

---

## 8. Configuration

Single YAML, loaded into a Pydantic config model.

### `config.example.yaml`

```yaml
corpus:
  chunks_path: data/chunks.parquet

ontology:
  foodon_path: data/foodon.owl
  cache_path: data/foodon_cache.parquet
  include_imports: false           # import_depth=0 in pronto

annotate:
  ner_model: sci_food_ner_v1
  scientific_embedder: allenai/specter2_base
  general_embedder: BAAI/bge-large-en-v1.5
  linker:
    lexical_threshold: 0.85
    dense_threshold: 0.78
    semantic_type_gate: true

layer_a:
  min_support: 20                  # min chunks per shelf
  max_depth: 5
  collapse_single_child_chains: true
  blacklist_terms:
    - material entity
    - physical object
    - manufactured product
  facets: [foods, health, sustainability, dietary_patterns, allergies, nutrients]

layer_b:
  min_chunks_per_shelf: 50
  algorithm: leiden               # leiden | hdbscan | bertopic
  resolution: 1.0
  recurse_threshold: 200

layer_c:
  llm_model: claude-sonnet-4-6
  prompt_version: v1
  sample_size: 12
  grounding_check: strict
  safety_sensitive_facets: [allergies]

storage:
  chunk_store:
    backend: elastic
    url: http://localhost:9200
    index: foodscholar_chunks
  graph_store:
    backend: neo4j
    url: bolt://localhost:7687
    user: neo4j
    password: ${NEO4J_PASSWORD}
```

Pydantic config model in `foodscholar/config.py` with full validation. Environment variable substitution (`${VAR}`) handled at load time.

---

## 9. CLI design

Use `typer`. Every command takes `--config path/to/config.yaml`.

```
foodscholar init             # scaffolds config + creates ES index + Neo4j constraints
foodscholar annotate         # phase: NER + linking + embeddings
foodscholar build-layer-a    # phase: backbone
foodscholar attach           # phase: write chunk→shelf edges + denormalize
foodscholar build-layer-b    # phase: themes
foodscholar build-layer-c    # phase: write-up cards
foodscholar build-all        # runs build-layer-a → build-layer-c
foodscholar query "..."      # interactive retrieval
foodscholar info             # show config + artifacts + model versions
```

Each command must:
- log every action with structured fields
- write an `ArtifactMeta` record on success
- be re-runnable (idempotent where reasonable)

---

## 10. `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "foodscholar"
version = "0.1.0"
description = "Hierarchical knowledge graph over nutrition literature"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2",
    "typer>=0.12",
    "pyyaml",
    "pyarrow",
    "numpy",
    "pandas",
    "structlog",
]

[project.optional-dependencies]
ontology  = ["pronto"]
annotate  = ["torch", "transformers", "sentence-transformers"]
clustering = ["leidenalg", "python-igraph", "hdbscan", "scikit-learn", "umap-learn"]
bertopic  = ["bertopic"]
elastic   = ["elasticsearch>=8"]
neo4j     = ["neo4j>=5"]
all       = ["foodscholar[ontology,annotate,clustering,elastic,neo4j]"]
dev       = ["pytest", "pytest-cov", "ruff", "mypy", "pytest-asyncio"]

[project.scripts]
foodscholar = "foodscholar.cli.main:app"
```

---

## 11. Testing strategy

- `tests/unit/` — pure logic tests using `InMemoryChunkStore` and `InMemoryGraphStore`. Must run in seconds, no external services.
- `tests/integration/` — real Elasticsearch + Neo4j via `docker-compose`. Marked with `@pytest.mark.integration`. Skipped by default; CI runs them explicitly.
- A canonical smoke test:
  1. Build a corpus of 5 synthetic chunks
  2. Run annotate (mock NER + mock embedder)
  3. Build a 3-shelf Layer A with a hand-curated mini-ontology
  4. Attach chunks
  5. Discover 1–2 themes
  6. Generate 1–2 mock cards
  7. Run a query end-to-end and assert it returns cited chunks
- Coverage target: 80% on phase logic.

---

## 12. First-week implementation plan

Strict order. Do not skip ahead.

1. **Scaffold the package.** `pyproject.toml`, package layout, empty modules, `README.md`, `config.example.yaml`. Set up `ruff` and `mypy`.
2. **Implement all Pydantic models** in `io/`. These are the lingua franca.
3. **Implement Protocols** in `storage/protocols.py`.
4. **Implement `InMemoryChunkStore` and `InMemoryGraphStore`** with full Protocol coverage.
5. **Write the canonical smoke test** (Section 11). Make it pass against the in-memory stores.
6. **Wire up the CLI skeleton** with `typer`. Implement `foodscholar init` (no-op for in-memory) and `foodscholar info`.
7. **Implement `versioning.py`**: `config_hash(config) -> str`, `ArtifactMeta` helpers.
8. **Implement `ontology/foodon.py` + `ontology/api.py`** against a real FoodOn .owl file. Cache to Parquet. Write tests using a tiny test ontology.

At this point the foundation is solid. Phase implementations (annotate, layer_a, layer_b, layer_c, retrieval) follow in order.

---

## 13. Implementation principles

- **Pydantic v2 only.** Use `model_config = ConfigDict(frozen=True)` where appropriate.
- **Protocols, not ABCs.** Duck typing keeps adapters thin.
- **No I/O in pure logic modules.** `propagate.py`, `prune.py`, `community.py`, `grounding.py` etc. take typed inputs and return typed outputs. I/O is in `storage/` and orchestrators only.
- **Structured logging via `structlog`.** Every phase logs `phase`, `config_hash`, `record_count`, timing.
- **Stamp everything with versions.** NER model version, embedder version, ontology release, prompt version, pipeline version. Stamp onto every output artifact.
- **Versioned prompt templates.** Prompts live in `layer_c/prompts.py` as constants with explicit version strings.
- **Idempotency.** Upserts are keyed on stable IDs. Rerunning a phase produces the same graph state.
- **No MONDO, no ChEBI yet.** Food-only end-to-end first.
- **The retrieval architecture is the most under-specified part of the system.** Implement the version described in Section 14, but treat it as a v1 — measure, iterate, and document deviations.

---

## 14. Retrieval flow (v1)

Public entry point: `foodscholar.retrieval.answer(query: str, **kwargs) -> Answer`.

Internal pipeline:

1. **Parse** — extract entities and intent (harm / benefit / safety / compare / explain).
2. **Resolve to shelves** — entity → `foodon_id` → `shelf_id` lookup against GraphStore. Promote 1-hop neighbors of each anchor shelf.
3. **Resolve to themes** — for each activated shelf, fetch attached themes. Score themes by number of activated shelves they touch (intersection bonus). Apply intent filter where appropriate.
4. **Fetch cards** — pre-generated cards for top themes (and top shelves) from GraphStore. No LLM cost.
5. **Hybrid chunk retrieval** — ChunkStore.search() with `theme_ids` filter, hybrid BM25 + kNN via RRF. Rerank by evidence quality / recency / source-type.
6. **Synthesize** — LLM produces a final answer from top cards + top chunks + original query.
7. **Grounding check** — every claim in the answer must trace to at least one retrieved chunk. Failures: drop the claim, hedge, or trigger a follow-up retrieval round.

Output:

```python
class Answer(BaseModel):
    text: str
    tips: list[str]
    cited_chunks: list[ChunkId]
    cited_cards: list[CardId]
    activated_shelves: list[ShelfId]
    activated_themes: list[ThemeId]
    grounding_passed: bool
    llm_model: str
    prompt_version: str
```

---

## 15. Out of scope for v1

- MONDO / ChEBI integration (defer to v2)
- Sustainability ontology — keep as a curated facet only for v1
- Agentic / multi-step retrieval — deterministic only for v1
- User personalization / dietary profile filtering
- Real-time corpus updates — assume offline rebuild

---

## 16. Notes on the corpus

- Chunks are already produced upstream. The package consumes `chunks.parquet` (or equivalent) with at minimum: `chunk_id`, `text`, `source_doc_id`, `source_type`, `section_type`, `year`.
- Section-aware chunking is assumed — section types like `abstract`, `results`, `discussion` matter for downstream weighting.
- Approximate scale: ~100k abstract documents × 5–10 chunks each → ~500k–1M chunks.
- Plan for batch processing: embarrassingly parallel up to embedding step.

---

## 17. Sanity gates before each phase advances

Each phase must pass a hand-check gate before its outputs feed the next phase.

- **After annotate:** entity-linking coverage ≥ 70%; top-100 FoodOn frequency list looks like nutrition; 50 random links hand-checked.
- **After Layer A:** facet tree reviewed by a domain-savvy reviewer; depths capped; no `material entity`-style technical artifacts.
- **After Layer B:** 20 random themes inspected; community labels readable; min-chunk thresholds respected.
- **After Layer C:** 30 random cards reviewed; grounding-check pass rate ≥ 95%; safety-flagged cards held for expert review.

Document gate results in `evaluation/` artifacts. Don't advance past a failing gate.

---

End of brief.