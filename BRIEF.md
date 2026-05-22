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
| Food NER | **GLiNER bio** (`urchade/gliner_large_bio-v0.1`). Deterministic, batched, runs locally. *(Deviation: the brief originally specified SciFoodNER. Agentic-LLM NER was tried as an interim and dropped — see §3.5.)* |
| Entity linking | **HNSW kNN over BioLORD embeddings** of every FoodOn term (`hnswlib` `ip` index, built on first use and cached to disk). Encoder pluggable via `cfg.annotate.linker.nel_encoder ∈ {biolord, sapbert, minilm, mpnet}`. Elastic `dense_vector` backend opt-in. |
| Embeddings | **SPECTER2** for scientific abstract chunks, **BGE-large** (`BAAI/bge-large-en-v1.5`) for textbook / guide chunks |
| Clustering | **Leiden / hierarchical Leiden** primary, **HDBSCAN** fallback. BERTopic acceptable for fast prototyping. |
| LLM (Layer C + linker `llm` tier) | Provider-agnostic — `anthropic`, `openai`, `groq`, `gemini`, `ollama`, configured via `cfg.llm` with an ordered fallback chain. `config.example.yaml` default: Groq `llama-3.3-70b-versatile`, fallback local Ollama. Tests use a mock client. |
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
        │   ├── entity.py    # Entity (first-class linked entity)
        │   ├── graph.py     # Shelf, Theme, Card
        │   └── artifacts.py
        ├── corpus/
        │   ├── __init__.py
        │   ├── loader.py     # read chunks.parquet/csv/jsonl → list[Chunk]
        │   ├── csv_reader.py # legacy (chunk_id, chunk_text, type, chunk_metadata) CSV
        │   ├── nel_loader.py # prototype (chunk_id, chunk_entities_ner, chunk_uri_nel) CSV
        │   └── sources.py    # adapters per source_type
        ├── annotate/
        │   ├── __init__.py
        │   ├── gliner_ner.py # GLinerNER — GLiNER-bio → list[Mention]
        │   ├── nel_index.py  # NELIndex protocol + HNSWNELIndex / ElasticNELIndex
        │   ├── linker.py     # HNSWLinker: mention → ontology_id (dense kNN)
        │   └── embedder.py   # SPECTER2 / BGE / SapBERT wrappers
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
        │   ├── protocols.py        # ChunkStore / EntityStore / GraphStore Protocols
        │   ├── elastic.py          # ElasticChunkStore (chunks)
        │   ├── elastic_entities.py # ElasticEntityStore (first-class entities)
        │   ├── neo4j.py            # Neo4jGraphStore (incl. (:Entity) + [:MENTIONS])
        │   └── memory.py           # InMemoryChunkStore / EntityStore / GraphStore
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

`FoodScholar` owns five pluggable backends: `chunk_store`, `entity_store`, `graph_store`, `embedder`, `llm`. Stores are constructed from `cfg.storage`; embedder is lazy (built on first access — production SPECTER2+BGE-large weigh ~1.7 GB and we don't load them just to print `info()`); LLM defaults to a mock and is pluggable via keyword args to either factory. Methods (in the order an end user runs them):

| Method | Maps to |
|---|---|
| `fs.info()` | dict of version + backends + active models (embedder = `lazy(...)` until first use) |
| `fs.init()` | provisions all three stores: ES chunk index + ES entity index + Neo4j constraints. No-op for in-memory backends |
| `fs.ingest(corpus_dir, nel_dir=None)` | **the one-call pipeline.** With `nel_dir`: load chunks → attach precomputed mentions+links → upsert. Without: delegates to `load_and_annotate` (GLiNER+HNSW) |
| `fs.load_and_annotate(path)` | load → GLiNER + HNSW annotate → upsert → optional parquet snapshot, idempotent |
| `fs.embed(only_missing=True)` | fill in chunk-text vectors (SPECTER2 / BGE via `SourceTypeRouter`) without touching annotations |
| `fs.build_entities()` | derive first-class `Entity` records from chunks + ontology; write to entity store + `(:Entity)` graph nodes |
| `fs.entities` | `.list(prefix=)`, `.get(id)`, `.search(q)`, `.chunks_for(id)` — read view over the entity store |
| `fs.load_chunks(path)` | corpus loader → `chunk_store.upsert` (raw — no annotations) |
| `fs.upsert_chunks(chunks)` | direct upsert (notebooks/tests) |
| `fs.annotate()` | NER + linker + embeddings over chunks already in the store (returns `ArtifactMeta`) |
| `fs.ner` / `fs.linker` | lazy-built `NER` + `Linker` objects, individually probeable |
| `fs.build_layer_a()` | backbone phase |
| `fs.attach()` | chunk→shelf attachments + denormalize |
| `fs.build_layer_b()` | theme discovery phase |
| `fs.build_layer_c()` | write-up cards phase |
| `fs.build()` | annotate → layer_a → attach → layer_b → layer_c |
| `fs.query(text)` | retrieval pipeline (§14) |

The **recommended end-to-end flow** for an end user with pre-computed NER/NEL on disk:

```python
fs = FoodScholar.from_config({...})   # dict / Path / FoodScholarConfig all accepted
fs.init()                              # ES indexes + Neo4j constraints
fs.ingest(corpus_dir, nel_dir=nel_dir) # chunks + annotations
fs.embed()                             # vectors for kNN search (optional)
fs.build_entities()                    # first-class entities + (:Entity) nodes
```

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

### Entities view (`fs.entities`)

Linked entities are first-class. Each distinct `ontology_id` discovered in the corpus (FOODON, CHEBI, GAZ, PATO, …) becomes an `Entity` record carrying the ontology-resolved label/synonyms/ancestors (for FoodOn) plus corpus-side aggregates: `mention_count`, `chunk_count`, sample `chunk_ids` (capped at 50), `facet_hint` (max-voted from `Mention.entity_type`), `last_seen`. Built by `fs.build_entities()` from chunks already in the chunk store; persisted both to `fs.entity_store` (Elastic — own index, `foodscholar_chunks_entities`) and to `fs.graph_store` as `(:Entity)` nodes connected by `(:Chunk)-[:MENTIONS {confidence, method}]->(:Entity)` edges.

```python
fs.build_entities()                                # one-call: derive + persist + edges
fs.entities.list(prefix="FOODON", k=20)            # top-k by chunk_count
fs.entities.get("FOODON:03309927")                 # Entity | None
fs.entities.search("olive", prefix="FOODON")       # BM25 over label + synonyms
fs.entities.chunks_for("FOODON:03309927", k=10)    # ES terms-filter on foodon_ids
```

`fs.entities.chunks_for(id)` takes a fast path against Elastic when the id is `FOODON:*` (terms filter on the denormalized `foodon_ids` array on chunks); other prefixes walk the entity's inline `chunk_ids` sample. `fs.build_entities()` is idempotent — re-running over the same corpus produces identical records; re-running after `fs.ingest`-ing new chunks updates `mention_count` / `chunk_count` / `chunk_ids` in place.

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

### Annotate (`fs.ingest` / `fs.ner` / `fs.linker` / `fs.annotate()` / `fs.load_and_annotate(path)`)

The single recommended entry point for end users is `fs.ingest(corpus_dir, nel_dir=...)`:

```python
# Pre-computed NER/NEL on disk (skips GLiNER + HNSW; fast):
fs.ingest("data/foodscholar/corpus", nel_dir="data/foodscholar/ner")

# No pre-computed annotations — runs GLiNER + HNSW end-to-end:
fs.ingest("data/foodscholar/corpus")
```

`fs.ingest` reads every CSV / parquet / JSONL chunk file in the directory,
attaches annotations (from the supplied `nel_dir` CSVs in the prototype's
`(chunk_id, chunk_entities_ner, chunk_uri_nel)` shape, or via GLiNER + HNSW
when the directory is omitted), embeds each chunk via the source-type router,
and upserts everything to the configured `chunk_store`. A parquet snapshot
lands at `cfg.corpus.annotated_snapshot_path` when set; an existing
non-empty snapshot short-circuits the whole call.

The phase pieces remain inspectable for tests and debugging:

```python
fs.ner.extract("Mediterranean diet rich in olive oil.")  # list[Mention]
fs.linker.link(mention)                                  # EntityLink | None
fs.linker.dry_run("evo")                                 # convenience: text -> EntityLink
fs.annotate()                                            # full phase over the store
fs.load_and_annotate("data/chunks.csv")                  # load → annotate → snapshot
```

Defaults:

- **NER** — `GLinerNER` (`cfg.annotate.ner: gliner`). Wraps `urchade/gliner_large_bio-v0.1` (a fine-tuned biomedical GLiNER), runs locally, batched: `ner.extract_batch(texts)` is a single `GLiNER.inference(batch_size=N)` call. Character offsets come from GLiNER directly. The configured labels (`cfg.annotate.gliner.labels`) are the `EntityType` literal — the bridge from GLiNER's output to `Mention.entity_type` is a verbatim string copy. Default vocabulary covers `food`, `nutrient`/`micronutrient`/`macronutrient`/`food component`/`dietary supplement`, `dietary pattern`, `medical condition`, `biomarker`, and four pragmatic context tags (`Country`, `Measurement`, `Population`, `Time expression`). Gated behind the `[annotate]` extra.
- **Linker** — `HNSWLinker` over a `NELIndex`. The default backend is `HNSWNELIndex`: every non-obsolete FoodOn term is encoded once with a sentence-transformer (BioLORD by default; SapBERT / MiniLM / MPNet selectable via `cfg.annotate.linker.nel_encoder`) and inserted into an `hnswlib` `ip` index built on first use and cached to disk (key derives from encoder model id + a content hash of the term set, so a re-encode happens iff the ontology or encoder changes). At link time the surface form is encoded with the same model, top-1 kNN is taken, and the hit is accepted iff cosine ≥ `nel_min_sim`. `link_many(mentions)` batches every surface form in a chunk batch through a single encode + single kNN call — the path the runner uses. A surface-form cache on the index instance amortizes repeats across chunks. Every accepted link records `method = "dense"` and the cosine score as `confidence`.

  An `ElasticNELIndex` backend (same `NELIndex` protocol) is opt-in via `cfg.annotate.linker.nel_backend: elastic`; it queries an ES `dense_vector` index. Stub today — implementation lands with the storage milestone.
- **Embedder** — `HashEmbedder` for in-memory; `HFEmbedder("allenai/specter2_base")`, `SapBERTEmbedder` (also available as a `nel_encoder` choice), or `SourceTypeRouter(scientific=SPECTER2, general=BGE-large)` for production. The router dispatches per `chunk.source_type` — `abstract` → scientific; `textbook`/`guide` → general — per BRIEF §2.

Override defaults with `fs.attach_ner(...)`, `fs.attach_linker(...)`, or by passing `embedder=...` to either factory.

**Single-pass ingest.** `fs.load_and_annotate(path)` mirrors the validated standalone prototype: load chunks (CSV / parquet / JSONL — the CSV reader lifts the field-size limit to 10MB for large abstracts), run batched GLiNER + batched HNSW, upsert annotated chunks back to the store, and optionally write a parquet snapshot to `cfg.corpus.annotated_snapshot_path`. If the snapshot exists and is non-empty, the call short-circuits — idempotent reruns over a fixed corpus.

**Deviations from §2.** The original brief listed SciFoodNER + lexical→dense linking. SciFoodNER was dropped (no proprietary models); an LLM-driven `AgenticNER` was tried as an interim and superseded — it was non-deterministic, expensive, and required local span reconciliation because LLM offsets are unreliable. A four-tier `ThreeTierLinker` (lexical-exact + fuzzy + dense + LLM-select) was the matching interim linker; the fuzzy tier accounted for the §17 audit's wrong links and the LLM tier was a per-residue cost. Both are now removed in favor of GLiNER + HNSW, which validated cleanly in the prototype and is deterministic, fast, and pure dense.

### LLM client (`fs.llm`)

The LLM used by the `llm` linker tier and by Layer C card generation is provider-agnostic. `cfg.llm` declares a `primary` provider plus an ordered `fallbacks` list:

```yaml
llm:
  primary:   { provider: groq, model: llama-3.3-70b-versatile }
  fallbacks:
    - { provider: ollama, model: llama3.1, host: http://localhost:11434 }
  timeout_s: 30
```

Supported providers: `anthropic`, `openai`, `groq`, `gemini`, `ollama` — one thin adapter each in `foodscholar.llm.providers`, all implementing the `LLMClient` protocol. `build_llm(cfg.llm)` constructs the chain: a single client if there are no fallbacks, otherwise a `FallbackLLMClient` that tries primary → fallbacks in order, falling through on any error (timeout, rate limit, auth, service down). API keys are read from the environment (`GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) — never from config files; Ollama needs no key. SDKs are lazy-imported and gated behind the `[llm]` extra. When `cfg.llm` is absent the facade uses a built-in mock client (used by `in_memory()` and tests).

The `LLMClient` protocol has two methods: `generate(prompt) -> str` and `generate_json(prompt, schema) -> dict`. `generate_json` uses each provider's native structured-output mode (OpenAI/Groq `response_format`, Gemini `response_schema`, Ollama `format`; Anthropic falls back to instructed-JSON + tolerant parse). It guarantees the result *parses* and matches the schema's shape — **not** that the values are semantically correct (an LLM-reported character offset can be a valid integer yet wrong; callers needing exact positions verify them against the source themselves, as `AgenticNER` does).

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

EntityType = Literal["food", "nutrient", "health", "dietary_pattern", "allergen", "other"]

class Mention(BaseModel):
    text: str
    start: int
    end: int
    score: float
    ner_model_version: str
    entity_type: EntityType = "other"   # classified by AgenticNER; "other" otherwise

class EntityLink(BaseModel):
    mention: Mention
    ontology_id: str          # e.g. "FOODON:03309927"
    confidence: float
    method: Literal["lexical_exact", "lexical_fuzzy", "dense", "llm"]
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

### `io/entity.py`

```python
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

class Entity(BaseModel):
    """First-class linked entity. One record per distinct ontology_id across
    the corpus. Built by fs.build_entities()."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ontology_id: str                  # canonical PREFIX:LOCALID (FOODON:03309927)
    prefix: str                       # FOODON | CHEBI | GAZ | PATO | UBERON | ...
    label: str                        # ontology label (FoodOn) or most-frequent surface form
    synonyms: tuple[str, ...] = ()
    ancestor_ids: tuple[str, ...] = () # closed transitive set (FoodOn only — empty for other OBO)
    facet_hint: Facet | None = None    # max-voted mapping from Mention.entity_type
    mention_count: int = 0
    chunk_count: int = 0
    chunk_ids: tuple[str, ...] = ()    # sample, capped at ENTITY_CHUNK_SAMPLE_CAP=50
    last_seen: datetime = Field(default_factory=datetime.utcnow)
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

    # Entity graph
    def upsert_entities(self, entities: list[Entity]) -> None: ...
    def attach_chunks_to_entity(
        self, ontology_id: str,
        chunk_links: list[tuple[ChunkId, float, str]],   # (chunk_id, confidence, method)
    ) -> None: ...

class EntityStore(Protocol):
    """First-class entities live in their own indexable store."""
    def upsert(self, entities: Iterable[Entity]) -> None: ...
    def get(self, ontology_id: str) -> Entity | None: ...
    def get_many(self, ontology_ids: list[str]) -> list[Entity]: ...
    def list_by_prefix(self, prefix: str, *, k: int = 100) -> list[Entity]: ...
    def search(self, query: str, *, prefix: str | None = None, k: int = 10) -> list[Entity]: ...
    def scan(self) -> list[Entity]: ...

class Embedder(Protocol):
    model_id: str
    @property
    def dim(self) -> int: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class LLMClient(Protocol):
    model_id: str
    def generate(self, prompt: str, max_tokens: int = 1024) -> str: ...
    def generate_json(self, prompt: str, schema: dict, max_tokens: int = 1024) -> dict: ...
```

`InMemoryChunkStore` and `InMemoryGraphStore` must be implemented from day one — every unit test runs against them.

---

## 6. Neo4j model

```cypher
// Constraints — create on first run (fs.init() runs these)
CREATE CONSTRAINT shelf_id  IF NOT EXISTS FOR (s:Shelf)  REQUIRE s.shelf_id    IS UNIQUE;
CREATE CONSTRAINT theme_id  IF NOT EXISTS FOR (t:Theme)  REQUIRE t.theme_id    IS UNIQUE;
CREATE CONSTRAINT card_id   IF NOT EXISTS FOR (c:Card)   REQUIRE c.card_id     IS UNIQUE;
CREATE CONSTRAINT chunk_id  IF NOT EXISTS FOR (n:Chunk)  REQUIRE n.chunk_id    IS UNIQUE;
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.ontology_id IS UNIQUE;

// Nodes
(:Shelf  {shelf_id, label, facet, depth, foodon_id, chunk_count})
(:Theme  {theme_id, label, parent_theme_id, chunk_count, discovered_by, discovery_version})
(:Card   {card_id, title, summary, tip, evidence_quality, llm_model, prompt_version, generated_at})
(:Chunk  {chunk_id})                       // stub only — body lives in Elastic
(:Entity {ontology_id, prefix, label, synonyms, ancestor_ids, facet_hint,
          mention_count, chunk_count, last_seen})

// Edges
(:Shelf)-[:PARENT_OF]->(:Shelf)
(:Shelf)-[:HAS_THEME]->(:Theme)
(:Theme)-[:HAS_SUBTHEME]->(:Theme)
(:Theme)-[:HAS_CHUNK {weight}]->(:Chunk)
(:Shelf)-[:HAS_CHUNK {weight}]->(:Chunk)
(:Card)-[:DESCRIBES]->(:Shelf|:Theme)
(:Card)-[:CITES]->(:Chunk)
(:Chunk)-[:MENTIONS {confidence, method}]->(:Entity)
```

A theme can be attached to multiple shelves via `HAS_THEME` from each. `(:Chunk)-[:MENTIONS]->(:Entity)` edges are written by `fs.build_entities()` and carry the per-mention `confidence` + `method` (`dense` for prototype-derived linking).

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

The chunk store also ships nested `mentions` + `entity_links` fields and a `foodon_ids: keyword[]` denormalization (set by `fs.ingest`) so callers can `terms`-filter on FoodOn ids without unnesting.

### Entity index (`<chunk_index>_entities`)

`fs.build_entities()` writes a sibling index for first-class linked entities. Index name is derived from the chunk index by appending `_entities` (so a `foodscholar_chunks` chunk index pairs with a `foodscholar_chunks_entities` entity index).

```json
{
  "settings": { "index": { "number_of_shards": 1, "number_of_replicas": 0 } },
  "mappings": {
    "dynamic": "false",
    "properties": {
      "ontology_id":   { "type": "keyword" },
      "prefix":        { "type": "keyword" },
      "label":         { "type": "text", "fields": { "keyword": { "type": "keyword" } } },
      "synonyms":      { "type": "text" },
      "ancestor_ids":  { "type": "keyword" },
      "facet_hint":    { "type": "keyword" },
      "mention_count": { "type": "integer" },
      "chunk_count":   { "type": "integer" },
      "last_seen":     { "type": "date" }
    }
  }
}
```

Entity search uses `multi_match` over `label^2` + `synonyms` with an optional `prefix` term filter. `fs.entities.list_by_prefix(...)` sorts by `chunk_count` descending.

---

## 8. Configuration

Single YAML, loaded into a Pydantic config model.

### `config.example.yaml`

```yaml
corpus:
  chunks_path: data/chunks.parquet
  annotated_snapshot_path: data/annotated.parquet

ontology:
  foodon_path: data/foodon.owl
  cache_path: data/foodon_cache.parquet
  include_imports: false           # import_depth=0 in pronto

annotate:
  ner: gliner
  gliner:
    model_id: urchade/gliner_large_bio-v0.1
    threshold: 0.4
    batch_size: 16
  scientific_embedder: allenai/specter2_base
  general_embedder: BAAI/bge-large-en-v1.5
  batch_size: 16
  linker:
    nel_backend: hnsw              # hnsw | elastic
    nel_encoder: biolord           # biolord | sapbert | minilm | mpnet
    nel_top_k: 1
    nel_min_sim: 0.70

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

> **Current status (2026-05-22):** scaffold + annotate + **Layer A are done** —
> projection, `fs.attach()`, `fs.audit()`, `fs.quality_report()`, and
> `fs.semantic_consolidate()` (LLM-as-judge dedup) have all landed. Layer A is
> validated as a Layer B foundation (audit passes, ~116 clusterable foods
> shelves; the ~18% synthetic-root orphans are an accepted long-tail). **Next
> milestone: Layer B (theme discovery).** Full history + the Layer B handoff
> note is in [PROGRESS.md](PROGRESS.md) (newest entry on top).

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