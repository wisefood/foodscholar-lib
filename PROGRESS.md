# Progress log

Running log of what landed in each working iteration. Newest entries on top. Each entry covers what changed, why, and the verification that confirmed it works.

For *what's next*, see [BRIEF.md](BRIEF.md) §12. For *what exists today*, run `foodscholar info --config config.yaml` or open [notebooks/build_graph.ipynb](notebooks/build_graph.ipynb).

---

## 2026-05-14 — Iteration 3 (M1): FoodOn ontology layer

**Goal:** land BRIEF §12 step 8 — the FoodOn loader + lookup API. This is the prerequisite for every downstream phase (annotate's linker, layer_a's backbone projection, layer_c's prompts).

### What changed

- **`OntologyTerm` Pydantic model** ([src/foodscholar/io/ontology.py](src/foodscholar/io/ontology.py))
  - Frozen Pydantic v2 carrier with `id`, `label`, `synonyms`, `related_synonyms`, `parent_ids`, `ancestor_ids` (closed transitive), `obsolete`.
  - Re-exported from `foodscholar.io` and `foodscholar`.

- **Pronto-based loader** ([src/foodscholar/ontology/foodon.py](src/foodscholar/ontology/foodon.py))
  - `load_ontology(path, *, cache_path=None, include_imports=False)` — pure function.
  - Materializes ancestors transitively at load time so the API doesn't pay re-traversal cost on every call.
  - Filters out the self-reference that `pronto.Term.superclasses()` includes.
  - Exact-vs-related synonym scopes preserved (linker uses exact only by default).
  - Parquet cache keyed on `(source_size, source_mtime)` via a sidecar `.meta.json` — auto-invalidates when FoodOn is updated on disk.
  - Friendly `ImportError` when `pronto` isn't installed (points at `pip install 'foodscholar[ontology]'`).

- **`FoodOnAPI` lookup surface** ([src/foodscholar/ontology/api.py](src/foodscholar/ontology/api.py))
  - O(1) lookups: `name_to_id`, `name_to_ids`, `id_to_label`, `id_to_synonyms` (with `include_related=False` default), `id_to_ancestors`, `id_to_parents`, `id_to_descendants`, `is_subclass_of`, `search`.
  - Obsolete terms are loaded but excluded from name lookups so the linker never resolves to a deprecated id.
  - `search` is a deterministic substring prefilter (shortest match first); the dense SapBERT fallback is a separate concern.
  - Implements `__contains__`, `__len__`, `__iter__`, `terms()`.

- **Facade integration** ([src/foodscholar/facade.py](src/foodscholar/facade.py))
  - `fs.ontology` lazily loads the FoodOn declared in `cfg.ontology` on first access.
  - `fs.load_ontology(refresh=False)` for eager / forced reload.
  - `fs.attach_ontology(api)` to skip the loader entirely (notebooks, unit tests).
  - `fs.info()["ontology"]` reports `"loaded"` / `"configured"` / `"none"`.
  - Clear `RuntimeError` if `cfg.ontology` is missing and the user accesses `fs.ontology`.

- **Synthetic test fixture** ([tests/fixtures/mini_foodon.obo](tests/fixtures/mini_foodon.obo))
  - 11-term mini-ontology covering hierarchy (food → plant food → fruit → olive → olive oil), exact + related synonyms, an obsolete term, multiple facets. Used by every ontology unit test so we never need the real ~100MB FoodOn release.

- **Storage protocols touched indirectly:** none. The ontology lives outside the `ChunkStore` / `GraphStore` split; it has its own loader + API.

- **Notebook updated** ([notebooks/build_graph.ipynb](notebooks/build_graph.ipynb))
  - New §5 "Load the FoodOn ontology" — uses the test fixture so the notebook stays self-contained.
  - The annotate stub (§6) now uses `fs.ontology.name_to_id(...)` and `id_to_label(...)` — the linker surface the real annotate phase will call.
  - The layer_a stub (§7) derives `foods_root_id` from `fs.ontology.id_to_ancestors(...)`, exercising the same lookup the real backbone projection will use.

- **Docs** — BRIEF §3.5 gained the `fs.ontology` subsection. README has a new "Loading the ontology" section and the layout block flags `ontology/` as M1 ✓.

- **Dev workflow** — `pronto` added to the `[dev]` extra so `pip install -e '.[dev]'` is enough to run the suite.

### Design decisions worth remembering

- **Ancestors materialized at load time.** `OntologyTerm.ancestor_ids` is the *closed transitive* set, not direct parents only. Phases that walk ancestors (layer_a propagation, the linker's semantic-type gate) get O(1) access rather than re-walking the DAG. `parent_ids` stays separate for tree walks.
- **Obsolete terms loaded but hidden from name lookups.** They stay in `terms()` and `__contains__` so historical references resolve (`api.get("FOODON:legacy")`), but `name_to_id` won't return them — the linker can't accidentally resolve to a deprecated FoodOn id.
- **Cache invalidation by file stat, not content hash.** size + mtime is fast and good enough for a file the user explicitly drops in `data/`. Content-hashing a 100MB OWL file every load would be wasteful.
- **No `OntologyView` wrapper.** For read-only lookup, an extra wrapper layer would just re-export the same methods. `fs.ontology` *is* the `FoodOnAPI`. Mutation isn't a real operation here — the ontology is upstream of foodscholar.
- **`pronto` deferred to `[ontology]` extra in production but included in `[dev]`.** Keeps the core install slim while making the dev workflow one command.

### Verification

- `ruff check src tests` — clean
- `pytest` — **72 passed** (44 → 72; +28 new tests across loader, cache round-trip, cache invalidation, every API method, facade lazy/eager/attach/refresh)
- Notebook executes every cell end-to-end on the conda env, with real ontology lookups in §6 (annotate) and §7 (layer_a)
- `fs.attach_ontology(api)` works for tests; `fs.from_config(cfg).ontology` lazy-loads against the fixture

### Status at end of iteration

- v0.1.0 — UX foundation + ontology layer complete. Surface area for annotate, layer_a, layer_c is now exercisable through `fs.ontology` even before those phases land.
- Next milestone (BRIEF §12 step 9): the **annotate** phase — wire SciFoodNER + a real lexical/dense linker over `fs.ontology`, plus SPECTER2/BGE embedders.

---

## 2026-05-14 — Iteration 2: Public API surface (facade + graph view)

**Goal:** make the library intuitive before any phase code lands, so every future milestone plugs into a stable user-facing surface.

### What changed

- **`FoodScholar` facade** ([src/foodscholar/facade.py](src/foodscholar/facade.py))
  - `FoodScholar.from_config("config.yaml")` and `FoodScholar.in_memory()` factories.
  - One method per phase: `annotate()`, `build_layer_a()`, `attach()`, `build_layer_b()`, `build_layer_c()`, `build()`, `query()`. Deferred ones raise `NotImplementedError` with a precise message ("phase 'X' is not implemented yet in foodscholar v0.1.0; see BRIEF.md §12").
  - Convenience: `info()`, `load_chunks()`, `upsert_chunks()`, `init()`.
  - Owns four pluggable backends: `chunk_store`, `graph_store`, `embedder`, `llm`. Embedder/LLM default to mocks for the in-memory case; pluggable via kwargs on either factory.

- **`fs.graph` — fluent graph access** ([src/foodscholar/graph_view.py](src/foodscholar/graph_view.py))
  - `GraphView` exposes reads + writes over the chunk + graph stores.
  - Reads return `ShelfHandle` / `ThemeHandle` / `CardHandle`. Handles **wrap** Pydantic models (rather than subclass) so models stay serializable; navigation methods (`.parent()`, `.children()`, `.themes()`, `.chunks()`, `.card()`, `.cited_chunks()`, ...) live on the handle. `handle.model` returns the underlying Pydantic object.
  - Writes: `add_shelf`, `add_theme`, `add_card`, `attach_chunks`. `attach_chunks` updates *both* the graph edges and the denormalized `shelf_ids`/`theme_ids` on chunks in one idempotent call — the single most drift-prone part of the design becomes a one-liner.
  - Lookup misses return `None` rather than raise; lookups are lazy (no caching, always agrees with whatever the phase modules wrote).

- **CLI rewritten as a thin facade wrapper** ([src/foodscholar/cli/main.py](src/foodscholar/cli/main.py))
  - One line per command: every CLI command builds a `FoodScholar` and calls the matching method. No business logic in the CLI module.
  - `_build()` catches `NotImplementedError` so realistic configs (`elastic`/`neo4j` backends) give a friendly one-line message instead of a stack trace.
  - New `foodscholar version` command.

- **Storage protocols extended.** Added `ChunkStore.scan()` and `GraphStore.list_shelves()` / `list_themes()` so `GraphView` reads cleanly through the protocol rather than reaching into private store internals. In-memory stores implement them in two lines.

- **Top-level re-exports.** `FoodScholar`, `GraphView`, `ShelfHandle`, `ThemeHandle`, `CardHandle` exported from `foodscholar` so users never have to learn the internal module layout.

- **Notebook restructured** ([notebooks/build_graph.ipynb](notebooks/build_graph.ipynb))
  - New **§1 Quickstart** at the top — the 5-line happy path with `FoodScholar.in_memory()`.
  - Walk-through (§3–§12) now drives *everything* through `fs` and `fs.graph` — no raw store access anywhere. Each stub cell documents the exact one-line facade call that will replace it once its phase ships.

- **BRIEF.md** gained **§3.5 Python API surface** with rationale, the facade method table, and the `fs.graph` surface. §5 updated with the new protocol methods.

- **README rewritten** with a Quickstart, "Exploring the graph" section, and CLI overview.

### Design decisions worth remembering

- **Handles wrap, not subclass.** Pydantic v2 models stay frozen-friendly, serializable, and free of hidden store refs. Navigation lives on the handle layer.
- **One way to do common things.** `attach_chunks` is the only sanctioned way to add chunks-to-shelves. Users don't have to remember to mirror state across the two stores.
- **Stores stay protocol-only.** `GraphView` is a layer *above* the protocols. Future `ElasticChunkStore` / `Neo4jGraphStore` only have to implement the protocol — the fluent API comes for free.
- **Same code path for CLI and Python.** Every CLI command is `FoodScholar.from_config(...).<method>()`. Bugs and improvements land in one place.

### Verification

- `ruff check src tests` — clean
- `pytest` — **44 passed** (up from 21; new tests: facade ×9, graph_view ×14)
- `foodscholar version` / `info` / `init` / phase-deferred — all produce clean output
- Notebook executes every cell end-to-end on the conda env (Python 3.11.15)

### Status at end of iteration

- v0.1.0 — UX foundation complete. Public surface stable. Zero phase implementations.
- Surface area: `FoodScholar` facade (12 methods) + `fs.graph` (≈20 methods/handles). Everything below is internal.

---

## 2026-05-14 — Iteration 1: Scaffold (BRIEF §12 steps 1-7)

**Goal:** stand up the package end-to-end against the in-memory backend, with every module from BRIEF §3 present so phase code drops in without touching plumbing.

### What changed

- **`pyproject.toml` rewritten** to hatchling per BRIEF §10: full optional extras (`ontology`, `annotate`, `clustering`, `bertopic`, `elastic`, `neo4j`, `all`, `dev`), `foodscholar` console script, ruff + mypy + pytest config. `requires-python>=3.11`.

- **Pydantic v2 data contracts** ([src/foodscholar/io/](src/foodscholar/io/)) — `Chunk`, `Mention`, `EntityLink`, `Shelf`, `Theme`, `Card`, `ArtifactMeta`, with all Literal types from the brief.

- **Storage protocols** ([src/foodscholar/storage/protocols.py](src/foodscholar/storage/protocols.py)) — `ChunkStore`, `GraphStore`, `Embedder`, `LLMClient` as `@runtime_checkable` Protocols.

- **In-memory stores** ([src/foodscholar/storage/memory.py](src/foodscholar/storage/memory.py)) — full implementations of both store protocols. Toy hybrid search (token overlap for BM25 surrogate + cosine for kNN, combined via RRF) so unit tests can exercise the search path without Elasticsearch.

- **Versioning** ([src/foodscholar/versioning.py](src/foodscholar/versioning.py)) — stable `config_hash()` (order-independent JSON canonicalization → SHA-256[:16]) and `make_artifact_meta()` helper.

- **Pydantic config + YAML loader** ([src/foodscholar/config.py](src/foodscholar/config.py), [config.example.yaml](config.example.yaml)) with `${ENV}` substitution at load time.

- **Structured logging** ([src/foodscholar/logging.py](src/foodscholar/logging.py)) — `structlog` setup with console/JSON renderer, called once per CLI invocation.

- **Typer CLI** with `init` and `info` working against the in-memory backend; `annotate`, `build-layer-a/b/c`, `build-all`, `attach`, `query` wired but printing a deferred message.

- **Canonical smoke test** ([tests/unit/test_smoke_pipeline.py](tests/unit/test_smoke_pipeline.py)) walking corpus → annotate → Layer A → attach → Layer B → Layer C → query end-to-end against in-memory stores, per BRIEF §11.

- **Stubs** with clear docstrings for `annotate/`, `ontology/`, `layer_a/`, `layer_b/`, `layer_c/`, `evaluation/`, `storage/elastic.py`, `storage/neo4j.py`, and the four `examples/*.py` scripts.

- **Build notebook** ([notebooks/build_graph.ipynb](notebooks/build_graph.ipynb)) — 27-cell phase-by-phase walk-through. Stubs use the in-memory backend directly; each is labeled `[STUB]` with the future phase call.

### Environment

- Conda env `foodscholar` at `/mnt/miniconda3/envs/foodscholar` (Python 3.11.15). All commands target this interpreter; system 3.10 is incompatible with `requires-python>=3.11`.

### Verification

- `pip install -e '.[dev]'` — clean
- `pytest` — **21/21 passing** including the canonical smoke test
- `ruff check src tests` — clean (after fixing 8 small modernization warnings)
- `foodscholar info` — works against `config.example.yaml`

### Status at end of iteration

- v0.1.0 — every module from BRIEF §3 exists. Zero phase implementations. Surface usable only via internal modules (no `FoodScholar` facade yet).
