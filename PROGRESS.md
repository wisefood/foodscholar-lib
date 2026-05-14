# Progress log

Running log of what landed in each working iteration. Newest entries on top. Each entry covers what changed, why, and the verification that confirmed it works.

For *what's next*, see [BRIEF.md](BRIEF.md) §12. For *what exists today*, run `foodscholar info --config config.yaml` or open [notebooks/build_graph.ipynb](notebooks/build_graph.ipynb).

---

## 2026-05-14 — Iteration 4.1 (M2 hardening): Linker quality on real FoodOn

**Goal:** the M2 linker passed every test on the 11-term mini fixture but produced obvious garbage when pointed at the real 39k-term FoodOn `.owl`. Tighten the fuzzy tier and add ontology-pool filtering so the surface holds up against real data.

### What changed

- **Fuzzy scorer: `token_set_ratio` → `WRatio`** ([src/foodscholar/annotate/linker.py](src/foodscholar/annotate/linker.py))
  - `token_set_ratio` ignores the size of the *target* — `"oliv oil"` scored 1.00 against `"oil"` because "oil" is a subset. WRatio penalizes length mismatch internally, so `"oliv oil"` now correctly prefers `"olive oil"` over `"oil"`.
  - Same fix made `"whole grains"` resolve to `"whole grain"` (was: `"whole"`), `"salmons"` to `"salmon"` (was: `"22960 - salmons (efsa foodex2)"`), `"olives"` to `"olive"` (was: `"olives (canned)"`).

- **Short-query strict threshold.** Queries ≤4 chars require fuzzy ≥ 0.95 (vs the default 0.85). WRatio is generous against short queries — `"evo"` previously scored 0.90 against `"devonshire cream"` (a coincidence of overlapping characters). The strict floor rejects these cleanly.

- **Length-ratio gate.** Reject any fuzzy match where `len(target) / len(query) < 0.5`. Catches the residual "oliv oil → oil" failure mode in shape, not just in score.

- **Tie-break: label > synonym, then closest length.** Within a 0.5pt window of the top score, prefer label matches over synonym matches, then prefer the target whose length is closest to the query. Fixes `"arachis"` → `"Arachis hypogaea"` (vs `"peanut oil"` via synonym) on the mini fixture; on real FoodOn this case still misses because NCBITaxon is filtered out (next bullet), but the bias is the right default.

- **Punctuation/whitespace normalization in `name_to_id`** ([src/foodscholar/ontology/api.py](src/foodscholar/ontology/api.py))
  - `_normalize(name)` collapses any run of non-alphanumeric to a single space, then lowercases + strips.
  - `omega 3`, `omega-3`, `omega_3`, `wholegrain` and `whole grain` all collapse to the same lookup key.
  - Applied uniformly in `name_to_id`, `name_to_ids`, `search`, and `_index_name` so query normalization and target indexing stay symmetric.

- **`FoodOnAPI(prefix_filter=...)`** ([src/foodscholar/ontology/api.py](src/foodscholar/ontology/api.py))
  - Real FoodOn `.owl` files embed NCBITaxon, CHEBI, BFO, ENVO, AfPO, OBI, RO, IAO terms inline (~9k of the ~39k total). Without filtering, the linker matched `"EVOO"` → `NCBITaxon:Brevoortia` (a fish genus) and `"iron"` → `CHEBI:iron(2+)`.
  - Default `prefix_filter=("FOODON:",)` keeps only FOODON ids. `None` disables filtering (used by tests with synthetic `TEST:` ids).
  - Wired through `cfg.ontology.prefix_filter` (defaults to `["FOODON:"]`) so users can opt into ChEBI/CDNO via YAML.

- **Tests + gold set updated**
  - All test sites that build `FoodOnAPI` from the mini fixture now pass `prefix_filter=None`.
  - `linker_gold.jsonl`: added stripped-whitespace case, `"x"` (degenerate single-char), and demoted `"evo"` to a `miss` (correct under the new short-query gate).
  - One existing test asserted `evo → olive oil` on the mini fixture; now asserts the opposite (None) plus `"oliv oil" → olive oil` as the fuzzy probe.

- **`config.example.yaml`** documents `prefix_filter` with a comment block.

### What's still wrong (and why we're stopping here)

Real-FoodOn probe after the hardening:

| Query | Got | Notes |
|---|---|---|
| `omega 3 fatty acids` | "magnesium salts of fatty acids" | tie-break length-bias preferring shorter label over `"high omega-3 fatty acids"` |
| `peanut allergy` | "peanut candy food product" | "allergy" doesn't exist in FoodOn; the linker has no signal that this query is about a disease |
| `iron deficiency` | "beef flat iron steak (raw)" | same: deficiency isn't a food concept |
| `cardiovascular disease`, `CVD` | None ✓ | correctly rejected |
| `EVOO`, `evo` | None | rejected by short-query gate; lexical can't catch these without a synonym in FoodOn |

Root cause for the remaining wrong answers: **no amount of lexical/fuzzy matching can reject a query that isn't *semantically* a food** — there's no signal in token overlap to distinguish "iron deficiency" (a clinical condition) from "iron-rich foods". This is exactly the gap the **dense tier (SapBERT)** exists to fill: terms in different semantic neighborhoods will have low cosine similarity, regardless of orthographic overlap.

### Verification

- `ruff check src tests` — clean
- `pytest` — **121 passed** (was 120; +1 for the short-query rejection test)
- Mini-ontology linker eval: still 100% coverage on the gold set
- Real FoodOn (39,278 terms): no longer produces NCBITaxon / CHEBI false positives; lexical-fuzzy matches now look reasonable for in-vocabulary queries

### Status at end of iteration

- M2 still v0.1.0 — no API surface changed except the new `prefix_filter` argument.
- Remaining linker quality issues are explicitly **dense-tier territory**. Bringing SapBERT online is a separate decision (cost: ~400MB download, torch install) that we deferred from M2.
- Going to M3 (Layer A backbone) is unblocked — it consumes whatever `foodon_ids` end up on chunks, and the lexical-exact + fuzzy pipeline now produces reasonable food ids on real-world text.

---

## 2026-05-14 — Iteration 4 (M2): Annotate phase (NER + 3-tier linker + embedders)

**Goal:** land BRIEF §12 step 9 — the annotate phase. End state: `fs.annotate()` runs NER → 3-tier linker → embedders over every chunk and writes the enriched copies back, idempotently.

### What changed

- **NER + Linker protocols** ([src/foodscholar/storage/protocols.py](src/foodscholar/storage/protocols.py))
  - `NER.extract(text) -> list[Mention]`
  - `Linker.link(mention) -> EntityLink | None`
  - Both `@runtime_checkable` — pluggable per-config like the other backends.

- **Three-tier linker** ([src/foodscholar/annotate/linker.py](src/foodscholar/annotate/linker.py))
  - `lexical_exact` → `lexical_fuzzy` (rapidfuzz `token_set_ratio`) → `dense` (cosine over precomputed term embeddings).
  - Dense tier is opt-in: pass `dense_embedder=None` and the linker degrades to exact+fuzzy. Keeps unit tests fast and lets v0.1.0 run without SapBERT installed.
  - Each link records `method` and `confidence` — surfaces which tier resolved a mention so audits are mechanical.
  - Resolves the "evo → olive oil" question from earlier: the fuzzy tier already gets it (token_set_ratio with the EVOO synonym scores 0.86). Dense covers cases where the surface form shares no tokens with any FoodOn name.

- **NER adapters** ([src/foodscholar/annotate/ner.py](src/foodscholar/annotate/ner.py))
  - `KeywordNER` — deterministic, dependency-free, word-boundary regex. `KeywordNER.from_ontology(api)` builds it from every (non-obsolete) ontology label + exact synonym.
  - `SciFoodNERAdapter` — wraps a HuggingFace token-classification pipeline (SciFoodNER per BRIEF §2). Lazy-imports `transformers`; behind `[annotate]` extra and `@pytest.mark.slow`.

- **Embedder adapters** ([src/foodscholar/annotate/embedder.py](src/foodscholar/annotate/embedder.py))
  - `HashEmbedder` — deterministic toy embedder; default for `FoodScholar.in_memory()`.
  - `HFEmbedder` — sentence-transformers backed; default model `allenai/specter2_base`.
  - `SourceTypeRouter(scientific, general)` — dispatches per `chunk.source_type` (abstract → SPECTER2; textbook/guide → BGE-large) per BRIEF §2. Itself an `Embedder` so it composes.

- **Phase runner** ([src/foodscholar/annotate/runner.py](src/foodscholar/annotate/runner.py))
  - Pure function: takes injected NER + Linker + Embedder + ChunkStore; for each chunk runs NER → link mentions → embed → write back.
  - Idempotent: re-running replaces mentions/links/embedding (Pydantic models are frozen, so writes go through `model_copy(update=...)`).
  - Routes through `SourceTypeRouter.embed_chunk(text, source_type)` when the embedder is a router; otherwise calls `embedder.embed([text])` and stamps the result.
  - Returns `ArtifactMeta` so callers can persist provenance. Also exposes a `dry_run(text, *, ner, linker)` helper.

- **Facade integration** ([src/foodscholar/facade.py](src/foodscholar/facade.py))
  - `fs.ner` and `fs.linker` are lazy properties built from `cfg.annotate` on first access.
  - `fs.attach_ner(...)` / `fs.attach_linker(...)` to override defaults.
  - `fs.annotate()` is now real — calls `runner.run(...)` and returns `ArtifactMeta`. The deferred-message version is gone.
  - `fs.linker.dry_run(text)` answers "what would the linker do for this string?" without going through the full phase.

- **Evaluation gate** ([src/foodscholar/evaluation/linker.py](src/foodscholar/evaluation/linker.py))
  - `evaluate_linker(linker, gold) -> LinkerEvalReport` with `coverage`, `accuracy`, per-tier breakdown, and per-record misses.
  - 28-record gold set ([tests/fixtures/linker_gold.jsonl](tests/fixtures/linker_gold.jsonl)) covers exact, fuzzy, and negative cases.
  - BRIEF §17 gate ("entity-linking coverage ≥ 70%") enforced as a unit test. Currently 100% on the mini ontology.

- **Notebook** — §6 now reads "Annotate" (no `[STUB]`); body is a single `fs.annotate()` call. Added a "Probe the linker" subsection with the dry-run table covering `evo`, `olives`, `oliv oil`, `Arachis hypogaea`, `quinoa`. Quickstart's deferred-error probe widened to catch both `RuntimeError` and `NotImplementedError`.

- **Pyproject** — `rapidfuzz` added to `[annotate]` and `[dev]`. New `slow` pytest marker registered (deselected by default; opt-in via `pytest -m slow`).

- **Docs** — BRIEF §3.5 gained the annotate subsection (NER, linker, embedder, evaluation gate). README has a new "Annotating chunks" section above ontology. `annotate/` now flagged M2 ✓.

### Design decisions worth remembering

- **NER and Linker are separate protocols.** A user can swap one without the other (e.g. real SciFoodNER + dev-time KeywordNER linker, or vice versa). The runner only knows protocols.
- **Linker tiers fall through, not vote.** First hit wins. This is deliberate — exact > fuzzy > dense in trustworthiness, and the `method` field makes the choice auditable.
- **Dense tier is opt-in even when configured.** The default `_build_linker` doesn't pass `dense_embedder`. Wiring SapBERT explicitly is a Layer-3 decision the user makes per-environment. Keeps v0.1.0 functional without ML downloads.
- **`KeywordNER.from_ontology` is the in-memory default.** It's not as good as SciFoodNER, but it gets every term that appears verbatim, with zero dependencies, and exercises the full pipeline. The right floor.
- **Idempotent annotate.** Re-runs replace mentions/links rather than appending, so the chunk store never accumulates stale annotations from earlier model versions.
- **Slow tests opt-in, not opt-out.** Default `pytest` stays under 35s. Real-model coverage is one flag away (`pytest -m slow`).

### Verification

- `ruff check src tests` — clean
- `pytest` — **120 passed** (72 → 120; +48 new tests across linker tiers, NER, embedder, runner, evaluation, facade integration, slow stubs)
- Linker coverage on mini gold set: 100% (28/28 — far above the 70% gate)
- Notebook executes end-to-end on the conda env. `fs.annotate()` is one line; the linker probe shows all three tiers in action.

### Status at end of iteration

- v0.1.0 — UX foundation + ontology + annotate complete.
- `fs.annotate()` is real. `fs.build()` will now run annotate then trip on `build-layer-a` (next milestone).
- Next milestone (BRIEF §12 step 10): **Layer A backbone** — frequency-weighted ancestor propagation, prune (min-support, depth cap, single-child collapse, blacklist), facet merge. The annotate output (`Chunk.foodon_ids`) is the input to this phase.

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
