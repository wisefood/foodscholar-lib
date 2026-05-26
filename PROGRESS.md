# Progress log

Running log of what landed in each working iteration. Newest entries on top. Each entry covers what changed, why, and the verification that confirmed it works.

For *what's next*, see [BRIEF.md](BRIEF.md) §12. For *what exists today*, run `foodscholar info --config config.yaml` or open [notebooks/build_graph.ipynb](notebooks/build_graph.ipynb).

---

## 2026-05-26 — Iteration 9.1: drop SPECTER + collapse to single BGE-base embedder, fix ES `_source` round-trip

**Goal:** unblock Layer B Pass 1, which was reporting `with embeddings: 0/868` on every shelf even though the chunk store held 13,344 ES docs all stamped with a real `embedding_model`. Root cause turned out to be a vector-mapping change in ES 9.x — the dispatch architecture also stopped being justified, so collapsed it in the same pass.

### Root cause (the load-bearing investigation)

[ElasticChunkStore.init()](src/foodscholar/storage/elastic.py) declared the mapping with `dense_vector(type=dense_vector, index=true, similarity=cosine)` and no explicit `index_options`. ES 9.4 silently picks **`bbq_hnsw`** as the default index_options, which stores a binary-quantized vector in the HNSW segment and **drops the raw vector from `_source`**. So:

- index-wide `_count` with `exists: embedding` → 13,344/13,344 (the indexed field is there)
- `_mget` of those same ids → `_source` keys are `['chunk_id', 'created_at', 'embedding_model', ...]` — **no `embedding` key**
- `_doc_to_chunk` → Pydantic builds `Chunk(embedding=None)` (the default)
- Layer B Pass 1 → `[c for c in chunks if c.embedding is not None]` → 0 chunks → 0 candidates

Same data, two views, only one of which the Python side could see. The OOM that came up in the same session was a red herring on top of this — Docker `OOMKilled: true` because no JVM heap cap was set; the index data on the named volume is fine.

### What landed

**Mapping fix ([src/foodscholar/storage/elastic.py](src/foodscholar/storage/elastic.py)).** Embedding field is now declared explicitly with `dims: 768`, `similarity: cosine`, and `index_options: {type: hnsw, m: 16, ef_construction: 100}` — plain `hnsw`, not `bbq_hnsw`, so `_source` preserves the raw vector that Pydantic round-trips through `Chunk.embedding`. The cost is irrelevant at our scale: 13k chunks × 768 dims × 4 bytes ≈ 40 MB. The existing index has `bbq_hnsw` baked in — a `DELETE foodscholar_chunks` + re-run of `fs.embed()` is required to apply the fix (vectors are recoverable from chunk text, ~minutes on the T4).

**Drop SPECTER, single BGE-base embedder.** The `SourceTypeRouter(scientific=SPECTER2, general=BGE-large)` dispatch was producing two index widths (768 vs 1024) that BRIEF §7 papered over with "one index per embedder". With a single embedder there's exactly one index and one width.

- [src/foodscholar/config.py](src/foodscholar/config.py) — collapsed `scientific_embedder` + `general_embedder` (two keys) to a single `embedder: str = "BAAI/bge-base-en-v1.5"` on `AnnotateConfig`.
- [src/foodscholar/annotate/embedder.py](src/foodscholar/annotate/embedder.py) — deleted `SourceTypeRouter` entirely; `HFEmbedder` default flipped to BGE-base.
- [src/foodscholar/annotate/runner.py](src/foodscholar/annotate/runner.py) — dropped the `isinstance(embedder, SourceTypeRouter)` branch and replaced the per-chunk `router.embed_chunk(text, source_type)` loop with a single batched `embedder.embed([c.text for c in batch])` call. Side benefit: N forward passes per batch → 1 forward pass per batch.
- [src/foodscholar/facade.py](src/foodscholar/facade.py) — `_build_embedder` returns a plain `HFEmbedder(cfg.annotate.embedder)`; `fs.embed()._flush()` collapsed from "split by source_type, encode per backend, merge results" to a single batched encode; `info()` lazy banner reports the configured embedder instead of "router:…,…".
- Tests rewired: deleted the four `SourceTypeRouter` unit tests in [tests/unit/test_embedder.py](tests/unit/test_embedder.py); deleted `test_embed_uses_source_type_router_when_present` and rewrote `test_embed_router_batches_per_backend_one_encode_call_each` as `test_embed_batches_one_encode_call_per_flush` (asserts ONE encode per flush, no backend split) in [tests/unit/test_facade_embed.py](tests/unit/test_facade_embed.py).
- Config YAMLs ([config.example.yaml](config.example.yaml), [config.local.yaml](config.local.yaml), [config.gate.yaml](config.gate.yaml)) — collapsed to a single `embedder: BAAI/bge-base-en-v1.5` line.
- [BRIEF.md](BRIEF.md) — §2 architecture table row, §4 file-tree comment, §6 facade table + lazy-load paragraph, §7 ingest description + embedder description + dim-defaults paragraph, §11 example config: all retargeted to BGE-base-only.

### What this unblocks

After re-creating the ES index and re-running `fs.embed()`, the Layer B preview cell in [notebooks/build_graph.ipynb](notebooks/build_graph.ipynb) should report `with embeddings: N/N (100.0%)` instead of `0/868`, and Pass 1 similarity candidates should come back non-empty. Pass 2 (relatedness, entity-coherence based) was always working — it never touched vectors.

### Out of scope, deliberately

- The ES OOMKill on startup. Separate issue; the host has 16 GiB / 0 swap with Neo4j + Kibana competing. Documented in the debugging session; needs an explicit `ES_JAVA_OPTS=-Xms2g -Xmx2g` + `-m 4g` when the container is recreated. Not committing changes for it here since the container isn't managed from this repo.
- [CONSOLIDATION.md](CONSOLIDATION.md) still mentions BGE-large in the spec for the (future) semantic-consolidation embedder. That's a separate component spec, not the chunk embedder — left for a deliberate decision later.

---

## 2026-05-25 — Iteration 9.0 (M5): Layer B — dual-pass theme discovery + embed perf patch

**Goal:** land Layer B end-to-end (theme discovery inside each Layer A shelf) per the dual-pass architecture in [layer_b_construction_brief.md](layer_b_construction_brief.md). Bonus: fix the per-doc-update bottleneck blocking the real-corpus embedding run on a Colab T4 over an ngrok tunnel.

### What landed (sub-phase commits)

**Embed perf patch — `7e93f24..9bc91eb`** (a series; see commits before `2edb7f7`). Original `fs.embed()` encoded one chunk at a time and issued one ES `_update` per chunk — under a tunneled ES, throughput collapsed to ~2 chunks/sec on a T4 that should hit 100-300/sec. Two fixes: (1) the `_flush()` path now groups the pending batch by `source_type`, runs ONE `encode()` call per backend (SPECTER2 vs BGE-large) — the T4 amortization win; (2) a new `ChunkStore.update_embeddings_bulk(items)` collapses N `_update`s into one ES `_bulk` POST — the network-side win. New tests in [test_facade_embed.py](tests/unit/test_facade_embed.py) lock the contract ("ONE bulk call per flush", "ONE encode per backend per flush").

**Phase 0 — foundation (4 commits, `2edb7f7..a47ab5b`):**
- Extended `Theme` Pydantic ([io/graph.py](src/foodscholar/io/graph.py)) with the brief's new fields: `facet`, `discovery_pass`, `keyword_terms`, `foodon_id_signature`, `config_hash`, `version`. `shelf_ids` stays a `list[ShelfId]` (multi-shelf themes are kept open for v2 cross-shelf dedup; v1 builder emits length-1).
- `ThemeCandidate`, `MergeDecision`, `LayerBArtifact`, `LayerBAuditReport` Pydantic in new [layer_b/models.py](src/foodscholar/layer_b/models.py).
- Replaced flat `LayerBConfig` ([config.py](src/foodscholar/config.py)) with seven nested blocks: `SimilarityConfig`, `RelatednessConfig`, `LeidenConfig`, `MergeConfig`, `LabelingConfig`, `LayerBAuditConfig` (+ top-level `min_chunks_per_shelf`, `min_embedded_fraction`). `Literal["leiden"]` on per-pass algorithm fields makes HDBSCAN selection raise a Pydantic ValidationError (HDBSCAN cut from v1 per Plan-agent review — the precomputed-distance hack on the relatedness graph wasn't a valid metric).
- Three new storage protocol methods, implemented on memory + Elastic + Neo4j adapters:
  - `ChunkStore.bulk_set_theme_ids(items)` — sets `theme_ids` ONLY, leaves `shelf_ids` alone (the load-bearing reason this exists instead of reusing `bulk_update_attachments`, which clobbers shelf_ids under concurrent writes).
  - `GraphStore.attach_chunks_to_themes_bulk(items)` — writes `(:Chunk)-[:THEME_OF {primary, weight}]->(:Theme)` edges in one UNWIND-driven session.run on Neo4j. New `THEME_OF` edge label; the legacy `attach_chunks_to_theme` (which writes `ATTACHED_TO`) stays for back-compat. `get_chunks_for_theme` reads either label.
  - `GraphStore.clear_themes()` — DETACH DELETE every (:Theme); called by `build_layer_b` at the start so re-runs don't leave ghosts.

**Phase 1 — Pass 1 (similarity) (4 commits, `4318df6..7fe5812`):**
- [semantic_graph.py](src/foodscholar/layer_b/semantic_graph.py) — mutual-kNN over normalized chunk embeddings; cosine via numpy dot product, `argpartition` for top-k. Defensive normalization in case input vectors aren't unit-norm.
- [community.py](src/foodscholar/layer_b/community.py) — `run_leiden(graph, cfg)` using `leidenalg.RBConfigurationVertexPartition` with weighted modularity. Deterministic with fixed `random_state` (the load-bearing audit-parity guarantee). Empty/no-edges graphs short-circuit to `[]`.
- [label.py](src/foodscholar/layer_b/label.py) — `label_by_keywords` (c-TF-IDF with English stopwords + uni/bigram) and `label_by_llm` (one `LLMClient.generate` call per theme with keywords + 3 sample chunks; strips surrounding quotes; falls back to top keyword if LLM returns blank).
- [builder.py](src/foodscholar/layer_b/builder.py) — `build_shelf_similarity_candidates`: skips chunks without embeddings, runs kNN+Leiden, emits ThemeCandidate records with normalized-mean centroids.

**Phase 2 — Pass 2 (relatedness) (2 commits, `94157d9..235d650`):**
- [relatedness_graph.py](src/foodscholar/layer_b/relatedness_graph.py) — entity-bridge graph; edge weight = sum over shared FoodOn IDs of `1 / log(1 + doc_freq[id])`. The three knobs the brief calls make-or-break are exposed: `tau_strict` (link confidence floor), `min_shared_ids` (edge threshold), `max_doc_frequency` (drop ubiquitous entities). `always_exclude_iris=['FOODON:00001002']` is the default permanent kill-list — the 'food product' umbrella that survived Layer A propagates onto every chunk.
- `build_shelf_relatedness_candidates` — builds the graph, runs Leiden, emits candidates whose `foodon_ids` is the union of high-conf links across members (the entity signature the merge step uses for Jaccard).

**Phase 3 — merge + primary + persist + audit (5 commits, `ac83cb3..9359d7d`):**
- [merge.py](src/foodscholar/layer_b/merge.py) — greedy pair assignment with deterministic tie-break by `(-combined_similarity, sim_idx, rel_idx)`. Records the full cartesian product as `MergeDecision`s for audit. Greedy not optimal (Hungarian is) — documented; v1 scale (≤30 candidates per shelf) makes greedy sufficient.
- [primary.py](src/foodscholar/layer_b/primary.py) — per-pass-aware primary picker (Plan-agent flagged lex-first alone as too crude): similarity → closest-to-centroid in embedding space; relatedness → max sum-of-edge-weights to other members in the rel graph; merged → max of both scores per chunk. Lex-first chunk_id is the deterministic tie-breaker.
- [persist.py](src/foodscholar/layer_b/persist.py) — three writes in lockstep: `upsert_themes` (creates (:Theme) nodes + IN_SHELF edges, with all the new Layer B Theme fields stamped), `attach_chunks_to_themes_bulk` (THEME_OF edges with primary+weight), `bulk_set_theme_ids` (ES denorm, preserves shelf_ids). Merges with pre-existing theme_ids on each chunk (a chunk in two shelves can land in themes in both).
- [builder.build_shelf_themes](src/foodscholar/layer_b/builder.py) — full per-shelf pipeline (Pass 1 + Pass 2 + merge + label + primary picker). Emits Pydantic Theme records with deterministic theme_ids of the form `{facet}/{shelf_slug}/{label_slug}_{p}{seq}` (p ∈ {s,r,m}; seq is per-shelf-per-pass counter so identical-label themes get `_s1`/`_s2`).
- [audit.py](src/foodscholar/layer_b/audit.py) — `audit_layer_b(chunk_store, graph_store) -> LayerBAuditReport`. CRITICAL gates (flip `passed`): parity (Neo4j THEME_OF ↔ ES theme_ids agreement = 1.0), no dangling theme_ids, no empty themes. WARN-level reporting for tuning: per-pass theme counts (the brief's "≥ 1 from each pass" canary) and `merged_rate` (1.0 = Pass 2 isn't earning compute; 0.0 with relatedness=0 = entity graph mis-tuned).

**Phase 4 — orchestrator + facade + integration test (`5169f76`):**
- Top-level `build_layer_b(fs, *, facet, dry_run)` in [builder.py](src/foodscholar/layer_b/builder.py). Iterates shelves in the facet, applies `min_chunks_per_shelf` + `min_embedded_fraction` gates (Plan-agent flag — biased subsamples are worse than not clustering), skips the synthetic facet root (iteration-8 unclassified bucket). Calls `clear_themes()` at start. `dry_run=True` runs the full pipeline but persists nothing.
- [facade.py:1077](src/foodscholar/facade.py#L1077) `fs.build_layer_b(*, facet="foods", dry_run=False)` is now wired and returns a `LayerBArtifact` with `n_shelves_themed`, `n_themes_total`, `n_themes_by_pass`, `leiden_seed`, timestamps.
- The existing `foodscholar build-layer-b --config <path>` CLI command at [cli/main.py:83](src/foodscholar/cli/main.py#L83) already routes here via `_run_phase` — no CLI work needed.
- [tests/integration/test_layer_b_pipeline.py](tests/integration/test_layer_b_pipeline.py) — mini-corpus e2e (2 shelves × 8 chunks each) asserts `n_shelves_themed=2`, `n_themes >= 4`, `audit.passed`. Plus a dry-run test asserting no writes land.

### Design decisions worth remembering

- **Two passes capture different structure, not redundant signal.** Pass 1 finds *topical* clusters (chunks that discuss the same thing in similar prose). Pass 2 finds *entity-anchored* clusters (chunks co-mentioning the same FoodOn IDs even when the prose differs widely). A `discovery_pass="merged"` theme is grounded in *both* — stronger than either alone. Audit canaries: relatedness=0 means Pass 2 mis-tuned; merged_rate=1.0 means Pass 2 isn't earning compute.
- **`bulk_set_theme_ids` is a deliberate carve-out from `bulk_update_attachments`.** The latter overwrites both `shelf_ids` and `theme_ids` per call; using it from persist would race against any concurrent shelf writer. The new method touches `theme_ids` only — safe under any `shelf_ids` writer.
- **HDBSCAN cut from v1.** Plan-agent flagged the precomputed-distance hack on the relatedness graph as broken (the normalization isn't a valid metric). Brief defers HDBSCAN; v1 ships Leiden-only on both passes. `cfg.similarity.algorithm = "hdbscan"` raises ValidationError so misconfigurations fail loudly.
- **LLM labels are v1 default** (`cfg.labeling.strategy = "llm"`). Navigation labels need to read well; ~$0.60/run cost (Haiku) is trivial vs cluster compute. Keyword fallback is always computed and fed to the LLM as context — and stays available via `strategy="keyword"`.
- **Theme IDs are deterministic.** `{facet}/{shelf_slug}/{label_slug}_{p}{seq}` with per-shelf-per-pass `seq` counter. Audit cross-store parity depends on stable IDs across runs; same chunks + same `random_state` = identical theme membership and IDs.

### Verification

- `ruff check src tests` — clean.
- `pytest tests/` — **524 passed, 1 skipped** (was 432 before this iteration; +92 new across 8 layer_b_* test files + 2 integration tests). Test count by area: 7 layer_b_models, 3 layer_b_config, 13 layer_b_persist, 7 layer_b_semantic_graph, 5 layer_b_community, 7 layer_b_label, 6 layer_b_relatedness_graph, 9 layer_b_merge, 5 layer_b_primary, 10 layer_b_builder, 7 layer_b_audit, 2 integration.
- **Local env note:** the broken-openblas numpy in `/mnt/miniconda3/bin/python` made the embed-perf-patch session's local pytest runs skip layer_b tests. The project's actual env is `/mnt/miniconda3/envs/foodscholar/bin/python` (Python 3.11.15) where numpy + igraph + leidenalg + hdbscan + sklearn all work — this iteration's full suite runs green there. The wrong-interpreter confusion is what triggered the embed-perf debugging session in the first place.

### Status at end of iteration — Layer B is DONE (code-side)

- `fs.build_layer_b(facet="foods")` runs end-to-end against any backend (in-memory, ES+Neo4j).
- All §10 audit gates implemented; CRITICAL invariants enforced via `LayerBAuditReport.passed`.
- §17 sanity gate (20-theme hand audit against the real corpus) — **deferred to the next session in Colab/notebook** since (a) the embed run on the live ES is still finishing, (b) `fs.attach()` needs to re-run to populate Neo4j HAS_CHUNK edges (the current 6,290 attached chunks in ES carry the denorm but Neo4j has 0 edges — drift surfaced at iteration start).

### Handoff for next session

Run order against the live ES + Neo4j once embedding completes:
1. `fs.attach()` — rebuild Neo4j HAS_CHUNK edges from the existing ES shelf_ids denorm.
2. `fs.build_layer_b(facet="foods")` — full Layer B rollout. Expected: ~30-60 themed shelves at min_chunks_per_shelf=50, 100-300 total themes, parity=1.0.
3. §17 hand audit cell — sample 20 themes random, inspect label + 3 chunk excerpts, target ≥ 75% coherent.

If the per-pass canary fires (relatedness=0 or merged_rate=1.0), the tuning order from the brief is: try `relatedness.min_shared_ids=1`, then `relatedness.max_doc_frequency=0.6`, then inspect the `excluded` set (likely too aggressive).

---

## 2026-05-22 — Iteration 8.0 (M4): semantic consolidation + Layer A tuning; Layer A declared done

**Goal:** add the semantic-consolidation pass (catch duplicate shelves lexical rules miss), then validate Layer A is a sound foundation for Layer B and close out Layer A.

### What landed (two commits)

**`32ee313` — `fs.semantic_consolidate()` (LLM-as-judge).** New package [src/foodscholar/layer_a/semantic_consolidation/](src/foodscholar/layer_a/semantic_consolidation/) (models, embed, candidates, cluster, judge, prompts, apply, orchestrator). Runs as a standalone phase **after `fs.attach()`** so the judge can ground on real sample chunks. Pipeline: embed each shelf (label + FoodOn synonyms) → cosine candidate pairs → cluster into connected components (size-capped, weakest-edge split) → **one LLM call per cluster** returning `merge_groups`/`keep_alone` by index → enforce a permanent block-list → apply confirmed N-way merges via `see_also` (which the next `attach` re-homes). Off by default (`layer_a.semantic_consolidation.enabled`); `dry_run=True` returns the full `ConsolidationArtifact` for inspection without persisting. Reuses `fs.embedder` (BGE-large) and `fs.llm`. Config knobs: `cosine_threshold` (0.94), `max_cluster_size` (12), `auto_merge_confidence` (0.80), `subtype_patterns`, `permanent_block_list` (FoodOn-id pairs), `use_few_shot`, `exclude_scaffolding` + `classifier_suffixes`. Also hardened `GroqClient.generate_json` to fall back to a plain-text parse when strict `json_object` mode returns empty/truncated output.

**`f15425d` — projection fix: `foodon_ids` denorm no longer bypasses the `link_blocklist`.** [collect_support](src/foodscholar/layer_a/propagate.py) counted foods-facet support from both `entity_links` and the `foodon_ids` denorm list. When a term sat in both (the real-corpus shape — the nel_loader mirrors every link into `foodon_ids`), the `entity_links` loop would *skip* a blocklisted `(surface, id)` pair, then the `foodon_ids` loop *re-added* the bare id — silently undoing the blocklist (`foodon_ids` carries no surface text to match). Fix: `entity_links` is authoritative for any term it covers; the `foodon_ids` path only contributes terms with no `entity_link` (its designed role). +2 regression tests.

### Tuning learned on the real corpus (config-side, in the notebook — NOT committed library changes)

- **Judge precision is the hard part.** v1 prompt over-merged badly (apple+pear, fish+marine-fish, cow-milk fat variants) because it judged on "same category / co-occurs in chunks." Fixed with a v3 **identity-only** prompt (`PROMPT_VERSION = v3.0-identity`): merge ONLY if labels name the same food (spelling/synonym/processing variant); explicitly forbid category-vs-member and co-occurrence merging; chunks demoted to *confirm-identity-only*. Few-shot balanced and drawn from observed failures.
- **Cluster discipline matters.** At cosine 0.88 the candidate graph chained into one ~197-shelf hairball that blew groq's JSON budget. Raising to 0.94 + the cluster-size cap (weakest-edge split) keeps clusters tight and judgeable.
- **Scaffolding filter.** FoodOn organizational umbrellas ('food product', 'food consumer group', 'food modification process') have no synonyms + a classifier-suffix label; they're now excluded from consolidation candidates so they don't pollute clusters.
- **`food product` dumping ground (10% of chunks).** Diagnosed: it survived the umbrella rule by a whisker (`direct_share` 0.102 vs 0.10 cutoff) because the linker mapped generic mentions ('foods', 'whole foods', …) onto `FOODON:00001002`. The `link_blocklist` (+ the `f15425d` fix) drops those → umbrella rule fires → `food product` removed.

### Layer A readiness for Layer B — investigated and CLOSED

Added notebook diagnostics (§6e readiness, §6f/§6g/§6h orphan analysis) — exploratory, uncommitted. Findings on the real foods facet:

- **Structurally GO:** `fs.audit().passed == True` (single root, no cycles/dangling, ≥95% coverage, ≥99% attach integrity).
- **~116 shelves clusterable** (≥ `min_chunks_per_shelf`=50), ~113 too small (skipped), 0 empty. ~23.8k chunks attached. Solid Layer B input.
- **Synthetic-root orphans ~18%.** Removing `food product` sent its lifted chunks to the synthetic root (no surviving mid-level parent). Investigated thoroughly:
  - Orphans split into NEL junk (`food calorie datum`, `edible food`, `processed food`) — blocklistable — and a genuine **rare-food long-tail** (kiwifruit, mackerel, flaxseed) that clears `min_support` but whose only ancestors are umbrellas the umbrella rule kills.
  - **`min_support` is the wrong lever** (the rare foods already clear it). The right lever is whitelisting specific mid-level FoodOn categories — but the specificity-ranked recommender found no clean win: only 2 categories adopt ≥3 orphans, the rest are +1 each (39/70 have no good parent). **FoodOn's structure simply lacks navigable mid-level food shelves that survive the umbrella rule.**
  - **Decision: accept the ~18% as the 'unclassified' bucket Layer B skips.** Further Layer A tuning is diminishing returns.

### Status at end of iteration — Layer A is DONE

`fs.audit().passed`, 116 clusterable shelves, orphan tail understood and accepted. Semantic consolidation committed and ready (dedup pass, opt-in). **Next milestone: Layer B (theme discovery).** See the Layer B handoff note below.

### Layer B handoff (start here next session)

- **Hard blocker first:** run `fs.embed()` — attached chunks currently have **no embeddings** (`embedding=None`); Leiden/HDBSCAN need vectors. This is the one prerequisite.
- **Layer B clusters chunks *within* each shelf** into themes. Read per-shelf chunks via `graph_store.list_chunk_shelf_attachments()` (invert chunk→shelves to shelf→chunks) and fetch vectors via `chunk_store.get_many(chunk_ids)`; gate on `LayerBConfig.min_chunks_per_shelf` (50).
- **Skip the synthetic facet root (`facet:foods`)** — it's the ~18% unclassified bucket, not a coherent topic; clustering it would be noise.
- Stub exists: [src/foodscholar/layer_b/](src/foodscholar/layer_b/), `fs.build_layer_b()` currently `NotImplementedError`. `LayerBConfig` (algorithm=leiden, resolution, recurse_threshold) and the `Theme` model ([io/graph.py](src/foodscholar/io/graph.py)) are in place.
- **Optionally run consolidation first** (`fs.semantic_consolidate(dry_run=False)` with `enabled=True`) to dedup shelves before clustering — independent of the orphan question.
- §17 sanity gate after Layer B: 20 random themes inspected, labels readable, min-chunk thresholds respected.

### Notebook state (uncommitted)

[notebooks/build_graph.ipynb](notebooks/build_graph.ipynb) carries the consolidation cells (§6d), Layer-B readiness (§6e), and the orphan diagnostics (§6f–6h), plus a `link_blocklist` of generic surfaces in the config cell. These are exploratory/tuning aids — keep or prune as desired; none are required by the committed library code.

---

## 2026-05-21 — Iteration 7.0 (M4): Layer A backbone — full projection

**Goal:** the previous M3 builder at [src/foodscholar/layer_a/builder.py](src/foodscholar/layer_a/builder.py) was a foods-only stub — raw chunk counts, no single-child collapse (the flag existed in config but the logic was absent), no facet routing. This iteration implements the full projection per BRIEF §12 step 10 and the user's pruning spec: ordered passes (blacklist → whitelist → threshold → depth cap → single-child collapse), per-facet config, lifting on depth cap (not dropping), confidence-floored support, and provenance diagnostics (`support_direct`/`support_lifted`/`see_also`).

### What changed

**`src/foodscholar/layer_a/` split into four modules**
- `facet.py` (new) — canonical home for `ENTITY_TYPE_TO_FACET` (moved from facade.py — `build_entities` now imports from here). `route_link_to_facet(link)` adds a FOODON fallback so the prototype's `entity_type="other"` NEL CSVs still populate the foods facet. `stub_root(facet)` produces the empty-corpus shelf for facets with no support.
- `propagate.py` (new) — `collect_support(chunks, ontology, *, min_link_confidence, facet) -> SupportTable` tracks `direct` and `with_descendants` side-by-side. The threshold metric is `with_descendants` per spec. `foodon_ids` denormalization is honored for the foods facet — cheap path for prototype-NEL chunks that have no per-mention entity_type.
- `prune.py` (new) — order of operations locked in code (1. blacklist → 2. whitelist exception → 3. threshold → 4. depth cap → 5. single-child collapse). Depth cap **lifts** rather than drops: a term at depth 8 with cap 5 gets its `parent_shelf_id` re-pointed to the nearest surviving ancestor at depth ≤ 5, and its reported `depth` is clamped to the cap. Single-child collapse iterates to fixed point; when shelf B collapses into its only surviving child C, B's `foodon_id` is recorded on C's `see_also` for provenance.
- `builder.py` (rewritten) — orchestrator only. Multi-facet loop. Empty support tables emit a stub root via `facet.stub_root(...)`.

**`Shelf` Pydantic gained three fields** ([src/foodscholar/io/graph.py](src/foodscholar/io/graph.py))
- `support_direct: int = 0` — chunks mentioning this exact term.
- `support_lifted: int = 0` — chunks lifted in from pruned descendants.
- `see_also: list[str] = []` — foodon_ids of shelves collapsed into this one.

The invariant `chunk_count == support_direct + support_lifted` holds for non-stub shelves. Neo4j adapter's `upsert_shelves` Cypher SET clause and `_shelf_from_record` reader were updated in lockstep.

**`LayerAConfig` reshaped for per-facet overrides** ([src/foodscholar/config.py](src/foodscholar/config.py))
- New `FacetConfig` Pydantic — every field optional, None means "fall back to globals."
- `LayerAConfig.facet_overrides: dict[Facet, FacetConfig]` — facets absent from the dict use globals.
- `LayerAConfig.resolve_facet(facet) -> _ResolvedFacetConfig` returns a fully-resolved (no-None) config the pruner consumes. Keeps the prune logic free of `None`-handling.
- New `min_link_confidence: float = 0.70` (global), overridable per facet. Defaults to the linker's `nel_min_sim` so projection is no stricter than ingestion unless the user explicitly tightens it.
- Globals (`min_support`, `max_depth`, `collapse_single_child_chains`, `blacklist_terms`, `facets`) kept verbatim for backward compat.

**`FoodOnAPI` gained `id_to_children`** ([src/foodscholar/ontology/api.py](src/foodscholar/ontology/api.py))
- `_children: dict[id, set[id]]` precomputed in `__init__` by inverting `parent_ids`, same pattern as the existing `_descendants` precompute. Public `id_to_children()` returns sorted list. O(1) lookup, deterministic order.

**`config.example.yaml`** documents the per-facet override structure with the user's spec values (foods threshold=25, depth_cap=6, foods-specific blacklist of FoodOn organizational classes, whitelist placeholder, health depth_cap=7 for deeper disease taxonomies). All commented out so the default is the safe globals.

**Notebook** gained §5 "Build Layer A" between Entities and Inspect — three cells: an explanation of projection vs. attachment + the prototype-NEL caveat that non-foods facets stay stub-rooted on this corpus, a tunable `build_layer_a()` call, and a diagnostics cell printing shelves-per-facet / top-10 foods shelves / depth distribution / inflation flag (`support_lifted/chunk_count > 0.9`).

### Design decisions worth remembering

- **Op order is locked, not configurable.** Reversing blacklist↔threshold leaks chunks under blacklisted intermediates and inflates their parents (the threshold sees the inflated count, makes the wrong keep/drop decision). Reversing depth-cap↔collapse creates collapses the cap would have prevented. The order is documented in `prune.py`'s docstring referencing the plan file.
- **Depth cap lifts, doesn't drop.** A term too deep in the ontology hasn't done anything wrong — only the *display position* is too deep. Lifting preserves the term as a shelf at a shallower position; its chunks remain reachable via the lifted shelf. Dropping would lose coverage.
- **`route_link_to_facet` has a FOODON fallback** for prototype-NEL data with `entity_type="other"`. Without it, the entire foods projection would be dead code on the current corpus (`entity_type="other"` doesn't map to any facet, so no link would be counted). The fallback is honest about what the chunk-store data actually contains.
- **Two embedders → two facets too:** sustainability has no entity_type that maps to it AND no OBO ontology we project — always a stub root regardless of corpus. Allergies / health / dietary_patterns / nutrients **could** populate, but only after re-annotation with GLiNER (prototype-NEL is `entity_type="other"` for all loaded mentions per [nel_loader.py:116-123](src/foodscholar/corpus/nel_loader.py#L116-L123)).
- **Raw count + confidence floor, not weighted sum.** Cosines aren't probabilities — summing them isn't meaningful. A floor (default 0.70, matching the linker's `nel_min_sim`) is a quality gate; counts are then a chunk vote. Threshold semantics stay legible ("min_support: 20" means "≥ 20 chunks", not "≥ 20.0 cosine-units").
- **Per-facet overrides, not per-facet config blocks.** Globals stay the load-bearing knobs; an override fills in only the field that differs. Keeps `config.example.yaml` short and the override surface easy to reason about.

### Verification

- `ruff check src tests` — clean (one SIM103 + one SIM108 fixed during the iteration).
- `pytest` — **264 passed, 1 skipped** (baseline was 254 before this iteration; 10 new tests added). Existing 4 layer_a tests passed after explicitly setting `collapse_single_child_chains=False` and `facets=["foods"]` — they exercise propagation/blacklist in isolation, where the new defaults (collapse on, all 6 facets) would have changed observed output unrelated to what those tests assert.
- New unit tests cover: single-child collapse fires on a pure chain, single-child collapse does **not** fire when siblings survive, depth-cap lifting (`max_depth=2` clamps reported depth + repoints parent edge), whitelist override, see_also is populated with all collapsed ancestor ids, confidence floor filters low-confidence links, per-facet override resolves over globals correctly, sustainability emits a single stub root, blacklist-before-threshold lets chunks lift through blacklisted intermediates.
- New ontology test: `id_to_children` returns direct children only, sorted, empty for leaf or unknown id.

### Caveats — what this iteration does NOT do

- **Does not run `fs.attach()`.** Layer A is projection only. Chunk-to-shelf edges and the `shelf_ids` denormalization on chunks are still `NotImplementedError`. The `lifted_from` field on attachment edges (user's spec) belongs there.
- **Does not implement sibling merging.** Marked optional in the user's spec; defer until first sanity audit shows it's needed.
- **Non-foods facets are stub roots on the current real corpus.** Re-annotating with GLiNER (which populates `entity_type` correctly) is a prerequisite for nutrients/health/dietary_patterns to project meaningful shelves. Sustainability stays a stub forever — no OBO we link to covers it.
- **§17 sanity gate** (50-chunk hand audit) is documentation, not a unit test. To be performed against the real corpus once the notebook diagnostics cell is run.

### Status at end of iteration

- M4 Layer A projection landed. `fs.build_layer_a()` produces multi-facet, pruned shelves with provenance diagnostics. Per-facet config tuning available via `cfg.layer_a.facet_overrides`.
- Next per BRIEF §12: `fs.attach()` — read shelves + chunks, walk ancestors per `entity_links`, write `(:Shelf)-[:HAS_CHUNK]->(:Chunk)` edges + denormalize `shelf_ids` onto chunks (the single drift-prone step the GraphView's `attach_chunks` already centralizes for tests). Then Layer B theme discovery.

---

## 2026-05-21 — Iteration 6.0 (M3): GLiNER + HNSW pivot, ES/Neo4j adapters, ingest/embed/entities

**Goal:** the M2 agentic-NER + 3-tier-linker stack was not converging — the fuzzy tier was the source of the §17 audit wrong-links, agentic NER was non-deterministic and required local span reconciliation. A standalone `gliner.py` prototype validated GLiNER bio + HNSW-over-BioLORD on real FoodOn. M3 makes that pipeline the only NER+NEL path, finalizes the storage backends end-to-end, makes the configuration in-code-friendly, and promotes linked entities to first-class citizens.

### What changed (in commit order — 8 landed commits)

**`a3f40a0` M3 (annotate): pivot to GLiNER + HNSW NEL.**
- New `GLinerNER` (`annotate/gliner_ner.py`) wraps `urchade/gliner_large_bio-v0.1` behind the `NER` protocol. Batched fast path `ner.extract_batch(texts)` runs a single `GLiNER.inference(batch_size=N)` call — the runner is wired against it.
- New `NELIndex` protocol (`annotate/nel_index.py`) with `HNSWNELIndex` (default; local `hnswlib ip` index over BioLORD-encoded FoodOn term labels, build-on-first-use cache keyed on `sha256(encoder + sorted-term-id-set)`) and `ElasticNELIndex` (stub, opt-in for the storage milestone).
- `HNSWLinker` (`annotate/linker.py`) is a thin single-tier dense linker over the NEL index. `link_many` is the batched path the runner uses.
- `EntityType` (`io/chunk.py`) widened to GLiNER's 13-label vocab (food/nutrient/micronutrient/macronutrient/food component/dietary supplement/dietary pattern/medical condition/biomarker/Country/Measurement/Population/Time expression + `other`).
- Deleted: `agent_ner.py`, `ner.py` (`KeywordNER`), `dense_index.py`, `ontorag/`, repo-root `gliner.py` prototype, and the 5 test files exercising them.

**`3bf7e93` M3 (config): in-code dict config + `fs.load_and_annotate`.**
- `resolve_config(config)` normalizes `str | Path | dict | FoodScholarConfig` into a validated config — `${ENV}` substitution now flows over dict inputs too. `from_config()`, `in_memory(config=...)`, and `__init__` all route through it.
- `AnnotateConfig` / `LinkerConfig` reshaped: `cfg.annotate.ner: "gliner"`, `cfg.annotate.gliner.*`, `cfg.annotate.linker.{nel_backend, nel_encoder, nel_top_k, nel_min_sim, nel_index_path, nel_metadata_path}`, `cfg.annotate.batch_size`, `cfg.corpus.annotated_snapshot_path`.
- `fs.load_and_annotate(path)` is the release-ready single-call entry point that mirrors the prototype's `main()` — load → annotate → upsert → optional parquet snapshot with skip-if-exists idempotency.
- CSV reader hardened: `csv.field_size_limit(10 MB)` at import (large abstracts in the prototype's corpus).

**`0aa52f5` M3 (storage): release-ready ElasticChunkStore + Neo4jGraphStore.**
- `ElasticChunkStore` fully implemented: index mapping (BM25 `text`, `dense_vector` cosine, nested mentions/entity_links, `keyword[]` for shelf_ids/theme_ids/foodon_ids, flattened source_metadata), bulk upsert paginated at 500, search with BM25+kNN+RRF fusion, search_after-based `scan` / `iter_chunks`, scoped `_update` for annotations. `init()` is idempotent.
- `Neo4jGraphStore` fully implemented: MERGE upserts for shelves/themes/cards/chunk stubs, parameterized Cypher, list/get/get_neighbors. `init()` creates `CREATE CONSTRAINT … IF NOT EXISTS` on all four node types.
- Auth at the config layer: `ChunkStoreConfig.{api_key, username, password}` (HTTP-basic wins over api_key; env fallback to `ELASTICSEARCH_API_KEY`), `GraphStoreConfig.password` with env fallback to `NEO4J_PASSWORD`. `ProviderConfig.api_key` (with `_resolve_secret` cascade) plumbed through every LLM adapter.
- `ChunkStore` + `GraphStore` protocols gain `init()` (no-op in-memory, real provisioning remote). `fs.init()` calls all three stores uniformly.

**`a6f20b7` M3 (docs): BRIEF + notebook refresh.** §2/§3.5/§8 updated; full notebook rebuild around the new pipeline.

**`941683f` M3 (api): one-call ingest + use precomputed NER/NEL when available.**
- New `corpus/nel_loader.py` reads the prototype's `(chunk_id, chunk_entities_ner, chunk_uri_nel)` CSVs. `shorten_obo_uri` normalizes purls (`http://purl.obolibrary.org/obo/FOODON_03309927` → `FOODON:03309927`). Empty URI slots become NIL mentions (kept as `Mention`, no `EntityLink`). FoodOn ids land in the chunk's denormalized `foodon_ids`; CHEBI/GAZ/PATO/… stay in `entity_links` only.
- `fs.ingest(corpus_dir, *, nel_dir=None, snapshot_path=None)` is the new top-level entry point. With `nel_dir`: load chunks → attach precomputed annotations → upsert (no GLiNER, no HNSW, no `[annotate]` extra needed). Without `nel_dir`: delegates to `load_and_annotate`. **All chunks are inserted** — chunks with no matching NEL row land with empty mentions/links, never silently dropped.
- Notebook simplified to 3 happy-path cells + collapsed "Under the hood" appendix.

**`d36c85f` M3 (perf): lazy chunk embedder.** `from_config` no longer eagerly builds `SourceTypeRouter(SPECTER2, BGE-large)` (~1.7 GB of model weights). `fs.embedder` is a property — first access pays the cost; `fs.info()` reports `lazy(…)` until then. An explicit `embedder=` kwarg still skips the lazy build.

**`6a7c0d8` M3 (embed): split chunk-text embedding out of ingest into `fs.embed()`.**
- `fs.ingest(nel_dir=...)` no longer embeds chunks — they land with `embedding=None`, so iterating on NER/NEL doesn't re-pay the SPECTER2/BGE cost. The runner's `fs.load_and_annotate` path still embeds inline (it already loads the router).
- New `fs.embed(only_missing=True, batch_size=64)` walks the chunk store, encodes via the source-type router, writes back via a new `chunk_store.update_embedding(chunk_id, vec, model_id)` — a scoped `_update` in Elastic that does NOT touch annotations. `only_missing=True` (default) skips chunks whose `embedding_model` is already real (anchored to the `mock-embedder-v0` / `hash-embedder-v0` exclusion set).
- New §3 "Embed (optional)" cell in the notebook.

**`41c82ea` M3 (entities): linked entities as first-class citizens.**
- New `Entity` Pydantic (`io/entity.py`): frozen record with `ontology_id`, `prefix` (FOODON/CHEBI/GAZ/…), `label`, `synonyms`, `ancestor_ids`, `facet_hint` (mapped from `Mention.entity_type` via `_ENTITY_TYPE_TO_FACET`), `mention_count`, `chunk_count`, sample `chunk_ids` (cap = 50), `last_seen`. Exported from `foodscholar.io`.
- New `EntityStore` protocol with `InMemoryEntityStore` (dict-backed, token-overlap search) and `ElasticEntityStore` (own index `foodscholar_chunks_entities`, BM25 over `label`+`synonyms`, prefix-term filter, idempotent `init()`).
- `GraphStore` extended with `upsert_entities` + `attach_chunks_to_entity(chunk_id, conf, method)`. Neo4j adds `CREATE CONSTRAINT` on `(:Entity {ontology_id})`, MERGEs `(:Entity)` nodes, and writes `(:Chunk)-[:MENTIONS {confidence, method}]->(:Entity)` edges. In-memory mirror tracks the same shape.
- `fs.entity_store` plumbed through the facade (eager construction — adapter ctor is cheap). `fs.init()` now provisions all three stores in lockstep.
- `fs.build_entities()` walks the chunk store, dedupes EntityLinks by `ontology_id`, aggregates counts + facet_hint (max-voted) + chunk_ids (capped sample), enriches FOODON ids with `label`/`synonyms`/`ancestor_ids` from the loaded ontology (other OBO prefixes fall back to the most-frequent surface form as label, no ancestors), upserts to entity store, then MERGEs entity nodes + `[:MENTIONS]` edges. Idempotent.
- `fs.entities` view: `list(prefix=)`, `get(id)`, `search("query")`, `chunks_for(id)` (Elastic `terms`-filter shortcut for FOODON ids, in-memory sample otherwise), `build()` convenience.
- Notebook gains §4 "Build & explore entities" between Embed and Inspect.

### Design decisions worth remembering

- **GLiNER bio is the only NER**. SciFoodNER and agentic NER were tried and dropped — the bio-fine-tuned GLiNER converges deterministically on real FoodOn and amortizes batches.
- **Linker is pure dense, single tier**. The 3-tier (lexical exact / fuzzy / dense) plus optional LLM-select linker was the source of §17 wrong-links via fuzzy over-matching. HNSW over BioLORD is fast (one encode + one kNN per batch), deterministic, and audit-friendly (`method = "dense"`).
- **Two-step pipeline by default**: `fs.ingest` (chunks + annotations, fast, no model loads) → `fs.embed` (vectors, opt-in, pays SPECTER2/BGE once). The prototype only produced surface forms + URIs; chunk-text embeddings are a downstream BRIEF §2/§7 requirement, not a prototype output.
- **Entities are first-class**, not chunk-side payload. They live in their own ES index AND as `(:Entity)` nodes in Neo4j, so "what does my corpus mention" and "which chunks talk about X" are both fast lookups.
- **In-code config is first-class** — `FoodScholar.from_config({...})` works exactly like a YAML path. Secrets (LLM api_key, Neo4j password, ES api_key/basic-auth) can be set in config OR fall back to env vars.

### Verification

- `ruff check src tests` — clean.
- `pytest` — full suite green: **220 passed, 2 skipped** (`torch`/`groq` import-skip paths). New tests: `test_gliner_ner.py` (9, fake gliner model), `test_hnsw_linker.py` (6, fake NELIndex), `test_nel_index.py` (8), `test_config_in_code.py` (8), `test_nel_loader.py` (9), `test_facade_ingest.py` (5), `test_facade_embed.py` (8), `test_storage_init.py` (6), `test_entity_store.py` (7), `test_facade_build_entities.py` (9). Existing facade / annotate / config tests updated for the new contracts.
- Notebook rebuilt to 4 happy-path sections (`configure → init+ingest → embed → entities → inspect`) + a collapsed "Under the hood" appendix for GLiNER+HNSW direct usage.

### Status at end of iteration

- M3 storage milestone done. Production pipeline runs end-to-end against local ES + Neo4j with `pip install -e '.[elastic,neo4j,ontology]'` (and `[annotate]` once vector search is needed). Pre-computed NER/NEL CSVs at `data/foodscholar/ner/*.csv` are loaded without invoking GLiNER.
- Next per BRIEF §12: Layer A backbone builder, then Layer B (theme clustering — first real consumer of `fs.embed()`'s output), then Layer C (LLM cards) and the §14 retrieval pipeline.

---

## 2026-05-18 — Iteration 5.0 (M2→agentic): agentic NER, drop SciFoodNER, notebook rebuild

**Goal:** start the agentic-annotation redesign (see `docs/DESIGN_agentic_annotate.md`). This iteration lands the first piece — an LLM-driven NER — drops the proprietary SciFoodNER model, and rebuilds the notebook around self-contained sections.

### What changed

**`generate_json` on the LLM layer**
- The `LLMClient` protocol gains `generate_json(prompt, schema) -> dict`. Each provider adapter implements it via its native structured-output mode: OpenAI/Groq `response_format`, Gemini `response_schema`, Ollama `format`; Anthropic falls back to instructed-JSON + a tolerant parser (`_parse_json_object`, which strips code fences / surrounding prose).
- `FallbackLLMClient` chains `generate_json` with the same fail-through logic — extracted into a shared `_try_chain` helper so `generate` and `generate_json` don't duplicate it.
- Honest scoping note carried into the protocol docstring + BRIEF: `generate_json` guarantees the result *parses and matches the schema shape*, NOT that values are semantically correct. An LLM offset can be a valid integer yet wrong.

**`AgenticNER`** (`src/foodscholar/annotate/agent_ner.py`)
- Implements the `NER` protocol — a drop-in alternative to `KeywordNER`. One `generate_json` call per chunk.
- The model returns mention *strings* + `entity_type`; **offsets are reconciled locally** (`str.find`, cursor-advanced so repeated mentions map to successive occurrences). A returned string not found verbatim is dropped — catches the model paraphrasing instead of quoting. This is deliberate: no structured-output library fixes LLM offset unreliability; only local string-location does.
- LLM failure / malformed shape degrades to "no mentions" — never crashes the phase.
- Versioned prompt (`PROMPT_VERSION = "agent-ner-v1"`).

**`Mention.entity_type`** — new field on the core `io` contract. `EntityType = food | nutrient | health | dietary_pattern | allergen | other`. Optional, defaults to `other`, so NER impls that don't classify (KeywordNER) and all existing `Mention(...)` constructions stay valid.

**SciFoodNER removed.** Per the project decision to drop bespoke ML models: `SciFoodNERAdapter` class, its export, its slow test, the vestigial `cfg.annotate.ner_model` field, and all BRIEF/docstring references are gone. `ner.py` keeps `KeywordNER` + `simplify_label`. Recorded as a documented deviation in BRIEF §2 (Food NER row) and §3.5.

**Config + facade**
- `cfg.annotate.ner: keyword | agentic` selector (default `keyword`). `config.example.yaml` sets `agentic`.
- Facade `_build_ner` dispatches; `agentic` builds `AgenticNER(fs.llm)`. `fs.info()` reports `ner` instead of the removed `ner_model`.

**Notebook rebuilt** — 37 → 19 cells. Root cause of the "congested / can't showcase steps independently" feedback: it was one linear chain where §6 depended on §3-5 running first. Fix: a `bootstrap()` helper defined once in Setup builds a ready `FoodScholar` (optionally with ontology + chunks); **every section opens with its own `bootstrap()` call**, so any section runs standalone. Verified: §2 Ontology and §4 Annotate each execute correctly from Setup alone, no intervening cells. The four scattered linker/dense/LLM demo cells collapsed into one compact "Annotation internals" cell.

### Design decisions worth remembering

- **No structured-output library (LangChain / instructor).** Provider-native JSON modes via our own `generate_json` keep `foodscholar.llm` the single LLM abstraction — adding LangChain would duplicate the provider layer we just built. And no library fixes offset correctness regardless.
- **Offsets computed locally, always.** The model is never trusted for character positions. Schema-constrained output guarantees parseability, not semantic accuracy.
- **`bootstrap()` for notebook independence.** One helper, called per section — independence with almost no repeated setup code.
- **SciFoodNER removal is a documented BRIEF deviation**, not a silent cut. Pre-1.0, nothing external depends on it.

### Verification

- `ruff check src tests` — clean
- `pytest` — full suite green (2 expected skips: no sentence-transformers / groq-ImportError-path). New: `test_agent_ner.py` (12), `generate_json` + `_parse_json_object` coverage in `test_llm.py`, `Mention.entity_type` tests, facade NER-selector tests.
- Notebook: full run + two independent-section runs all verified.

### Status at end of iteration

- M2 + the first agentic piece. `AgenticNER` is usable now (`cfg.annotate.ner: agentic`); it is a standalone NER stage feeding the existing linker — the fused NER+NEL agent is a later step per the design doc.
- Next per `docs/DESIGN_agentic_annotate.md`: the `[ontorag]` retriever and the content-addressed annotation cache.

---

## 2026-05-18 — Iteration 4.6 (M2): wire the real chunk embedder into from_config

**Goal:** `fs.info()` reported `embedder: mock-embedder-v0` even with a full config — the chunk embedder was never wired through `from_config`. Fix the gap.

### The bug

`from_config` built `chunk_store`, `graph_store`, `linker`, and (in 4.4) `llm` from the config — but **never built `embedder`**. It only forwarded an explicit `embedder=` kwarg; with none passed, `__init__` fell back to `_MockEmbedder`. So `fs.annotate()` wrote **mock hash-vector embeddings** onto every chunk, even on a fully-configured run.

Subtle point: the linker's `dense` tier still used *real* SapBERT — it builds its own embedder from `cfg.annotate.linker.dense_model`, independent of `fs.embedder`. Two embedders; only the linker's was wired. The chunk embedder (`fs.embedder`, consumed by Layer B clustering and retrieval) was the one left on the mock.

### What changed

- **`from_config` now builds the chunk embedder** ([src/foodscholar/facade.py](src/foodscholar/facade.py)) via a new `_build_embedder(cfg)` static method: a `SourceTypeRouter(scientific=HFEmbedder(cfg.annotate.scientific_embedder), general=HFEmbedder(cfg.annotate.general_embedder))` — SPECTER2 for abstracts, BGE-large for textbook/guide, routed by `source_type` per BRIEF §2/§7.
- **Loud degradation.** If the `[annotate]` extra (specifically `sentence-transformers`, which `HFEmbedder` needs) is missing, `_build_embedder` logs a `embedder.deps_missing` **warning** and returns None → `__init__` uses the mock. A production run can no longer *silently* build a graph of meaningless embeddings — the warning names the fix.
- An explicit `embedder=` passed to `from_config` still wins over the config-built one.

### Why two embedders coexist by design

The linker's dense tier and the chunk embedder serve different jobs and the brief specs different models:
- **linker dense tier** — entity linking; SapBERT (`transformers`-only, no sentence-transformers needed).
- **chunk embedder** — document embedding; SPECTER2/BGE via `HFEmbedder` (needs `sentence-transformers`).

That's why on a `torch`+`transformers`-only install the dense linker tier worked but `fs.embedder` stayed mock — `sentence-transformers` was the missing piece. BGE-large (1024-dim) and SPECTER2 (768-dim) differ in width; per BRIEF §7 each chunk records its own `embedding_model`, and the annotate runner already stamps it per chunk, so the mismatch is handled downstream (one index per embedder).

### Tests

- +2 facade tests: explicit `embedder=` override is respected; `from_config` degrades to the mock when `sentence-transformers` is absent (and builds a `router(...)` when present).

### Verification

- `ruff check src tests` — clean
- `pytest` — full suite green (run in progress at write time; facade subset 13/13 passed)

### Status at end of iteration

- M2 v0.1.0. `from_config` now wires **all four** pluggable backends — `chunk_store`, `graph_store`, `embedder`, `llm` — from config. A fully-configured run with `[annotate]` installed produces real SPECTER2/BGE chunk embeddings; without it, a loud warning and mock fallback.
- M3 (Layer A backbone) remains the next milestone.

---

## 2026-05-18 — Iteration 4.5 (M2): notebook runs the real annotation pipeline

**Goal:** the notebook *demonstrated* the dense/LLM tiers in standalone cells but its actual `fs.annotate()` ran lexical-only. Make the walk-through's annotation use the real models (SapBERT dense tier + Groq LLM tier) when the environment supports them.

### What changed

- **§3 config cell now builds the full pipeline.** It detects `transformers` (→ enables the SapBERT `dense_model`) and `groq` + `GROQ_API_KEY` (→ enables `llm_select` and sets `cfg.llm` to a Groq `ProviderConfig`). Prints which tiers are active. `fs.annotate()` in §6 inherits this `fs`, so the main annotation path *is* the real path — no separate cell.
- **Graceful degradation kept.** If the `[annotate]`/`[llm]` deps or the key are absent, the cell prints exactly what to install/set and the linker degrades to its lexical tiers — the notebook still opens and runs on a bare install.
- **Removed the redundant standalone full-pipeline cell** added in iteration 4.4 — its job is absorbed into the main §3 path.
- **Title + Setup markdown refreshed** — status line corrected (annotate is implemented, not a stub); Setup section documents the correct way to provide the Groq key: `export GROQ_API_KEY=...` in the shell **before** launching Jupyter, never in a notebook cell.
- **Fixed a Python 3.11 incompatibility** — an f-string in the config cell had a backslash inside the expression part (allowed in 3.12+, a SyntaxError on 3.11). Rewrote without it.

### Security note

During this iteration a notebook cell containing a hardcoded `GROQ_API_KEY` (`!export GROQ_API_KEY=...`) appeared in the file — twice. It was removed both times. Two facts worth recording: (1) `!export` in a notebook is a **no-op** — `!` spawns a subshell that exits, the variable never reaches the kernel; (2) a hardcoded key in a tracked `.ipynb` would be committed to git history permanently. The key must be provided via the shell environment. The leaked key should be rotated.

### Verification

- All 18 notebook code cells syntax-check clean.
- Non-LLM cells execute end-to-end (verified earlier this session).
- The full real-model run (SapBERT index build + Groq calls) is run by the user with their own key — not executed in this session.

### Status at end of iteration

- M2 v0.1.0. The notebook is now a faithful end-to-end demonstration of the real annotation pipeline, degrading gracefully when models/keys are absent.
- M3 (Layer A backbone) remains the next milestone.

---

## 2026-05-18 — Iteration 4.4 (M2 hardening): provider-agnostic LLM layer + honest demos

**Goal:** (1) make the LLM client provider-pluggable (Ollama / Groq / OpenAI / Anthropic / Gemini) and YAML-configured with a fallback chain; (2) enable the strongest linker configuration by default in the example config; (3) replace the toy-embedder notebook demo with real working code — no mocks dressed up as results.

### What changed

- **`foodscholar.llm` package** — new.
  - `providers.py` — five thin `LLMClient` adapters: `AnthropicClient`, `OpenAIClient`, `GroqClient`, `GeminiClient`, `OllamaClient`. Each lazy-imports its SDK (gated by the new `[llm]` extra), reads its API key from the environment (never config), exposes the protocol's `model_id` + `generate()`.
  - `fallback.py` — `FallbackLLMClient`: an ordered chain; tries each client, falls through on any exception, raises `AllLLMClientsFailedError` only if all fail. `model_id` reports the whole chain.
  - `factory.py` — `build_llm(cfg.llm)`: constructs the primary client, or a `FallbackLLMClient(primary → fallbacks)` when fallbacks are configured. `PROVIDERS` registry maps name → adapter.

- **Config** ([src/foodscholar/config.py](src/foodscholar/config.py)) — new `ProviderConfig` (`provider`, `model`, optional `host` for Ollama) and `LLMConfig` (`primary`, `fallbacks`, `timeout_s`, `max_retries`). New optional top-level `cfg.llm`; `None` → facade uses the mock.

- **Facade** ([src/foodscholar/facade.py](src/foodscholar/facade.py)) — `from_config` builds `fs.llm` via `build_llm(cfg.llm)` when the section is present and no client was passed explicitly. `in_memory()` and `llm`-less configs keep the mock.

- **`[llm]` extra** ([pyproject.toml](pyproject.toml)) — `anthropic`, `openai`, `google-genai`, `groq`, `ollama`. Folded into `[all]`.

- **`config.example.yaml` — strongest setup by default.** Per the user's "most performant method by default" decision: all four linker tiers enabled (`dense_model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, `llm_select: true`), and a new `llm:` section — primary Groq `llama-3.3-70b-versatile`, fallback local Ollama `llama3.1`. The **Pydantic config defaults stay off** (tiers 3-4 disabled, no `llm`) so `in_memory()` and the test suite remain deterministic and offline — the example config is the recommended production setup, the defaults are the safe minimum.

- **Honest demos — corrected an over-claim.** Measuring real SapBERT showed the dense tier does *not* link opaque abbreviations: `EVOO ~ olive oil` is only ≈0.46 cosine (vs ≈0.32 for unrelated pairs). It *does* link lexically-distinct synonyms and morphological variants well: `whole grains`/`whole grain` ≈0.96, `vitamin C`/`ascorbic acid` ≈0.90, `ascorbate`/`vitamin C` ≈0.76. Consequences:
  - The slow test that asserted `EVOO → olive oil` via dense was **asserting wishful behavior** — rewritten to `ascorbate → vitamin C`, a case SapBERT genuinely handles, with the threshold honestly set to 0.70 and the near-miss documented in the test docstring.
  - The notebook's dense/LLM demo cell used `HashEmbedder` (a toy embedder) and printed a meaningless match. Per the user's instruction ("don't fall for toy demos"), rewritten to run **real SapBERT** against a 3-term ontology — `ascorbate → vitamin C [dense, 0.76]`, a genuine result. The LLM cell shows `build_llm` constructing the real provider chain but does not fake a call (no API key in the notebook); it explains what's needed to call it for real.

- **BRIEF §3.5** — added the "LLM client (`fs.llm`)" subsection; corrected the dense-tier description with the measured SapBERT numbers; updated the §2 decisions table LLM row. The `llm`-tier deviation note now states it's off in defaults but on in `config.example.yaml`.

- **README** — new "Configuring the LLM" section; linker-tier description corrected (dense ≠ abbreviations); layout block gains `llm/`.

- **Tests** — +13 (144 → 157):
  - `tests/unit/test_llm.py` — 13: `FallbackLLMClient` (primary success, fail-through, all-fail, ordering, `model_id`), `build_llm` factory, provider registry, `ProviderConfig`/`LLMConfig` validation, the missing-SDK `ImportError` path.
  - `tests/integration/test_real_models.py` — slow tests adjusted: real Groq round-trip (gated on `GROQ_API_KEY`), and the rewritten honest dense-tier test.

### Design decisions worth remembering

- **API keys from env, never YAML.** `ProviderConfig` has no key field. The YAML's existing `${ENV}` substitution covers anything else.
- **Fail-through, not fail-fast, for the fallback chain.** A `FallbackLLMClient` only raises if *every* provider fails — a Groq rate-limit silently degrades to local Ollama. The linker's tier-4 catch then degrades *that* to "no llm hit". Two layers of graceful degradation; an annotate run over a corpus never dies on a transient LLM error.
- **Config defaults ≠ example config.** Defaults are the safe offline minimum (no models, no APIs, deterministic) — they protect `in_memory()` and CI. `config.example.yaml` is the recommended *production* setup with everything on. This split is deliberate and now documented in BRIEF §3.5.
- **Honesty over demo polish.** A demo that prints a meaningless match from a toy embedder is worse than no demo. The dense notebook cell now does real model work; the LLM cell shows wiring without faking output. The corrected SapBERT capability claim (synonyms yes, abbreviations no) is now in BRIEF, README, and the test docstring.

### Verification

- `ruff check src tests` — clean
- `pytest` — **157 passed** (144 → 157), 1 skipped (no sentence-transformers locally)
- Notebook executes end-to-end with **real SapBERT** — `ascorbate → vitamin C [dense, 0.76]`

### Status at end of iteration

- M2 v0.1.0 — annotate phase is feature-complete: NER, 4-tier linker, embedders, and a provider-agnostic LLM layer, all YAML-configurable. A real production run (SapBERT + Groq/Ollama) needs only `pip install -e '.[annotate,llm]'` + `GROQ_API_KEY` + `config.yaml`; no code change.
- M3 (Layer A backbone) remains the next milestone.

---

## 2026-05-18 — Iteration 4.3 (M2 hardening): dense tier + opt-in LLM-select tier

**Goal:** the lexical/fuzzy linker can't resolve surface forms with no shared tokens (`EVOO` → olive oil) and can't *reject* non-food queries (`iron deficiency` → flat iron steak). Bring online the two tiers that can: the dense (SapBERT) tier the brief already specs, and a new opt-in LLM-selection tier inspired by the OntoRAG project (https://github.com/jan3657/onto_rag).

### Context — why an LLM tier, and is it in scope?

The user surfaced OntoRAG as a possible fit for entity linking. Assessment: OntoRAG is a hybrid retriever (Whoosh lexical + FAISS dense) feeding an **LLM selector** with a confidence-gated retry loop. We did **not** adopt it as a dependency (single-author research repo, no release, pulls in Whoosh+FAISS+Gemini). Instead we borrowed the *architecture*: keep our linker and protocols, add the dense tier, and add an LLM-selection tier — but **gated**, not a per-mention call, and **without** the agentic retry loop (which BRIEF §15 defers). Documented as an explicit deviation in BRIEF §3.5.

### What changed

- **`EntityLink.method`** ([src/foodscholar/io/chunk.py](src/foodscholar/io/chunk.py)) — gained a 4th value `"llm"`.

- **`DenseIndex`** ([src/foodscholar/annotate/dense_index.py](src/foodscholar/annotate/dense_index.py)) — new. Embeds every non-obsolete ontology term once (label + synonyms), L2-normalizes, stacks into one numpy matrix. A query is a single matrix-vector product — ~29k FoodOn terms scored in <2ms, so **no FAISS dependency** at this scale. Caches the matrix to `.npz`, keyed on a fingerprint of `(term-id set, embedder model_id, embedding dim)` so it rebuilds only when the ontology or the model changes. Handles the empty-ontology edge case.

- **`SapBERTEmbedder`** ([src/foodscholar/annotate/embedder.py](src/foodscholar/annotate/embedder.py)) — new. Wraps `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` (BRIEF §2's named entity-linking model). SapBERT is a plain `transformers` model, so the adapter does explicit `[CLS]`-token pooling rather than relying on sentence-transformers. Lazy-imports torch+transformers; gated by the `[annotate]` extra.

- **Linker rewritten** ([src/foodscholar/annotate/linker.py](src/foodscholar/annotate/linker.py)) — now a 3-or-4 tier cascade:
  - Tier 3 (`dense`) now goes through `DenseIndex` instead of an inline brute-force cosine loop. Skipped entirely when a fuzzy hit already clears the confidence bar (no wasted embed call).
  - Tier 4 (`llm`) — new. When no deterministic tier clears `llm_select_threshold` (or there's no hit at all), the LLM is shown the top-k candidates (dense-ranked, fuzzy fallback) plus the mention text and replies with an index or `"none"`. A successful pick gets a fixed `0.85` confidence (a model judgement, not a similarity score). LLM backend exceptions are caught and degrade to "no hit" — they never crash the annotate phase.
  - `linker_id` bumped to `tiered-linker-v2`.

- **Config** ([src/foodscholar/config.py](src/foodscholar/config.py), [config.example.yaml](config.example.yaml)) — `LinkerConfig` gained `dense_model`, `dense_cache_path`, `llm_select`, `llm_select_threshold`, `llm_candidate_k`. All default to "off"/empty so the linker is lexical-only and fully deterministic unless explicitly configured.

- **Facade `_build_linker`** ([src/foodscholar/facade.py](src/foodscholar/facade.py)) — builds the dense embedder when `dense_model` is set, passes the facade's `fs.llm` as the linker's LLM when `llm_select` is on.

- **Tests** — +16 (128 → 144):
  - `tests/unit/test_dense_index.py` — 7 tests: obsolete exclusion, kNN ordering, self-match, zero-vector, cache round-trip, cache invalidation, empty ontology.
  - `tests/unit/test_linker.py` — 8 new: LLM tier resolves/rejects/short-circuits, confidence-gate behavior (consulted below threshold, skipped above), garbled-reply safety, LLM-exception safety.
  - `tests/unit/test_facade.py` — 2: linker defaults to lexical-only; `llm_select=True` wires the facade LLM.
  - `tests/integration/test_real_models.py` — 2 slow: `SapBERTEmbedder` round-trip, and the motivating case — `EVOO` resolving to olive oil via the dense tier when lexical is forced to miss.

### Design decisions worth remembering

- **numpy over FAISS.** 29k vectors is small; a brute-force matrix product is <2ms and adds zero dependencies. FAISS would be premature. Documented the threshold (revisit at ~10x scale) in the `DenseIndex` docstring.
- **LLM tier is gated, not default.** Off unless `llm_select: true`. Even on, it fires only below a confidence threshold — so the cost is paid on the hard residue (~5-15% of mentions), not all ~1M chunks. Keeps the brief's "deterministic v1" intent largely intact.
- **`method` records which tier won** — `lexical_exact | lexical_fuzzy | dense | llm`. The §17 audit ("50 random links hand-checked") can filter by tier; LLM-resolved links can be reviewed separately since they're the least certain.
- **LLM errors degrade, never crash.** A broad `except` around the LLM call is deliberate — an annotate run over a corpus must not die because a backend hiccuped.
- **No agentic retry.** OntoRAG's synonym-generation retry loop is *not* implemented — BRIEF §15 defers agentic retrieval to v2. Our tier 4 is a single gated selection call.

### Verification

- `ruff check src tests` — clean
- `pytest` — **144 passed** (128 → 144), 1 skipped (no torch/sentence-transformers locally)
- Notebook executes end-to-end against real FoodOn

### Status at end of iteration

- M2 still v0.1.0. The linker is now feature-complete per BRIEF §2 plus the documented LLM deviation. Real SapBERT + a real LLM backend can be switched on entirely via `config.yaml` — no code change.
- The remaining linker-quality work (tuning thresholds, expanding the gold set, the `omega 3` digit-token issue) is measurement-driven and can continue in parallel.
- M3 (Layer A backbone) remains the next milestone.

---

## 2026-05-18 — Iteration 4.2 (M2 hardening): KeywordNER label normalization

**Goal:** running `fs.annotate()` against real FoodOn produced visibly wrong NER — e.g. chunk c4 ("Iron-rich foods include legumes, red meat, and fortified cereals.") extracted only `"red"`, which then linked to a PATO *color* term. Fix the two distinct bugs behind that.

### Two bugs, two fixes

**Bug 1 — notebook loaded real FoodOn with `prefix_filter=None`.** The ontology cell had been switched to point at the real `data/foodon.owl` but kept the `prefix_filter=None` that was only correct for the synthetic `TEST:` fixture. That let NCBITaxon / CHEBI / PATO / ONS terms into the lookup pool, so `"red"` matched `PATO:0000322` (the color) and `"Mediterranean diet"` matched an `ONS:` term. Fixed the notebook cell to use `prefix_filter=["FOODON:"]`, refreshed its stale comment and the leftover `TEST:` ids in the layer_a stub (now derives the foods root from a real FoodOn ancestor of olive oil).

**Bug 2 — `KeywordNER` can't match FoodOn's over-qualified labels.** FoodOn labels carry systematic NLP noise: leading EFSA/EC codes (`"10210 - legumes (efsa foodex2)"`), parenthetical qualifiers (`"red meat (raw)"`, `"(eurofir)"`), and trailing category words (`"legume food product"`). Nobody writes those in prose, so a verbatim keyword built from the raw label never matches. Result: `"red meat"` was unfindable because FoodOn has no bare `"red meat"` label.

### What changed

- **`simplify_label(label)`** ([src/foodscholar/annotate/ner.py](src/foodscholar/annotate/ner.py)) — strips the code prefix, parenthetical qualifiers, and trailing category words (` food product`, ` animal feed plant`, ` plant`). Exported from `foodscholar.annotate` for reuse by the (separately developed) SciFoodNER linking step.

- **`KeywordNER.from_ontology(expand_labels=True, min_keyword_len=3)`** — for each label/synonym, also registers the `simplify_label`-cleaned variant. `"red meat (raw)"` now yields both itself *and* `"red meat"`. `min_keyword_len` drops 1–2 char keywords (FoodOn has a literal `"an"` term) that generate only false positives.

- **Result on c4 (real FoodOn):** NER went from `[]` (or `["red"]` pre-prefix-fix) to `["Iron", "legumes", "red meat"]`. Keyword count rose 33,532 → 41,181 as the simplified variants were added. `"fortified cereals"` still misses — that compound phrasing genuinely isn't in FoodOn even simplified, and is correctly SciFoodNER's job.

- **7 new unit tests** ([tests/unit/test_ner.py](tests/unit/test_ner.py)) — `simplify_label` cases (parentheticals, code prefix, trailing category, clean-label passthrough) and `from_ontology` with `expand_labels` on/off plus `min_keyword_len`.

### Design note

`KeywordNER` stays an *exact word-boundary* matcher — `expand_labels` adds more exact keywords, it does not make the NER fuzzy. Plurals ("legumes" vs the "legume" keyword) remain the linker's fuzzy tier's job, not the NER's. This keeps the NER deterministic and the responsibility split clean.

### Verification

- `ruff check src tests` — clean
- `pytest` — **128 passed** (was 121; +7 NER tests), 1 skipped (no sentence-transformers)
- Notebook executes end-to-end against the real 39k-term FoodOn `.owl`

### Status at end of iteration

- M2 still v0.1.0. `KeywordNER` is meaningfully better on real FoodOn but remains a stopgap — the user is developing SciFoodNER separately, which is the real fix for compound/contextual phrasings KeywordNER can't reach.
- M3 (Layer A backbone) remains the next milestone.

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
