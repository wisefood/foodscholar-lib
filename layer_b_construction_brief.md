# FoodScholar — Layer B Construction Brief

**Status**: Implementation brief, hand to Claude Code alongside existing repo + progress log.
**Prerequisites**: Layer A complete and audited (PASS on all critical gates). Chunks have FoodOn entity_links with confidence; Elastic has chunk embeddings; Neo4j has shelves and attachments.
**Out of scope**: Layer C card generation; cross-shelf themes; ChEBI/MONDO integration.

---

## 0. Ground truth and conventions

This brief refines `foodscholar_package_brief.md` Section 15 (Layer B dual-clustering detail) and Section 4 (Pydantic Theme model). Claude Code should treat that brief as authoritative for naming, data contracts, and storage model. This document specifies the construction work concretely.

**Progress log**: append a one-line entry per phase boundary to the existing log. Entry format:
```
YYYY-MM-DD  phase-name  artifact_id=...  config_hash=...  notes="..."
```

**Determinism requirement**: every Leiden run uses a fixed `random_state` (config-driven). Two runs with the same inputs and config must produce identical theme assignments.

---

## 1. What Layer B does

For each shelf with ≥ `min_chunks_for_clustering` (default 100) chunks attached, run **two parallel community detections** on different graphs over the same node set, then merge their outputs into a unified theme list.

- **Pass 1 — Similarity**: graph edges weighted by chunk-embedding cosine similarity. Captures topical neighborhoods.
- **Pass 2 — Relatedness**: graph edges weighted by count of shared high-confidence FoodOn IDs between chunks. Captures entity co-occurrence neighborhoods (SiReRAG-inspired).
- **Merge**: pair every (S<sub>i</sub>, R<sub>j</sub>) candidate, compute combined Jaccard similarity over chunks and entities, merge above `dedupe_threshold` (default 0.70).

Both passes use **Leiden** (`leidenalg` + `python-igraph`). HDBSCAN remains the documented fallback for the similarity pass if Leiden produces a degenerate single community at scale, but should not be invoked in v1 unless that happens.

---

## 2. Module layout

Under `foodscholar/layer_b/` (already scaffolded per package brief):

```
foodscholar/layer_b/
  __init__.py              # public API: build_layer_b(facet, …)
  models.py                # Pydantic: Theme, ThemeCandidate, MergeDecision, LayerBArtifact
  semantic_graph.py        # Pass 1 graph construction (kNN over embeddings)
  relatedness_graph.py     # Pass 2 graph construction (entity bridges)
  community.py             # Leiden runner shared by both passes
  label.py                 # theme labeling (keyword-based, optional LLM polish)
  merge.py                 # cross-pass candidate merging
  builder.py               # orchestrator: per-shelf dual pass → unified themes
  persist.py               # write to Neo4j + denormalize to Elastic
  audit.py                 # quality gates (see §10)
foodscholar/cli/
  build_layer_b.py         # `foodscholar build-layer-b --facet foods`
tests/layer_b/             # unit + integration tests, see §11
```

**No I/O in pure-logic modules.** `semantic_graph.py`, `relatedness_graph.py`, `community.py`, `merge.py`, `label.py` take typed inputs and return typed outputs. All I/O is in `persist.py` and the builder orchestrator.

---

## 3. Pydantic data contracts

In `foodscholar/layer_b/models.py`:

```python
from pydantic import BaseModel, Field
from typing import Literal

ChunkId = str
ShelfId = str
ThemeId = str
FoodonId = str

DiscoveryPass = Literal["similarity", "relatedness", "merged"]
DiscoveredBy  = Literal["leiden", "hdbscan", "bertopic"]


class ThemeCandidate(BaseModel):
    """Intermediate object emitted by a single pass. Not persisted."""
    pass_name: Literal["similarity", "relatedness"]
    chunk_ids: set[ChunkId]
    foodon_ids: set[FoodonId]       # union of high-conf entity links across chunks
    centroid_embedding: list[float] | None = None
    discovered_by: DiscoveredBy = "leiden"


class MergeDecision(BaseModel):
    """Records why two candidates were (or were not) merged."""
    similarity_candidate_idx: int
    relatedness_candidate_idx: int
    chunk_jaccard: float
    entity_jaccard: float
    combined_similarity: float
    merged: bool


class Theme(BaseModel):
    """Persisted theme node (matches existing Theme model in package brief §4)."""
    theme_id: ThemeId
    label: str
    parent_shelf_id: ShelfId
    parent_theme_id: ThemeId | None = None
    facet: str
    chunk_count: int
    foodon_id_signature: list[FoodonId]   # top-N most representative entities
    keyword_terms: list[str]                # for navigation hover/preview
    discovery_pass: DiscoveryPass
    discovered_by: DiscoveredBy
    config_hash: str
    version: str                             # e.g. "v0.1"


class LayerBArtifact(BaseModel):
    """Audit log for one builder run."""
    artifact_id: str
    facet: str
    config_hash: str
    n_shelves_themed: int
    n_shelves_skipped: int                  # below min_chunks
    n_themes_total: int
    n_themes_by_pass: dict[DiscoveryPass, int]
    leiden_seed: int
    started_at: str
    finished_at: str
```

---

## 4. Algorithm details

### 4.1 Pass 1 — Similarity graph (`semantic_graph.py`)

```python
def build_similarity_graph(
    chunks: list[ChunkRecord],
    embeddings: dict[ChunkId, np.ndarray],
    cfg: SimilarityConfig,
) -> ig.Graph:
    """
    Returns an igraph weighted undirected graph.
    Nodes: chunks attached to the shelf.
    Edges: top-k mutual nearest neighbors with cosine >= edge_threshold.
    """
    ids = [c.chunk_id for c in chunks]
    M = np.stack([embeddings[i] for i in ids])
    # M is L2-normalized → cosine = dot product
    sims = M @ M.T
    np.fill_diagonal(sims, -1.0)

    # top-k per row
    k = cfg.knn_k
    topk_idx = np.argpartition(-sims, k, axis=1)[:, :k]

    edges = set()
    for i, neighbors in enumerate(topk_idx):
        for j in neighbors:
            if sims[i, j] < cfg.edge_threshold:
                continue
            # mutual-neighbor requirement
            if i in topk_idx[j]:
                edges.add(tuple(sorted((i, j))))

    g = ig.Graph()
    g.add_vertices(len(ids))
    g.vs["chunk_id"] = ids
    g.add_edges(list(edges))
    g.es["weight"] = [sims[i, j] for (i, j) in edges]
    return g
```

**Config:**
```yaml
similarity:
  knn_k: 15
  edge_threshold: 0.55         # below this, edges are noise
  require_mutual: true          # mutual kNN is more conservative; recommended
```

### 4.2 Pass 2 — Relatedness graph (`relatedness_graph.py`)

The critical piece. Three knobs make or break Pass 2: which entities count, how to down-weight ubiquitous ones, and what minimum-shared threshold creates an edge.

```python
def build_relatedness_graph(
    chunks: list[ChunkRecord],
    cfg: RelatednessConfig,
) -> ig.Graph:
    """
    Nodes: chunks attached to the shelf.
    Edges: weight = sum over shared FoodOn IDs of (1 / log(1 + doc_freq[id])).
    Inspired by TF-IDF intuition: rare entities count more than ubiquitous ones.
    """
    # 1. Collect high-confidence FoodOn IDs per chunk
    chunk_entities: dict[ChunkId, set[FoodonId]] = {}
    for c in chunks:
        ids = {
            link.foodon_id
            for link in c.entity_links
            if link.confidence >= cfg.tau_strict
        }
        chunk_entities[c.chunk_id] = ids

    # 2. Compute document frequency per entity (across this shelf's chunks)
    from collections import Counter
    doc_freq = Counter()
    for ents in chunk_entities.values():
        for e in ents:
            doc_freq[e] += 1
    n_chunks = len(chunks)

    # 3. Exclude ubiquitous IDs (appear in too many chunks)
    excluded = {
        e for e, f in doc_freq.items()
        if f / n_chunks > cfg.max_doc_frequency
    }
    # also blacklist the shelf's own canonical IRI + facet root IRIs
    excluded |= set(cfg.always_exclude_iris)

    # 4. Build edges
    edges = []
    chunk_ids = list(chunk_entities.keys())
    for i, ci in enumerate(chunk_ids):
        for j in range(i + 1, len(chunk_ids)):
            cj = chunk_ids[j]
            shared = (chunk_entities[ci] & chunk_entities[cj]) - excluded
            if len(shared) < cfg.min_shared_ids:
                continue
            # IDF-style weighting
            w = sum(1.0 / np.log(1 + doc_freq[e]) for e in shared)
            edges.append((i, j, w))

    g = ig.Graph()
    g.add_vertices(len(chunk_ids))
    g.vs["chunk_id"] = chunk_ids
    g.add_edges([(i, j) for i, j, w in edges])
    g.es["weight"] = [w for i, j, w in edges]
    return g
```

**Config:**
```yaml
relatedness:
  tau_strict: 0.80              # confidence floor for participating entities
  min_shared_ids: 2              # require ≥ 2 shared entities for an edge
  max_doc_frequency: 0.40        # exclude entities appearing in >40% of shelf's chunks
  always_exclude_iris:
    - "synthetic:Foods"
    - "FOODON:00001002"          # food product (your top shelf, ubiquitous)
```

**Why each knob matters:**
- `tau_strict` (0.80): noisy links shouldn't shape themes. Should match Layer A's propagation threshold.
- `min_shared_ids` (2): single shared entity is weak signal (every cow_milk chunk shares "cow milk" itself). Two creates real co-mention.
- `max_doc_frequency` (0.40): entities appearing in nearly every chunk of the shelf carry no discriminative signal. Drop them.
- `always_exclude_iris`: top-of-hierarchy classes that appear via ancestor propagation should never be edge-creators.

If Pass 2 produces ~0 themes after merge, debug in this order: (1) check `min_shared_ids` — try 1; (2) check `max_doc_frequency` — try 0.6; (3) inspect `excluded` set — is it eating too many entities?

### 4.3 Community detection (`community.py`)

```python
def run_leiden(
    g: ig.Graph,
    cfg: LeidenConfig,
) -> list[set[int]]:
    """
    Returns a list of communities, each a set of node indices.
    Filters out communities below min_community_size.
    """
    import leidenalg as la
    partition = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=cfg.resolution,
        n_iterations=cfg.n_iterations,
        seed=cfg.random_state,
    )
    communities = [set(c) for c in partition if len(c) >= cfg.min_community_size]
    return communities
```

**Config:**
```yaml
leiden:
  resolution: 1.0                # higher = more, smaller communities
  n_iterations: 10               # default for Leiden refinement
  min_community_size: 15         # below this, the "community" is noise; drop
  random_state: 42
```

Both passes share this runner. Pass-specific configs only differ in graph construction, not in Leiden parameters (unless empirical results force divergence; document if so).

### 4.4 Merge step (`merge.py`)

```python
def merge_candidates(
    sim_cands: list[ThemeCandidate],
    rel_cands: list[ThemeCandidate],
    cfg: MergeConfig,
) -> tuple[list[Theme], list[MergeDecision]]:
    """
    Pairwise merge across passes. Returns final theme list + audit decisions.
    """
    decisions = []
    merged_pairs: set[tuple[int, int]] = set()

    for i, s in enumerate(sim_cands):
        for j, r in enumerate(rel_cands):
            chunk_j = jaccard(s.chunk_ids, r.chunk_ids)
            entity_j = jaccard(s.foodon_ids, r.foodon_ids)
            combined = cfg.chunk_weight * chunk_j + cfg.entity_weight * entity_j

            decisions.append(MergeDecision(
                similarity_candidate_idx=i,
                relatedness_candidate_idx=j,
                chunk_jaccard=chunk_j,
                entity_jaccard=entity_j,
                combined_similarity=combined,
                merged=combined >= cfg.dedupe_threshold,
            ))

            if combined >= cfg.dedupe_threshold:
                merged_pairs.add((i, j))

    # Build final themes
    themes = []
    sim_merged = {i for i, _ in merged_pairs}
    rel_merged = {j for _, j in merged_pairs}

    # Merged themes: union chunks + entities from both candidates
    for i, j in merged_pairs:
        s, r = sim_cands[i], rel_cands[j]
        themes.append(build_theme(
            chunks=s.chunk_ids | r.chunk_ids,
            entities=s.foodon_ids | r.foodon_ids,
            discovery_pass="merged",
        ))

    # Unmerged similarity-only themes
    for i, s in enumerate(sim_cands):
        if i not in sim_merged:
            themes.append(build_theme(
                chunks=s.chunk_ids,
                entities=s.foodon_ids,
                discovery_pass="similarity",
            ))

    # Unmerged relatedness-only themes
    for j, r in enumerate(rel_cands):
        if j not in rel_merged:
            themes.append(build_theme(
                chunks=r.chunk_ids,
                entities=r.foodon_ids,
                discovery_pass="relatedness",
            ))

    return themes, decisions
```

**Config:**
```yaml
merge:
  chunk_weight: 0.6              # how much chunk overlap counts
  entity_weight: 0.4             # vs how much entity overlap counts
  dedupe_threshold: 0.70
```

**Edge case to handle:** if `sim_cand` `i` is merge-eligible with both `rel_cand` `j` and `rel_cand` `k`, prefer the higher `combined_similarity`. Same the other way. Use a greedy assignment in descending order of combined similarity. Implement, don't naively allow many-to-many.

### 4.5 Theme labeling (`label.py`)

Two strategies, configurable:

**Strategy A — Keyword (free, fast):** c-TF-IDF over chunks in the theme.

```python
def label_by_keywords(theme: Theme, chunks: list[ChunkRecord], k=5) -> list[str]:
    """Return top-k discriminative terms via class-based TF-IDF."""
    # Use scikit-learn TfidfVectorizer with the corpus = all chunks across all themes
    # The theme's "document" is the concatenation of its chunks' text.
    # Top-k terms by TF-IDF score = the keyword label.
```

**Strategy B — LLM polish (~$0.20 per run, optional):** Pass keyword terms + 3 sample chunks to Haiku, ask for a 3-5 word human-readable label.

```python
LABEL_PROMPT = """Given the keyword terms and chunk samples from a theme in a
nutrition research knowledge graph, write a 3-5 word human-readable label
for navigation purposes.

Theme keywords: {keywords}
Sample chunks:
1. {chunk_1}
2. {chunk_2}
3. {chunk_3}

Output a single label, no quotes, no explanation.
"""
```

Default to keyword-only for v1. Add LLM polish only if hand-review of keyword labels shows they're unintelligible.

---

## 5. Configuration

Extend `config/projection.yaml`:

```yaml
layer_b:
  min_chunks_for_clustering: 100
  facet_scope:
    foods: enabled
    health: deferred              # waiting on MONDO
    nutrients: deferred           # waiting on ChEBI
    dietary_patterns: deferred
    allergies: deferred
    sustainability: deferred

  similarity:
    knn_k: 15
    edge_threshold: 0.55
    require_mutual: true

  relatedness:
    tau_strict: 0.80
    min_shared_ids: 2
    max_doc_frequency: 0.40
    always_exclude_iris:
      - "synthetic:Foods"
      - "FOODON:00001002"

  leiden:
    resolution: 1.0
    n_iterations: 10
    min_community_size: 15
    random_state: 42

  merge:
    chunk_weight: 0.6
    entity_weight: 0.4
    dedupe_threshold: 0.70

  labeling:
    strategy: keyword             # "keyword" | "keyword_then_llm"
    top_keywords: 5

  audit:
    relatedness_zero_count_warns: true
    target_themes_per_shelf: [3, 12]  # warn if outside this range
```

The whole `layer_b` block participates in the config hash. Changing any knob produces a new artifact with a new hash.

---

## 6. Builder orchestrator (`builder.py`)

```python
def build_layer_b(
    facet: str,
    chunk_store: ChunkStore,
    graph_store: GraphStore,
    embedder: Embedder,
    cfg: LayerBConfig,
) -> LayerBArtifact:
    """
    Top-level orchestrator. For each shelf in facet that meets min_chunks:
      1. fetch chunks + embeddings + entity_links
      2. run similarity + relatedness in parallel
      3. merge candidates
      4. label themes
      5. persist
    Records a LayerBArtifact with run-level stats.
    """
    artifact = LayerBArtifact(...)

    for shelf in graph_store.shelves_in_facet(facet):
        if shelf.chunk_count < cfg.min_chunks_for_clustering:
            artifact.n_shelves_skipped += 1
            continue

        chunks = chunk_store.get_chunks_for_shelf(shelf.id)
        embeddings = {c.chunk_id: c.embedding for c in chunks}

        # Run both passes (could be parallelized with asyncio later)
        sim_graph = build_similarity_graph(chunks, embeddings, cfg.similarity)
        rel_graph = build_relatedness_graph(chunks, cfg.relatedness)

        sim_communities = run_leiden(sim_graph, cfg.leiden)
        rel_communities = run_leiden(rel_graph, cfg.leiden)

        sim_cands = communities_to_candidates(sim_communities, sim_graph, chunks, pass_name="similarity")
        rel_cands = communities_to_candidates(rel_communities, rel_graph, chunks, pass_name="relatedness")

        themes, decisions = merge_candidates(sim_cands, rel_cands, cfg.merge)
        themes = label_themes(themes, chunks, cfg.labeling)

        persist_themes(themes, shelf.id, graph_store, chunk_store, cfg)

        artifact.n_themes_total += len(themes)
        for t in themes:
            artifact.n_themes_by_pass[t.discovery_pass] += 1
        artifact.n_shelves_themed += 1

    return artifact
```

Use `asyncio` or `concurrent.futures` to parallelize over shelves once the per-shelf loop is correct. Both Leiden passes within a shelf can also run concurrently — they're independent.

---

## 7. Persistence (`persist.py`)

### 7.1 Neo4j writes

```cypher
// Per theme
MERGE (t:Theme {theme_id: $theme_id})
SET t.label = $label,
    t.parent_shelf_id = $parent_shelf_id,
    t.facet = $facet,
    t.chunk_count = $chunk_count,
    t.discovery_pass = $discovery_pass,
    t.discovered_by = $discovered_by,
    t.keyword_terms = $keyword_terms,
    t.foodon_id_signature = $foodon_id_signature,
    t.config_hash = $config_hash,
    t.version = $version;

// Shelf → theme
MATCH (s:Shelf {shelf_id: $parent_shelf_id}), (t:Theme {theme_id: $theme_id})
MERGE (s)-[:HAS_THEME]->(t);

// Chunk → theme (multi-label, primary flag on one)
UNWIND $chunk_attachments AS att
MATCH (c:Chunk {chunk_id: att.chunk_id}), (t:Theme {theme_id: att.theme_id})
MERGE (c)-[r:THEME_OF]->(t)
SET r.primary = att.primary,
    r.weight = att.weight;
```

### 7.2 Elastic denormalization

For each chunk attached to one or more themes, update its document:

```json
POST /foodscholar_chunks/_update/{chunk_id}
{
  "doc": {
    "theme_ids": ["foods/cow_milk/calcium_bone_health", ...],
    "primary_theme_id": "foods/cow_milk/calcium_bone_health"
  }
}
```

Use the existing bulk update pattern from Layer A's attach step. Same idempotency guarantees.

### 7.3 Theme ID scheme

```
{facet}/{parent_shelf_id_short}/{label_slug}_{discovery_pass_initial}{seq}
```

Example: `foods/cow_milk/calcium_bone_health_m1` (m = merged).
The `_m1`, `_s1`, `_r1` suffixes ensure determinism when labels happen to collide.

---

## 8. CLI

```
foodscholar build-layer-b --facet foods --config config/projection.yaml
foodscholar build-layer-b --facet foods --shelf cow_milk    # single-shelf dry run
foodscholar build-layer-b --facet foods --dry-run            # compute themes, don't persist
```

Output: the `LayerBArtifact` summary + per-shelf breakdown printed to stderr.

---

## 9. Implementation phases

Four phases, ~10 working days total. Each phase ends with a progress log entry.

### Phase 1 — Single-shelf similarity pass (2 days)

- Scaffold `layer_b/` modules + Pydantic models.
- Implement `semantic_graph.py` + `community.py` (Leiden runner).
- Implement keyword-based labeling.
- Run on **one shelf only** (`cow_milk`, 522 chunks). No persistence yet.
- Hand-inspect the 3–12 themes produced. Are they coherent?
- **Exit criterion**: themes for `cow_milk` look recognizable on hand audit; `discovery_pass = "similarity"` for all of them; keyword labels readable.

Progress log: `phase=layer-b/p1-similarity-cow-milk artifact_id=... n_themes=N`

### Phase 2 — Relatedness pass on the same shelf (2 days)

- Implement `relatedness_graph.py` with the entity-bridge edge weighting.
- Run on `cow_milk` only; emit candidates only (no merge yet).
- Inspect: are the relatedness candidates entity-coherent? Compare against the similarity candidates by hand — do they overlap meaningfully, or are they orthogonal?
- **Exit criterion**: ≥ 1 relatedness candidate that's genuinely entity-anchored (shares specific FoodOn IDs across chunks) and not just a duplicate of a similarity cluster.

**Common Phase 2 pitfalls** (debug in this order):
- 0 communities at default config: `min_shared_ids` likely too aggressive. Try 1.
- 1 giant community covering everything: `max_doc_frequency` too lenient or `always_exclude_iris` missing top-of-hierarchy IDs.
- Communities indistinguishable from similarity pass: try lowering `tau_strict` to 0.7 to widen entity coverage.

Progress log: `phase=layer-b/p2-relatedness-cow-milk artifact_id=... n_candidates_sim=X n_candidates_rel=Y`

### Phase 3 — Merge step + persistence (3 days)

- Implement `merge.py` with greedy pair assignment (highest combined similarity first).
- Implement `persist.py` (Neo4j writes + Elastic denormalization).
- Implement audit checks (see §10).
- Run end-to-end on `cow_milk` and verify cross-store parity.
- **Exit criterion**: cross-store consistency PASS on the single shelf; themes visible in both Neo4j and Elastic.

Progress log: `phase=layer-b/p3-merge-persist artifact_id=... n_themes_total=Z`

### Phase 4 — Roll out to all eligible shelves (3 days)

- Run on every shelf with ≥ `min_chunks_for_clustering` (~15 shelves).
- Parallelize across shelves and across passes within a shelf.
- Run full audit; check pass-distribution gates.
- Spot-check 20 random themes by hand (5 from each pass type + 5 from any source).
- **Exit criterion**: all audit gates PASS; theme distribution shows meaningful contribution from each pass; hand audit shows ≥ 75% coherent themes.

Progress log: `phase=layer-b/p4-full-rollout artifact_id=... n_shelves_themed=N n_themes=M by_pass={similarity:X, relatedness:Y, merged:Z}`

---

## 10. Audit and quality gates (`audit.py`)

Extend the existing audit framework. Layer B adds these checks:

| Check | Level | Threshold |
|---|---|---|
| theme cross-store parity (Neo4j `THEME_OF` ↔ Elastic `theme_ids`) | CRITICAL | parity = 1.0 |
| no dangling `THEME_OF` edges | CRITICAL | dangling = 0 |
| each themed shelf has ≥ 1 theme | CRITICAL | failures = 0 |
| each themed chunk has exactly 1 `primary` theme per shelf | CRITICAL | failures = 0 |
| no theme has zero chunks | CRITICAL | failures = 0 |
| target themes per shelf (3–12) | WARNING | shelves outside range ≤ 20% |
| each pass contributes ≥ 1 theme overall | WARNING | both passes > 0 |
| `merged` theme rate between 0.20 and 0.80 | WARNING | rate in range |

If `merged` rate is below 0.20, the two passes are barely overlapping — the relatedness graph is probably too sparse. Above 0.80, it's not earning its compute.

---

## 11. Testing strategy

### Unit tests (`tests/layer_b/`)

- `test_semantic_graph.py`: kNN edges, mutual-neighbor flag, edge threshold cutoff
- `test_relatedness_graph.py`: IDF weighting, doc-frequency exclusion, min-shared cutoff
- `test_community.py`: Leiden runs deterministic with fixed seed; min-community-size filter works
- `test_merge.py`: greedy pair assignment correctness; merged/sim/rel partition correctness
- `test_label.py`: keyword extraction stability; LLM polish (mock) returns expected shape

### Integration test (`tests/layer_b/test_pipeline_e2e.py`)

Fixture: a hand-built mini corpus of 60 chunks with engineered properties:
- 3 known topical clusters (chunks share embeddings)
- 2 known entity clusters (chunks share specific FoodOn IDs)
- 1 cluster that overlaps both
- 10 background chunks with no clear cluster

Expected output: 3–4 themes; at least 1 `merged`, 1 `similarity`-only, 1 `relatedness`-only.

### Eval set

Hand-curate 5 shelves' worth of expected themes. After Phase 4, score the run by:
- How many expected themes did Layer B find? (recall)
- How many surprise themes appeared? (acceptable if coherent)
- Were any expected themes split or fragmented? (quality bug)

---

## 12. Cost and latency

For your corpus (13k chunks, ~15 themed shelves, ~500 chunks per shelf average):

- **Embedding lookups**: already indexed in Elastic. Cost: ~0.
- **Graph construction (both passes)**: O(N²) per shelf. At N=500, ~250k cosine + ~250k entity-set ops. Under 5 seconds per shelf on CPU.
- **Leiden**: under 1 second per graph at this scale.
- **Labeling (keyword strategy)**: ~10 seconds per shelf total for c-TF-IDF.
- **Labeling (LLM polish via Haiku, optional)**: ~$0.20 per full run.
- **Total per run**: 2–5 minutes end-to-end on a single machine, without parallelism. Sub-1 minute parallelized.

Reproducibility: identical inputs + identical `random_state` produce identical themes.

---

## 13. Reusability for v2 facets

When MONDO (health) integrates, you'll run the same Layer B pipeline on the health facet with one config change:

```yaml
layer_b:
  facet_scope:
    health: enabled    # was deferred
  relatedness:
    always_exclude_iris:
      - "MONDO:0000001"  # disease root
      - "synthetic:Health"
```

Same algorithm, different ontology source, different blacklist of ubiquitous IDs. No new code needed if v1 is built generically.

---

## 14. Open decisions for the implementer

These are left to the engineer running Layer B; don't pre-decide:

- **Hierarchical Leiden** (`la.find_partition_multiplex` or repeated runs with sub-resolutions): the brief mentions it as an option. Skip for v1; revisit only if certain shelves produce 1-2 mega-themes that should be split into sub-themes.
- **Edge weighting in relatedness**: the IDF-style `1 / log(1 + doc_freq)` is a reasonable default. Could also use raw `1 / doc_freq` or BM25-style normalization. Tune in Phase 2 if themes look biased toward common entities.
- **Multi-shelf themes**: a chunk attached to two shelves might land in themes in both. That's fine; the brief explicitly supports themes attaching to multiple shelves (`HAS_THEME` from multiple shelves to the same Theme). But for v1, run themes per-shelf and don't try to deduplicate themes across shelves. v2 work.
- **Primary theme picker**: when a chunk lands in multiple themes within one shelf, which is primary? Default: highest weight (smallest distance to centroid). Document the rule; don't go beyond it for v1.

---

## 15. Success criteria — what "Layer B done" means

After Phase 4 with all audit gates green:

- ~15 themed shelves with 3–12 themes each → roughly 60–180 total themes in the foods facet.
- Pass distribution: roughly 30–60% `merged`, the rest split between `similarity` and `relatedness` (no pass at zero).
- Hand audit on 20 themes shows ≥ 75% coherent labels.
- Cross-store parity on `theme_ids` / `THEME_OF` is 1.0.
- A query like *"vitamin D in cow milk"* can be answered by filtering to a theme like `foods/cow_milk/calcium_bone_health_m1` instead of the whole shelf.

At that point, Layer A + Layer B are the navigation backbone. Layer C (cards) sits on top and turns themes into prose summaries with citations.

---

## Appendix: snippets

### Greedy pair assignment in merge

```python
def greedy_pair_assignment(decisions: list[MergeDecision]) -> set[tuple[int, int]]:
    """
    Sort decisions by combined_similarity descending.
    Assign each sim_cand and each rel_cand to at most one merge.
    """
    sorted_d = sorted(decisions, key=lambda d: -d.combined_similarity)
    used_sim, used_rel = set(), set()
    merged = set()
    for d in sorted_d:
        if not d.merged:
            continue
        if d.similarity_candidate_idx in used_sim:
            continue
        if d.relatedness_candidate_idx in used_rel:
            continue
        merged.add((d.similarity_candidate_idx, d.relatedness_candidate_idx))
        used_sim.add(d.similarity_candidate_idx)
        used_rel.add(d.relatedness_candidate_idx)
    return merged
```

### Theme ID slug generation

```python
import re

def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:48]

def theme_id(facet: str, shelf_id: ShelfId, label: str,
             discovery_pass: DiscoveryPass, seq: int) -> ThemeId:
    pass_initial = {"similarity": "s", "relatedness": "r", "merged": "m"}[discovery_pass]
    shelf_slug = shelf_id.split("/")[-1]  # last segment
    return f"{facet}/{shelf_slug}/{slugify(label)}_{pass_initial}{seq}"
```

### Audit query for cross-store parity

```python
def audit_theme_parity(graph_store, chunk_store) -> float:
    # For every Theme in Neo4j, get the set of chunks via THEME_OF
    # For every chunk doc in Elastic, get its theme_ids
    # Build two membership matrices and compute exact agreement rate
    ...
```

---

End of brief. Hand to Claude Code with this repo + the progress log. Estimated effort: ~10 working days. Cost per run: well under $0.50 with keyword labeling; ~$0.70 with LLM polish.
