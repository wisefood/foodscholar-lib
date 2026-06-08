# Layer B — Themes

Layer A gives you coarse shelves. Layer B finds the **fine-grained topics inside a
shelf** — *themes* — using two complementary signals and merging them.

```{admonition} Two discovery backends
:class: tip
Pass 1 (the embedding signal) has **two interchangeable backends**, selected by
`config.layer_b.algorithm`:

- **`"leiden"`** (default) — build a similarity *graph* over chunk embeddings, run Leiden.
- **`"bertopic"`** — cluster the chunk embeddings *directly* with BERTopic (no graph).

Pass 2 (the entity signal) and the merge step are **identical** for both. Everything below
describes the shared two-pass/merge architecture first; the
[BERTopic backend](#pass-1-backend-bertopic) section then explains the alternative and when
to reach for it.
```

## Two passes, two signals

A shelf's chunks can be related in two different ways, so Layer B runs two community
detections and keeps the best of both.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Pass 1 — Similarity
Build a mutual-kNN graph over chunk **embeddings** (cosine), run Leiden.

Catches chunks that *read alike*. Misses the same idea phrased differently.
:::

:::{grid-item-card} Pass 2 — Relatedness
Build an **entity-bridge** graph: an edge's weight is `Σ 1/log(1+df(id))` over the
FoodOn IDs two chunks share (rare entities count more), run Leiden.

Catches chunks that *cite the same foods*, even in different prose. Misses the same
idea expressed via different entities.
:::

::::

These are orthogonal: two chunks can share entities but read differently (low cosine),
or read alike but share no specific entity. Running both and merging is what makes the
themes robust.

## Both passes are per-shelf

```{important}
In the production configuration (`config.layer_b.pass1_mode = "per_shelf"`) **both
passes run per shelf** — each shelf's chunk set gets its own kNN/entity graph and its
own Leiden run. A theme therefore has a single, well-defined **origin shelf**: the
shelf whose chunks it was built from.
```

Pass 1 also has a `"global"` mode that runs one Leiden over the *entire* facet to find
cross-shelf "bridge" themes. It was tried and reverted: global communities span many
shelves and, because a chunk sits on many shelves ({term}`lifted attachment` — *not* the
same as a shelf's lifted support), they smear a theme across every shelf its members
touch — a `spice` shelf ends up showing themes about proteins and vegetables. Per-shelf
Pass 1 avoids this by construction.

```{warning}
**Naming landmine.** The `discovery_pass` value `global_similarity` is a historical name.
In the production per-shelf mode it means *per-shelf embedding similarity* — there is
nothing global about it. The value is retained only for schema continuity.
```

### Origin-shelf attachment

A theme is attached to its **origin shelf**, not to the union of its member chunks'
shelves. A `ThemeCandidate` (a community a pass emits *before* merging) carries an
`origin_shelf_id`; an unmerged similarity theme attaches to `[origin_shelf]`. The
chunk-union fallback is used *only* for true global Pass 1, where a community legitimately
spans shelves and has no single origin.

## The merge step

For each shelf, the similarity candidates are matched against the relatedness
candidates by a greedy, highest-first pairing on a combined Jaccard. `jaccard(chunks)` is
the overlap of the two candidates' **member-chunk sets**; `jaccard(entities)` is the
overlap of their **FoodOn-id sets**:

```text
combined = chunk_weight · jaccard(chunk sets) + entity_weight · jaccard(entity sets)
pair if combined ≥ dedupe_threshold   (each candidate used at most once)
```

```{mermaid}
flowchart LR
    C[shelf chunks] --> P1[Pass 1: kNN + Leiden]
    C --> P2[Pass 2: entity bridges + Leiden]
    P1 --> S[similarity candidates]
    P2 --> R[relatedness candidates]
    S --> M{greedy merge<br/>combined ≥ threshold?}
    R --> M
    M -->|both| MG[merged]
    M -->|sim only| GS[global_similarity]
    M -->|rel only| RL[relatedness]
```

Three buckets of themes come out, recorded in `Theme.discovery_pass`:

| `discovery_pass` | Source | Reading |
|---|---|---|
| `merged` | both passes agreed | strongest signal — embedding *and* entity evidence |
| `global_similarity` | Pass 1 only | embedding-coherent, no entity counterpart |
| `relatedness` | Pass 2 only | entity-coherent, prose stylistically distant |

### A real shelf's themes

The `mammalian milk product` shelf from a `foods` build, by origin:

| `discovery_pass` | theme | chunks | top keywords |
|---|---|---|---|
| `relatedness` | milk calcium lactose | 167 | milk, calcium, lactose, vitamin, fat |
| `relatedness` | milk protein foods | 103 | milk, protein, foods, eggs |
| `global_similarity` | calcium vitamin milk | 123 | calcium, vitamin, milk, protein |
| `global_similarity` | breast milk breast infant | 98 | breast milk, breast, infant, formula |

This build produced **zero `merged`** themes — a real, instructive outcome. The per-pass
distribution is an audit signal: all-`relatedness`/zero-`merged` means the passes aren't
overlapping enough to cross the merge threshold, so the thresholds (or Pass-1 coverage)
need [tuning](../guides/tuning-layer-b.md) — configuration, not architecture. (Labels look
keyword-ish here because this build used `labeling.strategy = "keyword"`; the LLM strategy
polishes them.)

(pass-1-backend-bertopic)=
## Pass 1 backend — BERTopic

`config.layer_b.algorithm = "bertopic"` swaps **only Pass 1**: instead of building a
similarity graph and running Leiden, it clusters the shelf's chunk **embeddings directly**
with [BERTopic](https://maartengr.github.io/BERTopic/). Pass 2 (relatedness) still runs
Leiden, and the merge step is unchanged — so a BERTopic build still produces `merged` /
`global_similarity` / `relatedness` themes exactly as above. The only differences are *how*
the embedding communities are found and which `discovered_by` value they carry.

```{mermaid}
flowchart LR
    C[shelf chunks + embeddings] --> A{algorithm?}
    A -->|leiden| G[mutual-kNN graph] --> L[Leiden] --> CAND[similarity candidates]
    A -->|bertopic| B[BERTopic over raw vectors] --> CAND
    CAND --> MERGE[merge with Pass-2 relatedness]
```

Each emitted community becomes a `ThemeCandidate` with `discovered_by="bertopic"` (vs
`"leiden"`); from there it flows through the identical merge/label/persist path.

### Two scope choices — `bertopic.scope`

BERTopic runs **per shelf**, but you choose *which* chunks a shelf contributes:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} `"direct"` (default)
Only the shelf's **own** directly-attached chunks. Themes are local to the node; a chunk is
clustered once, under the shelf it sits on. Mirrors the Leiden per-shelf scope.
:::

:::{grid-item-card} `"subtree"`
The shelf's chunks **plus every descendant shelf's** chunks. Inner nodes get broader,
roll-up themes spanning their branch; the same chunk participates in every ancestor's run.
:::

::::

```{mermaid}
flowchart TD
    subgraph direct["scope = direct"]
        d_dairy["dairy<br/>(own chunks only)"]
        d_milk["milk<br/>(own chunks only)"]
        d_cheese["cheese<br/>(own chunks only)"]
        d_dairy -.->|tree edge, NOT clustered together| d_milk
        d_dairy -.-> d_cheese
    end
    subgraph subtree["scope = subtree"]
        s_dairy["dairy<br/>(own + milk + cheese chunks)"]
        s_milk["milk<br/>(own chunks)"]
        s_cheese["cheese<br/>(own chunks)"]
        s_dairy ==>|inherits descendant chunks| s_milk
        s_dairy ==> s_cheese
    end
```

### Two clusterers — `bertopic.clusterer`

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} `"hdbscan"` (default)
BERTopic the way it ships: **UMAP → HDBSCAN**. Discovers the topic count from density and
emits a `-1` *outlier* bucket (dropped — those chunks stay un-themed). No `K` to choose;
naturally exhaustive. Best when you don't want to fix the number of themes.
:::

:::{grid-item-card} `"kmeans"`
A **passthrough** reducer + **KMeans** on the raw BGE vectors. Full coverage (no outlier
bucket), predictable count — `n_clusters`, or auto `clamp(round(√(n/2)), 2, 12)` when
`n_clusters` is `None`. Best when you want a controlled number of themes per shelf.
:::

::::

```{mermaid}
flowchart LR
    V[chunk vectors] --> CL{clusterer?}
    CL -->|hdbscan| U[UMAP 15→5d] --> H["HDBSCAN<br/>(min_cluster_size)"] --> HT["topics + −1 outliers"]
    CL -->|kmeans| PT["passthrough<br/>(raw vectors)"] --> K["KMeans(n_clusters)"] --> KT["topics<br/>(full coverage)"]
    HT --> F["drop −1 + size filter<br/>(min_topic_size)"]
    KT --> F
    F --> GRP[chunk-id groups → candidates]
```

Both clusterers post-filter with `bertopic.min_topic_size` (the BERTopic analogue of
Leiden's `min_community_size`): topics smaller than this are dropped.

```{list-table} BERTopic knobs (`config.layer_b.bertopic`)
:header-rows: 1

* - Knob
  - Default
  - Meaning
* - `scope`
  - `"direct"`
  - `direct` = own chunks · `subtree` = own + descendants
* - `clusterer`
  - `"hdbscan"`
  - `hdbscan` = auto count + outliers · `kmeans` = matched/auto-K, full coverage
* - `min_topic_size`
  - `15`
  - Minimum chunks per theme (drops smaller topics)
* - `n_clusters`
  - `None`
  - KMeans only; `None` → auto `√(n/2)` clamped to `[2, 12]`
* - `random_state`
  - `42`
  - Determinism seed for UMAP / KMeans
```

```{note}
BERTopic and its deps (`bertopic`, `umap-learn`, `hdbscan`, `scikit-learn`) are lazy-imported
behind the `[bertopic]` / `[clustering]` extras — the core package imports without them. Install
with `pip install 'foodscholar[bertopic,clustering]'`.
```

```{seealso}
The Layer B & C methods brief (`docs/methods_layer_b_c_brief.md`) has the `run_bertopic`
internals, the Leiden-vs-BERTopic contract, and the full tuning matrix.
```

## Labeling

Each theme gets a short label. The default keyword labeler computes
[c-TF-IDF](glossary.md) over the theme's chunks and takes the top discriminative terms (filtering OCR codes, id-like
tokens, and sub-3-char fragments). With `labeling.strategy = "llm"`, those terms plus a
few sample chunks are handed to the LLM for a clean 3–5 word label.

## Coverage is a deliberate trade-off

Per-shelf Pass 1 themes *fewer* chunks than global Pass 1 — only chunks that land in a
Leiden community of at least `leiden.min_community_size` (default 15) within their own
shelf's graph get a theme. Un-themed chunks are **not lost**: they remain attached to
their shelf and fully searchable; they just aren't bucketed into a sub-topic. To theme
more chunks, loosen the per-shelf graph:

| Knob | Default | Effect of loosening |
|---|---|---|
| `leiden.min_community_size` | 15 | smaller communities survive (biggest lever) |
| `similarity.edge_threshold` | 0.55 | looser kNN edges → denser graph |
| `similarity.require_mutual` | true | one-directional neighbours kept too |

The trade-off is smaller, noisier themes; the **Guides** section walks through a tuning
sweep.

## The Theme record

```python
class Theme(BaseModel):
    theme_id: ThemeId
    label: str
    shelf_ids: list[ShelfId]      # origin shelf (per-shelf) or union (global)
    chunk_count: int
    facet: Facet
    discovery_pass: Literal["relatedness", "merged", "global_similarity"]
    keyword_terms: list[str]
    foodon_id_signature: list[str]
    discovered_by: Literal["leiden", "hdbscan", "bertopic"]
```

## Building it

```python
# Default: Leiden, per-shelf.
fs.build_layer_b(facet="foods")

# BERTopic backend (direct scope, HDBSCAN) — the tuned baseline:
fs.config.layer_b.algorithm = "bertopic"
fs.config.layer_b.bertopic.scope = "direct"        # or "subtree"
fs.config.layer_b.bertopic.clusterer = "hdbscan"   # or "kmeans"
fs.build_layer_b(facet="foods")
```

Then click a shelf in the interactive tree to see its themes grouped by origin:

```python
fs.viz.layer_a_tree("foods").render("tree", output="tree.html")
```
