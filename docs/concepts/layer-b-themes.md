# Layer B — Themes

Layer A gives you coarse shelves. Layer B finds the **fine-grained topics inside a
shelf** — *themes*.

## The two methods

Layer B offers **two discovery methods**, selected by `config.layer_b.algorithm`. They are
**not two flavors of one pipeline** — they run different pipelines and produce different theme
mixes. Pick one; they never co-run.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Method A — `"leiden"` (default)
**Two passes + merge.** Pass 1 finds embedding-coherent communities (similarity graph +
Leiden); Pass 2 finds entity-coherent communities (FoodOn-id bridge graph + Leiden); a merge
step fuses the two where they agree.

→ themes labelled `merged`, `global_similarity`, `relatedness`.
:::

:::{grid-item-card} Method B — `"bertopic"`
**Single pass.** Clusters the shelf's chunk embeddings *directly* with BERTopic — no graph, no
Pass 2, no merge. Never touches a Leiden code path.

→ themes labelled `global_similarity` only.
:::

::::

| | **Leiden** (`"leiden"`) | **BERTopic** (`"bertopic"`) |
|---|---|---|
| Passes | **two** (similarity + relatedness) | **one** (embedding clustering) |
| Merge step | **yes** | **no** |
| Signals used | embedding **and** entity | embedding only |
| Theme buckets | `merged` / `global_similarity` / `relatedness` | `global_similarity` only |
| `discovered_by` | `"leiden"` | `"bertopic"` |
| `pass1_mode` | honored (`per_shelf` default, `global` opt-in) | **ignored** (always per-shelf) |
| Reach for it when… | you want entity+embedding evidence fused; robust, audited themes | you want fast, exhaustive embedding topics; the tuned production baseline |

**Shared by both methods.** Both run **per shelf**; both honor [`scope`](#scope) (which chunks
a shelf contributes); both feed the same **labeling**, the same `Theme` record, and the same
persistence. Only the discovery step differs.

---

## Method A — Leiden (two-pass + merge)

A shelf's chunks can be related in two different ways, so Leiden runs two community detections
and keeps the best of both.

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

### Both passes are per-shelf

```{important}
In the production configuration (`config.layer_b.pass1_mode = "per_shelf"`) **both
passes run per shelf** — each shelf's chunk set gets its own kNN/entity graph and its
own Leiden run. A theme therefore has a single, well-defined **origin shelf**: the
shelf whose chunks it was built from.
```

Pass 1 also has a `"global"` mode (`pass1_mode = "global"`) that runs one Leiden over the
*entire* facet to find cross-shelf "bridge" themes. It is **opt-in** — kept available but off
by default, because global communities span many shelves and, because a chunk sits on many
shelves ({term}`lifted attachment` — *not* the same as a shelf's lifted support), they smear a
theme across every shelf its members touch — a `spice` shelf ends up showing themes about
proteins and vegetables. Per-shelf Pass 1 avoids this by construction. (`pass1_mode` is a
**Leiden-only** axis — BERTopic ignores it; see [Method B](#method-b-bertopic-single-pass).)

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

### The merge step

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

### Coverage is a deliberate trade-off

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

---

(method-b-bertopic-single-pass)=
## Method B — BERTopic (single-pass)

`config.layer_b.algorithm = "bertopic"` clusters each shelf's chunk **embeddings directly**
with [BERTopic](https://maartengr.github.io/BERTopic/) instead of building a similarity graph
and running Leiden.

```{important}
**BERTopic is single-pass.** It runs Pass 1 only — **Pass 2 (relatedness) and the merge step
do NOT run**. A BERTopic build produces `global_similarity` themes *only* (no `merged`, no
`relatedness`), and it never invokes a Leiden code path, so the two methods never co-run.

Why: BERTopic partitions a shelf on an embedding axis that is *orthogonal* to FoodOn
entities, so a BERTopic topic and an entity-relatedness community almost never overlap — the
merge produced **zero** merged themes and just concatenated two disjoint sets. Running Pass 2
there bought noise and double the compute for no synthesis, so bertopic mode skips both.
```

```{mermaid}
flowchart LR
    C[shelf chunks + embeddings] --> B[BERTopic over raw vectors] --> BTC[topic candidates]
    BTC --> TH["themes: global_similarity only<br/>(no Pass 2, no merge, no Leiden)"]
```

Each emitted community becomes a `ThemeCandidate` with `discovered_by="bertopic"`; each
candidate becomes a theme directly (no merge), then flows through the identical label/persist
path. BERTopic is inherently per-shelf and **ignores `pass1_mode`** — setting
`pass1_mode="global"` with `algorithm="bertopic"` runs BERTopic per-shelf anyway (and logs a
notice); it never falls back to a Leiden global run.

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

---

## Shared by both methods

The discovery step is the only thing that differs between the methods. Everything below applies
to both.

(scope)=
### Scope — `layer_b.scope`

Both methods run **per shelf**, and `config.layer_b.scope` chooses *which* chunks each shelf
contributes to Pass 1 — it applies to **both Leiden and BERTopic**:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} `"direct"` (default)
Only the shelf's **own** directly-attached chunks. Themes are local to the node; a chunk is
clustered once, under the shelf it sits on.
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

```{note}
`layer_b.scope` is the single source of truth and governs both methods. `bertopic.scope` is a
**deprecated back-compat alias**: it only takes effect for the BERTopic path when set to a
non-default value, where it overrides `layer_b.scope` for BERTopic only. Prefer `layer_b.scope`.
```

### Labeling

Each theme gets a short label. The default keyword labeler computes
[c-TF-IDF](glossary.md) over the theme's chunks and takes the top discriminative terms (filtering OCR codes, id-like
tokens, and sub-3-char fragments). With `labeling.strategy = "llm"`, those terms plus a
few sample chunks are handed to the LLM for a clean 3–5 word label.

### The Theme record

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

### Building it

```python
# Method A — Leiden, per-shelf (the default).
fs.build_layer_b(facet="foods")

# Method B — BERTopic (direct scope, HDBSCAN) — the tuned baseline:
fs.config.layer_b.algorithm = "bertopic"
fs.config.layer_b.scope = "direct"                 # or "subtree" — shared knob, both methods
fs.config.layer_b.bertopic.clusterer = "hdbscan"   # or "kmeans"
fs.build_layer_b(facet="foods")
```

Then click a shelf in the interactive tree to see its themes grouped by origin:

```python
fs.viz.layer_a_tree("foods").render("tree", output="tree.html")
```
