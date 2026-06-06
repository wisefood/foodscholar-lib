# Layer B — Themes

Layer A gives you coarse shelves. Layer B finds the **fine-grained topics inside a
shelf** — *themes* — using two complementary signals and merging them.

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
    discovered_by: Literal["leiden", "hdbscan"]
```

## Building it

```python
fs.build_layer_b(facet="foods")   # pass1_mode defaults to "per_shelf"
```

Then click a shelf in the interactive tree to see its themes grouped by origin:

```python
fs.viz.layer_a_tree("foods").render("tree", output="tree.html")
```
