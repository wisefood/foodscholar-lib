# Layer A & Layer B — methods brief

A walk-through of what each pruning rule and each clustering pass actually
does, written for someone auditing the notebook output and tuning the knobs.
This is the conceptual companion to [`layer_b_construction_brief.md`](../layer_b_construction_brief.md)
(implementation contract) and to the §5/§6/§7 cells in
[`notebooks/build_graph.ipynb`](../notebooks/build_graph.ipynb).

---

## Layer A — backbone construction

Layer A projects FoodOn (and any other linked OBO ontologies) onto **this**
corpus. The output is a small, navigable set of "shelves" — typically
100–300 nodes, one per facet — that every chunk attaches to.

### Lifting (before pruning)

For each FoodOn class, count two things:

- `support_direct` — chunks with a high-confidence entity link to *exactly*
  this class
- `support_lifted` — chunks linking to this class **or** any of its
  descendants

Lifting populates the entire ontology tree with corpus-grounded counts. The
pruning rules then walk that tree.

### The three pruning rules

The pruning cascade is the heart of Layer A. Each rule answers a different
failure mode of the raw ontology.

#### Rule 1 — `min_support`: drop low-evidence shelves

> A class survives only if `support_lifted ≥ min_support` (default 25).

FoodOn has ~30k classes; most have single-digit support on any real corpus.
Rule 1 sweeps the long tail in one pass.

**Failure mode when too aggressive.** Rare-but-real foods (kefir, sumac,
amaranth) get orphaned up to the synthetic root. §6g/§6h in the notebook
diagnose this and recommend a mid-level whitelist as the fix.

#### Rule 2 — the umbrella rule: drop **inflated** shelves

> A class is dropped iff **both**:
> - `direct_share = support_direct / support_lifted < umbrella_direct_share_max` (default 0.10), AND
> - `support_lifted ≥ umbrella_min_count` (default 100).

"Inflated" means a class with huge `support_lifted` but tiny `support_direct`
— almost no chunk links to it directly; it just accumulates lifted count
from every descendant. The signature umbrella nodes are `food product`,
`food material`, `material entity`. They survive Rule 1 trivially (they're
*large*) and they're shallow (so Rule 3 doesn't help). Rule 2 is the only
rule that catches them.

The AND between the two conditions matters:

- `direct_share` alone would kill specific-but-rarely-direct-linked terms
  (e.g. niche varieties that the linker prefers to resolve to a parent).
- `support_lifted` size alone would kill big-and-legitimate shelves.

Only the intersection — *big because of descendants* AND *almost never used
directly* — is the umbrella signature.

**Diagnostic.** §6f in the notebook prints which of the two thresholds an
inflated shelf fails. If `direct_share = 0.13` for `food product`, raise
`umbrella_direct_share_max` to 0.15. If `support_lifted = 80`, lower
`umbrella_min_count` to 50.

#### Rule 3 — `max_depth`: cap and **lift** deep classes

> Classes at depth > `max_depth` (default 6) are dropped; their chunk
> attachments re-home onto the nearest surviving ancestor at depth ≤ cap
> via `see_also`.

FoodOn goes 15+ levels deep for some food classifications. A 12-deep shelf
isn't a destination, it's a leaf. Rule 3 keeps the tree shallow enough to
render and browse.

**Non-obvious part.** Raising `max_depth` doesn't add shelves — every deep
class was already surviving Rules 1 and 2. What it changes is *which*
shelves carry which chunks: a deep class either appears as its own shelf or
gets folded up into an ancestor. §5c plots this trade-off directly.

### Collapse single-child chains (cleanup)

After the three rules, long chains `A → B → C` where each parent has only
one surviving child collapse into the deepest survivor, with the collapsed
ancestors recorded in `see_also`. This turns FoodOn's organizational
scaffolding into a browsable tree.

### Operation order matters

§7.0 of `PROGRESS.md` notes that lift → prune → cap → collapse is the
deliberate order. Reordering them silently changes the shelf set — for
example, capping before lifting would lose the deep terms' chunk
attachments instead of re-homing them.

### Two senses of "inflated"

The word is used for two distinct phenomena. They co-exist on the same
corpus and want different fixes:

| Term                | What it is                                            | Caused by                               | Fixed by                          |
|---------------------|-------------------------------------------------------|-----------------------------------------|-----------------------------------|
| **Inflated shelf**  | high `support_lifted`, low `support_direct`           | umbrella surviving the prune cascade    | Rule 2 (tighten thresholds)       |
| **Inflated facet**  | synthetic root (`facet:foods`) catches too many chunks | Rules 1+2 over-prune; orphans fall through | §6h mid-level whitelist          |

The first is "a bad shelf lived"; the second is "good shelves died and the
chunks had nowhere to go."

### Facet routing — populating the non-food facets

A shelf belongs to a **facet**; which facet an entity link lands in is decided by
`route_link_to_facet`, in precedence order:

1. **NER `entity_type`** (`ENTITY_TYPE_TO_FACET`): `nutrient→nutrients`,
   `medical condition→health`, `allergen→allergies`, … — the intended signal.
2. **FOODON id → `foods`**: any `FOODON:` link is a food regardless of entity_type.
3. **OBO prefix** (`PREFIX_TO_FACET`): `CHEBI`/`CDNO`/`ONS`→nutrients,
   `UBERON`/`MONDO`/`PATO`→health, `ENVO`/`GAZ`→sustainability.

Step 3 is the load-bearing one on the **prototype corpus**, where the NER tagged *every*
mention `entity_type='other'` — so step 1 never fires and steps 2/3 carry the whole load.
Without step 3, the 16k non-FOODON OBO links (CHEBI, UBERON, ENVO, …) route nowhere and
the non-food facets collapse to a single stub root. With it (and an
`ontology.prefix_filter` that admits those OBO prefixes, which live inside `foodon.owl`),
they project onto real OBO hierarchies — measured on the reference corpus:

| Facet | prefix routing off | prefix routing on |
|---|---|---|
| nutrients | 1 (stub) | ~120 shelves |
| health | 1 (stub) | ~64 shelves |
| sustainability | 1 (stub) | ~6 shelves |
| foods | 246 | 246 (unchanged) |

The cost: the prototype linker mis-assigns across OBOs (`"heart disease"→UBERON`,
`"breakfast"→NCIT`), so prefix routing inherits some noise. The clean long-term fix is
re-annotation with a NER that populates `entity_type` correctly; prefix routing is the
pragmatic way to use the OBO links already present. `allergies`/`dietary_patterns` stay
at a stub root because the corpus links no allergen / dietary-pattern entities.

### How Layer A feeds Layer B

After pruning, `fs.attach()` walks every chunk and resolves each FoodOn id
via direct → collapsed (`see_also`) → lifted (deepest surviving ancestor).
A chunk lands on one or more shelves. The §7 builder reads those
attachments and clusters within them.

---

## Layer B — theme discovery

For each facet, find groups of chunks that "belong together" inside the
shelves Layer A produced. The shelves are the navigation skeleton; the
themes are the finer-grained topics a user actually wants to read about
("olive oil polyphenols and heart disease", not just `olive oil`).

Two passes run in parallel because "belong together" has two meanings, and
a single signal misses half the themes a careful reader would find.

### Setup before either pass

Per attached chunk, we already have:

- the **BGE-base embedding** (768-d, L2-normalized, cached in Elastic)
- the **FoodOn entity links** (`foodon_id`, confidence ≥ `tau_strict`)
- the **shelf attachments** (one or more shelves from Layer A)

### Pass 1 — Global similarity (cross-shelf, embedding-based)

> *Which chunks read like each other?*

**Graph construction.**

- Nodes: every attached chunk in the facet (all shelves pooled — *global*).
- For each chunk, find its `knn_k` nearest neighbors by cosine over
  embeddings.
- Keep an edge iff (a) cosine ≥ `edge_threshold` AND (b) the relationship is
  **mutual** — chunk A is in B's top-k and B is in A's top-k. Mutual-kNN
  suppresses "hub" chunks that link to everything.

**Leiden** finds communities (modularity-maximizing). The `resolution` knob
controls split aggressiveness.

**Why this finds cross-shelf themes.** Two chunks on different shelves (one
on `olive oil`, one on `cardiovascular health`) can still be each other's
nearest neighbors in embedding space — because both discuss olive oil's
effect on heart disease. Leiden doesn't know about shelves. After
clustering, each theme's `shelf_ids` is the **union** of shelves its chunks
were attached to. That's how a theme ends up with `shelf_ids` of length ≥ 2.

**Failure mode.** Too-low `edge_threshold` or too-high `knn_k` → a graph
where everything connects to everything → one giant community = megacluster.
§7c sweeps these knobs to find the legible region.

### Pass 2 — Relatedness (per-shelf, entity co-occurrence)

> *Which chunks talk about the same things, in the FoodOn sense?*

**Why we need it.** Two chunks can both discuss "fermented dairy products
and gut microbiota" in very different prose styles — different decade,
different journal, different authors. Their embeddings might not be close
enough for Pass 1's mutual-kNN. But if both high-confidence-link to
`FOODON:fermented_dairy_product` and `FOODON:gut_microbiota`, Pass 2 spots
the shared entity vocabulary directly.

**Graph construction (per-shelf, not global).**

- Nodes: chunks attached to *this one shelf*. (Scoped per-shelf because the
  pairwise entity-overlap cost blows up at corpus scale, and within-shelf
  themes are most of Pass 2's catch anyway.)
- For each pair of chunks: edge weight =
  `Σ over shared FoodOn ids of (1 / log(1 + doc_frequency_of_that_id))`.
  IDF trick — `food` (in 6,000 chunks) contributes near zero; `kefir` (in
  12 chunks) contributes meaningfully.
- Drop entities above `max_doc_frequency` (too generic to discriminate).
- Drop entity links below `tau_strict` confidence (linker noise).
- Drop edges below the relatedness threshold.

**Leiden** on this graph → per-shelf communities. Each is a candidate theme
scoped to one shelf.

**What it catches that Pass 1 misses.** Shared rare-entity vocabulary across
stylistically-divergent prose. A 1998 paper and a 2023 paper that both
focus on `kefir + lactic acid bacteria + gut barrier` land in the same
Pass-2 community even if their embeddings sit ~0.4 cosine apart.

### The merge step

> **Leiden mode only.** Pass 2 and the merge run when `algorithm="leiden"`.
> In `bertopic` mode Pass 1 (BERTopic) runs alone — its embedding-clustered
> topics are orthogonal to FoodOn entities and never merged with relatedness
> communities, so Pass 2 + merge are skipped there. See
> `methods_layer_b_c_brief.md` §1.1.

After both passes finish you have two disjoint lists:

- **G** = Pass-1 candidates (can span multiple shelves)
- **R** = Pass-2 candidates (each scoped to one shelf)

A `ThemeCandidate` carries:

- `chunk_ids: set[str]` — chunks in this candidate
- `foodon_ids: set[str]` — union of high-confidence FoodOn ids across those
  chunks

For every pair `(g, r)` with `g ∈ G, r ∈ R`:

1. **Chunk Jaccard** — `J_chunk = |g.chunks ∩ r.chunks| / |g.chunks ∪ r.chunks|`.
   The strongest signal: same documents.
2. **Entity Jaccard** — `J_entity = |g.foodon ∩ r.foodon| / |g.foodon ∪ r.foodon|`.
   Same FoodOn vocabulary even if the documents differ.
3. **Combined similarity** — weighted average, default `0.6·J_chunk + 0.4·J_entity`.
   Chunk overlap weighs more because it's harder evidence.
4. **Decision** — `combined ≥ dedupe_threshold` (default 0.70) → record
   `MergeDecision(merged=True)`.

A single Pass-1 candidate can match multiple Pass-2 candidates from
different shelves. The merge is greedy with a union-find over candidate
indices, so transitively-overlapping merges (g1↔r1, r1↔g2) collapse into
one theme.

**Three buckets emerge:**

| `discovery_pass`     | Source                                        | Typical shape                                    |
|----------------------|-----------------------------------------------|--------------------------------------------------|
| `merged`             | Pass-1 and Pass-2 candidate found each other  | Strongest themes — both signals agreed           |
| `global_similarity`  | Pass-1 candidate with no Pass-2 counterpart   | Often cross-shelf bridges (`len(shelf_ids) ≥ 2`) |
| `relatedness`        | Pass-2 candidate with no Pass-1 counterpart   | Always single-shelf; shared-vocab cases          |

### Labeling and persistence

Each surviving theme is labeled by:

1. **c-TF-IDF** over its chunks → top keyword terms.
2. Optional **LLM polish** — Groq `llama-3.3-70b-versatile` by default;
   takes the c-TF-IDF terms and a sample of chunk excerpts, returns a short
   label. Filters drop OCR/ID-string garbage from the term list before the
   prompt builds (see commit `a622642`).

Persisted to Neo4j as `(:Theme)` nodes with `shelf_ids: list[ShelfId]`
(possibly > 1) and denormalized back to Elastic via each chunk's
`theme_ids`. The bucket stamps as `Theme.discovery_pass`.

---

## Auditing a Layer B build (notebook §7d / §7e)

### §7d — marginals

Three histograms:

- **theme size** — peaked at small sizes is good; a long right tail is the
  megacluster signature.
- **shelves per theme** — many `1`s with a tail at `2+` is healthy. All
  `1`s means Pass 1 isn't bridging anything (`knn_k` too low or
  `edge_threshold` too high).
- **theme count by `discovery_pass`** — sanity check; expect all three
  buckets to be non-empty.

### §7e — theme × shelf distribution heatmap

A 2D view that the §7d marginals can't give you. Rows = top-N themes by
chunk_count, columns = shelves they touch, cell = share of that theme's
chunks attached to that shelf (row-normalized).

How to read it:

| Row signature                          | Meaning                                            |
|----------------------------------------|----------------------------------------------------|
| One bright cell                        | Single-shelf theme (`relatedness` territory)       |
| Two or three bright cells              | Genuine cross-shelf bridge (`global_similarity`)   |
| Long smear of dim cells                | Diffuse / megacluster — re-tune §7b/§7c            |

The left side-bar shows `log10(theme_size)` so you can correlate "big row"
with "diffuse row" — that's the megacluster signature.

Columns are reordered by hierarchical clustering on shelf co-occurrence
(average linkage on Jaccard distance) so co-occurring shelves sit adjacent
and bridge cells line up visually. Falls back to mass-ordering if SciPy is
unavailable.

### §7-final — cross-store audit

`audit_layer_b` enforces the critical invariants (chunk↔theme parity = 1.0,
no dangling `theme_ids`, no empty themes). Failure here means
`fs.build_layer_b()` produced inconsistent state across Neo4j and Elastic —
not a tuning issue, a correctness issue.

The per-pass metrics (`merged_rate`, theme counts by pass) are WARN-level
tuning canaries; they don't flip `passed` but they tell you which knob to
turn.
