# Layer-A category re-tiering — prototype design

A notebook prototype that turns the flat, un-navigable foods facet into a
browsable category tree, by **re-cutting FoodOn's own hierarchy** — using only
real FoodOn ids and is-a edges.

## Problem

The live foods facet builds **~227 shelves**, most piled at depth 1 (the
"~186 categories at depth 1" a user is dumped into). Verified root cause:

- Support pruning (`min_support=25`, the umbrella rule) deletes FoodOn's
  **mid-level grouping nodes** (they have little/no *direct* support).
- `_nearest_included_ancestor` ([prune.py:200](../../../src/foodscholar/layer_a/prune.py#L200))
  then re-parents every survivor onto its *deepest surviving* ancestor — with
  the middle dissolved, that's a shallow node → enormous top-level fan-out.

The existing `semantic_consolidate` pass merges *duplicate* shelves
(yoghurt = yogurt); it only ever **reduces** shelf count and never introduces
grouping tiers, so it cannot fix fan-out.

## Hard constraint

Stay 100% within FoodOn. Only real FoodOn term ids and real is-a edges. No
invented categories, no relabeling into non-FoodOn terms. The LLM may only
**select among FoodOn ids we hand it** — it cannot fabricate structure.

## Goal (as a target shape)

Balance depth vs breadth: a few tiers, roughly even fan-out per level, so the
top level is a handful of intuitive FoodOn categories and the ~186 leaves are
reachable in 1–2 clicks (e.g. `vegetable → green / root / steamed vegetable →
…`).

## Three coupled parts

### Part A — Reconcile pruning (less aggressive)

Sweep `min_support` / umbrella thresholds / `max_depth` over the foods facet
and measure, per setting:

- shelf count (existing)
- **fan-out per level** (new — max and distribution of children-per-node)
- **count of FoodOn mid-level grouping nodes that survive** (new — the nodes
  that can serve as tiers)

Pick the setting that *keeps* the intermediate ancestors instead of dissolving
them. This is the "pruning too aggressive" fix and supplies the candidate tier
nodes for Part B.

### Part B — Re-tier (LLM picks FoodOn intermediates)

For each over-wide level:

1. **Enumerate candidates** — the real FoodOn intermediate ancestors sitting
   between the wide parent and its many children, from `api.id_to_ancestors`
   (closed transitive set). These are existing FoodOn ids only.
2. **LLM selects + assigns** — prompt **Groq `llama-3.1-8b-instant`** via
   `fs.llm.generate_json(prompt, SCHEMA, ...)`, mirroring
   [judge.py](../../../src/foodscholar/layer_a/semantic_consolidation/judge.py):
   render candidates as numbered blocks (label + FoodOn synonyms), the model
   answers **by index** picking which candidate ancestors are the intuitive
   tiers and which child goes under which. Index-based responses mean the model
   physically cannot return a non-FoodOn id; defensive parsing (out-of-range /
   unmentioned children) follows `_parse_cluster`.
3. **Emit ops** — an auditable list of
   `(insert FoodOn-id <X> as grouping shelf; reparent children [<c1>,<c2>,…]
   under <X>)`, in the spirit of `ConsolidationArtifact`. Applying them rewrites
   `parent_shelf_id`; inserted tier shelves carry `chunk_count` lifted from
   their assigned children (no direct support required — that's the point).

Model choice rationale: 8b-instant is sufficient because the task is
constrained selection over a provided candidate list, not open generation. (Per
project memory, avoid Groq *reasoning* models — they return empty content.)

### Part C — Before/after visualization

Reuse the `explore_foodon` nested-`<details>` tree renderer to show the foods
facet **today** (flat, ~186-wide) vs **after re-tiering** (`vegetable →
green/root/steamed…`), with fan-out-per-level stats beside each. This is the
demo artifact and the input for deciding where re-tiering eventually lands in
`src/`.

## Reconciliation with existing code

Re-tiering is the structural **inverse** of `semantic_consolidation`: it
*inserts grouping parents* rather than *merging duplicate siblings*. The
eventual production order would be:

```
prune (relaxed, Part A) → re-tier (Part B) → semantic_consolidate (existing)
```

The prototype **reuses** the consolidation module's conventions — Groq
`generate_json`, numbered-block prompts, index→id mapping, defensive parsing,
audit artifact, prompt versioning — but **not** its merge logic.

## Where it lives & how it reads data

- **New notebook** `notebooks/retier_layer_a.ipynb` (keeps `explore_foodon`
  purely diagnostic).
- **Not corpus-free** — Parts A/B need the real built shelves. Reads the live
  foods facet via the facade, exactly as `build_graph.ipynb`:
  - `fs.graph_store.list_shelves()` → the built `Shelf` list
  - `fs.llm` → the LLM client (configured to `llama-3.1-8b-instant`)
  - `api` (FoodOnAPI) for ancestors / labels / synonyms
- Runs on the `foodscholar` conda env. Requires a populated Neo4j/Elastic (a
  prior `fs.build_layer_a()`), same prerequisite as `build_graph.ipynb`.

## Non-goals (this round)

- No changes to `src/foodscholar/**` — prototype only. Where it lands in the
  build path is decided *after* Part C proves quality.
- No invented or relabeled categories (hard constraint above).
- Not wired into the pytest gate.
- Inline-in-build execution is out (drift/cost) — the eventual target is an
  offline, reviewable artifact, but that decision is deferred to post-prototype.

## Open questions to resolve during prototyping

- The exact fan-out target / "balanced" shape (tune in Part A/C, not fixed now).
- Whether one LLM pass per wide level suffices or tiers need to be inserted
  recursively (a tier may itself be too wide).
- How to handle FoodOn's DAG-ness when a child has several candidate ancestor
  tiers (multi-parent) — pick one for the nav tree, record others in `see_also`.
