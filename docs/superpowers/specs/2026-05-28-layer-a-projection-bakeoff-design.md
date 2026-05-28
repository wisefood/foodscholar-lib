# Layer-A projection bake-off — design

The current Layer-A projection produces a flat, un-navigable foods facet. The
re-tiering patch made it *worse* (food product 111→116 children). Conclusion:
the **methodology** is wrong for a browsing layer, not a knob. This is a
**bake-off notebook** that renders 2–3 competing projection methodologies on the
same foods data, side by side, judged **by eye** — no production code until one
wins.

## The objections being addressed (user, 2026-05-28)

1. **Shape driven by support, not design.** The tree is the residue of a pruning
   cascade; it was never designed to be browsed.
2. **Flat / wrong grouping.** Everything piles onto mega-parents (`food
   product`, `Foods`) because `_nearest_included_ancestor` collapses to the
   deepest *surviving* ancestor, and the mid-level tiers were pruned away.
3. **Single tree can't capture it.** Food is multi-faceted; one
   parent-per-node tree is lossy.

FoodOn stays as the entity backbone. The **projection** is what changes.

## Core reframe

Stop deriving the browse tree from support. Choose a category **backbone**
first (designed for browsability, independent of corpus), then **attach**
corpus evidence by lifting each chunk to its nearest backbone node. Support
becomes a count that *decorates* the tree, not the thing that *defines* it.
Empty categories still render.

## Bake-off columns (same foods data, rendered side by side)

| Col | Methodology | Attacks |
|-----|-------------|---------|
| **0 — Baseline** | current build (`build_layer_a`), the flat blob | reference |
| **2 — Structural cut** | FoodOn-derived, fixed-depth horizon cut, keep real intermediate tiers regardless of support; NO nearest-surviving-ancestor collapse | #2 |
| **1a — Auto backbone** | backbone = structural rule (e.g. the `X food product` direct children of `food product`); chunks lift to nearest backbone node; support decorates | #1, #2 |
| **1b — LLM backbone** | backbone = `llama-3.1-8b-instant` proposes ~12–20 top-level food categories mapped to real FoodOn ids; else identical to 1a; needs `GROQ_API_KEY`, degrades gracefully | #1, #2 |
| **3 — Multi-facet** | a node may sit under several backbone categories (small DAG of browse axes), informed by the multi-home stats surfaced in 1a/1b | #3 |

## Lift rule

Default: each chunk attaches to its **closest (most-specific) backbone
ancestor** — single home, clean tree. BUT the notebook **surfaces multi-home
cases** (chunks whose FoodOn term sits under several backbone nodes) with
counts, so the tree-vs-DAG decision for Approach 3 is made on real frequency
data, not guessed.

## What "judged by eye" means

Each column renders as a nested `<details>` tree (the `explore_foodon`
renderer) with per-node chunk counts, plus a small stats line: top-level
fan-out, depth, % chunks homed, # empty categories, # multi-home chunks. All
columns in one `data/viz/projection_bakeoff_foods.html`.

## Infra (reuse, verified)

- Build foods facet **in-process** from `data/annotated.parquet` + in-memory
  stores (no Neo4j/Elastic): `load_chunks` → `attach_ontology` →
  `build_layer_a` (baseline only) → `graph_store.list_shelves()`.
- `FoodOnAPI` for ancestors/labels; chunk→FoodOn evidence from the loaded
  chunks' `foodon_ids` / `entity_links`.
- Only add the `llm` config block when `GROQ_API_KEY` is set (the Groq client
  is built eagerly and raises otherwise); else mock LLM, column 1b degrades.
- New notebook `notebooks/projection_bakeoff.ipynb`, generated from
  `scripts/build_projection_bakeoff_nb.py` (edit the script, not the JSON).

## Non-goals (this round)

- No `src/foodscholar/**` changes. The winning methodology gets a real design +
  implementation plan *after* the bake-off picks it.
- Not wired into the pytest gate.
- No claim that any column is "the answer" — the deliverable is the comparison.

## Open question the bake-off should answer

- Which methodology produces the most browsable tree by eye?
- How often are chunks genuinely multi-home (tree vs DAG)?
- Auto vs LLM backbone — does LLM curation beat a structural rule enough to
  justify the dependency?
