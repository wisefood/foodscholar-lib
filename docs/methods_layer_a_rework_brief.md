# Layer A — reworked construction (brief)

What Layer A is now, why it changed, and the recipe we're porting into the
library. Companion to [`methods_layer_a_b_brief.md`](methods_layer_a_b_brief.md)
(the *old* top-down construction this supersedes for the foods facet).

## What Layer A is (reframed)

Layer A is a **facet / filter index**, not a browsable taxonomy. A user arrives
with a *specific question* ("does olive oil help heart disease?") and clicks
recognizable entry points (`olive oil`, `cardiovascular health`) to filter the
corpus. **"Works" = for any food a user would name, a recognizable entry point
exists and is findable.** Depth and completeness are secondary; nameability and
coverage are everything.

## Why the old construction was wrong

The old build was **top-down**: start at FoodOn roots, walk down, prune by
support, lift survivors to the nearest surviving ancestor. On the real corpus
this produced a flat, un-navigable foods facet:

- **No coverage guarantee** — ~2,198 corpus-mentioned foods (e.g. `bean`,
  `mackerel`, `porridge`) were pruned out entirely → a user names them, the
  filter returns nothing.
- **Flat blob** — `nearest_surviving_ancestor` collapsed everything onto a few
  mega-parents (`food product` had 111 direct children; the synthetic root 81).
- **Junk + fragmentation** — organizational labels (`food calorie datum`),
  auto-generated suffixes (`broccoli food product`, `meat (raw)`), and
  near-duplicate variants (`cow milk` / `cow whole milk` / `lowfat cow milk`).

We also established a hard structural fact: **FoodOn has no single foods tree.**
Foods are split across parallel axes (`food product` vs `food material` vs
by-process / by-consumer-group); ~38% of mentioned foods aren't under
`food product` at all, and its is-a graph doesn't reliably place common foods
under their common-sense group (e.g. `apple` doesn't connect to a usable "fruit"
food node). So a faithful is-a projection **cannot** yield clean food groups —
proven by a projected-tree prototype that, even with LLM tier-curation and
single-child collapse, stayed wide (fan-out 106) and deep (10 levels).

## The chosen recipe (bottom-up + LLM semantic grouping)

Validated in `notebooks/bottomup_entrypoints.ipynb` (Rule 3). Closes Layer A for
the foods facet:

1. **Bottom-up leaves = coverage by construction.** Start from the FoodOn leaf
   terms the corpus actually mentions (`DIRECT_CHUNKS`: leaf foodon_id →
   set(chunk_id)). A mentioned food can never disappear — the mention *is* the
   entry point. (Live corpus: 2,951 distinct mentioned food leaves.)

2. **LLM proposes ~14 human food groups.** Groq `llama-3.1-8b-instant` proposes
   recognizable top-level groups (Vegetables, Fruits, Dairy and Eggs, Grains and
   Bread, Meat and Poultry, Legumes and Beans, Nuts and Seeds, Fish and Seafood,
   Fats and Oils, Beverages, …). Each group is **anchored to a real FoodOn id**
   (resolve the group name → FoodOn node; never invent ids).

3. **LLM assigns each leaf to a group by LABEL (semantic, not is-a).** Batched
   classification of leaf labels into the group list (live: 2,686/2,951 by LLM,
   rest by keyword heuristic; 331 unassigned keep their own entry). This
   deliberately ignores is-a ancestry — because FoodOn's is-a graph is unreliable
   for grouping (the central finding above).

4. **Group entry = the group, displayed by its human name.** Chunk count per
   group = **distinct** chunks across its member leaves (union of chunk-id sets,
   so a chunk mentioning several foods in one group counts once).

5. **Variant-merge + synonym labels** (FoodOn's own data, no invention): collapse
   near-duplicate leaves by concept key; prefer a clean FoodOn synonym for the
   display label, else strip the ` food product` suffix.

Result: 14 recognizable, populated groups; every tracked food lands correctly
(`apple → Fruits`, `mackerel → Fish and Seafood`, `bean → Legumes and Beans`).
Navigable as a filter index — the goal.

### Accepted trade-off

Group **membership is LLM semantic judgment** (by label), anchored to real
FoodOn ids — it is **not** is-a-derived, so it isn't reproducible the way a pure
projection is. We accepted this: faithfulness to FoodOn's is-a structure costs
navigability, and navigability is the actual objective. The group set should be
**frozen + reviewed** (a committed artifact) to be stable across runs.

## Explored and rejected (kept in the notebook for reference)

- **Top-down prune** (current `prune.py`) — the flat blob above.
- **Re-tiering** (`retier_layer_a.ipynb`) — insert FoodOn intermediate tiers to
  break wide levels; made it worse.
- **Projection bake-off** (`projection_bakeoff.ipynb`) — backbone-first vs
  structural-cut vs multi-facet; surfaced the 43%-multi-home / no-single-tree
  findings.
- **Projected depth-preserving tree** (Rule 4, `bottomup_entrypoints.ipynb`) —
  faithful is-a tree + LLM tier-curation + single-child collapse; stayed wide
  (106) and deep (10). Most faithful, not navigable enough.

## What this means for the library port

The prototype's grouping is **leaf-label → group**. The library attaches
**chunks → shelves** via `attach.py`, which resolves chunks by FoodOn-id
ancestry. So the port must reconcile two things:

- A **group is a flat shelf** anchored to a real FoodOn id, with a **display
  label** distinct from the FoodOn label.
- A leaf's chunks attach to its assigned group's shelf. Since assignment is
  **by-leaf** (not is-a), `attach.py` either needs a precomputed
  `leaf_foodon_id → group_shelf_id` map, or the assignment must be persisted so
  attach can look it up (rather than re-deriving via ancestry).

### Concrete changes (planned)

1. **`Shelf` (io/graph.py):** add `display_label: str | None = None`. Groups are
   flat (depth 0/1), `foodon_id` = the group anchor, `label` = FoodOn label,
   `display_label` = human group name.
2. **`LayerAConfig` (config.py):** add a `bottom_up_grouping` config block
   (enable flag, group-proposal model, batch size, min-support floor for which
   leaves count). Per-facet via `facet_overrides`.
3. **New module `layer_a/grouping.py`:** `propose_groups(...)`,
   `assign_leaves_to_groups(...)`, `shelves_from_groups(...)` — mirrors the
   notebook's Rule 3, reusing `semantic_consolidation`'s LLM conventions
   (`generate_json`, batched, defensive int-coercion of indices, audit artifact).
4. **`build_layer_a` signature:** thread `llm: LLMClient | None` through (today
   the LLM is only used post-attach in `semantic_consolidate`).
5. **`_build_facet`:** if `bottom_up_grouping.enabled`, use the grouping path
   instead of `prune()`; else keep the old path (foods facet opts in;
   health/nutrients/etc. can stay on the old path until ported).
6. **Persist the leaf→group map** so `attach.py` resolves chunks to group
   shelves by assignment, not ancestry.
7. **Tests:** new `test_layer_a_grouping.py` (proposal, assignment, dedupe,
   shelves-from-groups, coverage = no dropped leaf, LLM-disabled fallback).

### Open tuning items (not blocking)

- Prompt the group proposal toward **mutually-exclusive food TYPES** (the live
  LLM sometimes emits cross-cutting groups like "Processed and Packaged Foods").
- A few organizational labels still leak among unassigned kept-leaf entries
  (`edible food`, `fish species`, `dietary supplement`) — nameability-guard polish.
- Decide whether to freeze the group set as a reviewed artifact (recommended for
  reproducibility) vs. re-propose each build.
