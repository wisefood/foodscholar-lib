# Layer A — method bake-off + agentic MCP construction (brief)

A plan to (1) **formalize the existing projection bake-off into a metric-driven
benchmark harness** that scores every Layer-A construction method on the same
footing, (2) add a new **agentic, MCP-style** construction method that stays
inside FoodOn, and (3) **consolidate** the scattered Layer-A exploration
notebooks into a single evaluation entry point.

Companion to [`methods_layer_a_rework_brief.md`](methods_layer_a_rework_brief.md)
(the bottom-up + LLM-grouping method currently on `main`). This brief does **not**
pick a winner — it builds the apparatus to choose one, and explicitly defers the
design of an *optimally balanced* method to a follow-up brainstorm (see §6).

> **Note (2026-06-02): a lot of this already exists.** Commit `b866e45` added a
> **`1a+` controlled-expansion** column to [`projection_bakeoff.ipynb`](../notebooks/projection_bakeoff.ipynb),
> and that notebook already has shared homing/render/stats helpers and six method
> columns. This brief is updated to *extend* that apparatus, not rebuild it — see
> §4/§5. `1a+` turns out to be the faithful-and-navigable mechanical method we'd
> have built anyway, and becomes the **control** the agentic method must beat.

## Why

Layer A construction has been judged "by eye" across several one-off notebooks.
Two goods are in tension and every method so far sacrifices one:

- **Faithfulness** — the hierarchy *is* FoodOn's real structure (defensible,
  reproducible).
- **Navigability** — a user names any food and finds a recognizable entry point
  in a shallow, readable tree.

We can't reason about the trade-off without measuring it. So: a harness that
scores methods on shared metrics, and a new method that tests whether an LLM
reasoning over the FoodOn graph beats the mechanical `1a+` rules on that trade-off.

## §2 — The methods under comparison

Run on identical input (one corpus snapshot, one FoodOn snapshot). The shared
support layer (`CHUNK_TERMS`, `TERM_DOC_FREQ`, `lift_to_backbone`) already exists
in `projection_bakeoff.ipynb`.

| # | Method | Status | Character |
|---|---|---|---|
| 0 | **Top-down prune / baseline** | built | flat blob — "where we started" |
| 1a | **Auto backbone** | built | structural-rule backbone, support decorates |
| **1a+** | **Auto backbone + controlled expansion** | **built (`b866e45`)** | **faithful + navigable mechanical method — the control** |
| 1b | **LLM backbone** | built | `llama-3.1-8b-instant` proposes the backbone |
| 3* | **Bottom-up + LLM grouping** | built (`main`) | navigable, groups by label, **ignores is-a** |
| — | **Agentic MCP editor** | **to build** | LLM reasons over the FoodOn graph; faithful by construction — see §3 |

(*“3” here is the `build_grouped_shelves` method from
`bottomup_entrypoints.ipynb`, distinct from the bake-off’s multi-facet col 3.)

**`1a+` in one line:** keep the auto backbone, then recursively open deeper FoodOn
tiers **only where supported** (`min 25 chunks`), **cap fan-out** (`12/parent`) and
**depth** (`6`), and **skip single-child filing chains** — every displayed node
stays a real FoodOn id. So it is faithful, bottom-up (coverage via support), and
navigable (bounded width/depth) at once. It is the strongest existing balance and
the baseline the agentic method has to justify itself against.

## §3 — The agentic MCP construction method

The sharpened question: **does an LLM making *local* keep/collapse/expand
decisions, with the ability to pull in more of the graph, beat `1a+`'s mechanical
support+cap rules** on navigability and nameability — and is it worth the LLM cost
and reproducibility hit? Same moves as `1a+`, but judged by reading the
neighborhood instead of by fixed thresholds, plus gap-bridging `1a+` can't do.

An **agent walks the FoodOn support DAG top-down and acts through a tool interface
(MCP-style) over the graph-under-construction.** Membership stays is-a-faithful
because the agent only acts on real FoodOn edges.

### Tool surface (the MCP)

**Read FoodOn:**
- `get_node(id)`, `get_parents/children/ancestors(id)`
- `get_relations(id)` — non-is-a FoodOn relations (`derives_from`, by-source …)
  *(prerequisite — see §3 note)*
- `search(label)` — backed by the existing `OntoRagRetriever` (tri-hybrid
  BM25 + MiniLM + SapBERT, already shipped)
- `lowest_common_ancestor(ids)` *(small new util over ancestor sets)*

**Read/write the graph-under-construction:**
- `current_subtree(shelf)`, `keep(id)`, `collapse(id)`, `reparent(child, parent)`
- `expand_scope(id)` — pull a real FoodOn node the induced subgraph had dropped
  (e.g. a `fruit` node) into the working set so it can become a parent

### Per-node decision

At node *N* (parent *P*, children *Cᵢ*) the agent sees a **lens** — *N*'s label,
*P*'s label, each child's label + support, *N*'s direct-vs-lifted support, and a
few sample chunk titles — and returns one action over **real edges only**:
`KEEP`, `COLLAPSE` (redundant → `see_also`), or `REPARENT` (organizational umbrella
→ lift children to *P*). These mirror the mechanical moves `1a+` already makes; the
experiment is whether LLM judgment makes them *better*.

### Guards (reuse `1a+`'s tuned values)

Start from `1a+`'s already-tuned constraints rather than inventing new ones:
**fan-out cap 12/parent**, **depth cap 6**, **min support 25 chunks** to open a
tier. Tune from the scorecard.

### Faithfulness setting: is-a + other FoodOn relations

`reparent`/`expand_scope` may use **real FoodOn is-a edges *or* real non-is-a
FoodOn relations** when is-a has no path — but **never fabricates** a parent.
Every edge logs its relation type (`is-a` / `other-relation`), feeding the
faithfulness metric and the audit. This is the dial we chose: close gaps while
staying 100% inside FoodOn.

> **Known prerequisite (decision pending).** The production ontology loader
> currently retains **is-a only** — the non-is-a relations this setting needs are
> not loaded. `LLMClient` also has **no native tool-calling**, so the agent loop
> is a manual `generate_json` loop with an `{action, args}` convention.
> *Recommended* path for the prototype: build a **throwaway relation index** in
> the notebook (load relations straight from the FoodOn OWL once) so the bake-off
> can measure whether relations actually close gaps *before* investing in a
> production loader change. Alternative: run the agent at **is-a-only** for v1.
> **Not yet decided — revisit when speccing.**

## §4 — The benchmark harness (extend, don't rebuild)

`projection_bakeoff.ipynb` already computes, per method, a `stats_line` with
**top fan-out, depth, %-chunks-homed (coverage), empty-category count, and
multi-home count**, and renders each method's tree side by side. That is the
bake-off's existing "by eye" layer.

**What to add to turn it into a real benchmark** (the missing "whys"):

| Metric | Status | Computation |
|---|---|---|
| **Coverage** | ✅ exists (`%-homed`) | % of mentioned leaves reachable from a root |
| **Fan-out** | ✅ exists | max / median children per shelf |
| **Depth** | ✅ exists | max / median |
| **Findability** | 🔨 add | held-out ~100 corpus-foods → min clicks root→containing shelf; median / p90 / %≤K |
| **Nameability** | 🔨 add | LLM-judge a random sample of shelf labels (recognizable y/n) → % |
| **Faithfulness** | 🔨 add | % of parent edges is-a / other-relation / fabricated (trivially is-a for projection methods; the discriminator for grouping + agentic) |
| **Reproducibility / cost** | 🔨 add | # LLM calls; run twice → shelf-set Jaccard stability |

**Findability query set:** ~100 corpus-mentioned foods, stratified common/rare,
held out from any prompt. Honest because a food with no chunks can't be found by
*any* method — so we only test foods actually present in the corpus.

**Output:** a single **scorecard** (methods × metrics) above the existing tree
columns, plus a **per-decision audit log** for the LLM methods (decision →
one-line rationale → edge type). Scorecard + trees + audits = compare with whys
and hows.

## §5 — Notebook consolidation

Layer-A exploration is spread across overlapping notebooks
(`bottomup_entrypoints`, `projection_bakeoff`, `retier_layer_a`,
`entrypoint_audit`) — the source of confusion.

**Plan:** make **`projection_bakeoff.ipynb` the single method-evaluation entry
point** (it already hosts most methods + the shared harness). Extend it with the
§4 metrics, fold in the bottom-up LLM-grouping method and the agentic method as
additional columns, and rename it to reflect its role (e.g.
`layer_a_method_bakeoff.ipynb`, regenerated from its build script). **Archive**
`retier_layer_a` and `entrypoint_audit` (move to `notebooks/archive/` or mark
"superseded"). `explore_foodon`, `build_graph`, and the ingestion smoke notebook
stay — they aren't method comparisons.

## §6 — Deferred: the *optimally balanced* method

This brief builds the apparatus, not the final answer. The next step (a separate
brainstorm) designs a method that **deliberately balances faithfulness and
navigability**, informed by the harness numbers. **`1a+` is the leading
candidate** — already faithful + navigable + covered — but it competes on the
scorecard like every other method; the question is whether it (or a tuned/hybrid
version of it, or the agentic editor) hits the bar. Open questions:

- Which metrics actually trade off, and where's the knee?
- Is the balanced method a *tuning* of `1a+` / the agentic editor (caps, expansion
  aggressiveness), a *hybrid* (faithful primary, grouping fallback for true
  orphans), or something the numbers suggest we haven't named?
- What's the acceptance bar on each metric for "good enough to ship"?

**Do not design the balanced method until the §4 metrics exist and we've looked at
real numbers.**

## Scope notes

- This brief covers the **prototype** (metrics + agentic method in the bake-off
  notebook). The production port of whatever wins, and any production
  ontology-loader extension for non-is-a relations, are **explicit follow-ups**.
- Reusable as-is: the entire `projection_bakeoff` harness (`lift_to_backbone`,
  `render_tree_from_edges`, `stats_line`, the `1a`/`1a+`/`1b` columns),
  `collect_leaf_chunks`, `collect_support`/`SupportTable`, `prune()`,
  `build_grouped_shelves()`, `OntoRagRetriever`, `FoodOnAPI` graph nav.
- To build: the §4 metric functions (findability, nameability, faithfulness
  edge-type, reproducibility) + scorecard, the findability query set, the agentic
  loop + tool wrappers (manual, no native tool-calling), `lowest_common_ancestor`,
  and (pending §3) the throwaway relation index.

## Decisions still pending

1. **Relation prerequisite (§3):** throwaway notebook relation index *(recommended)*
   vs. is-a-only v1 vs. extend the production loader now.
2. **Guard caps** — start from `1a+`'s values (fan-out 12, depth 6, min 25), tune
   from the scorecard.
3. **Archive location** for the superseded notebooks (`notebooks/archive/` vs.
   in-place "superseded" banner).
