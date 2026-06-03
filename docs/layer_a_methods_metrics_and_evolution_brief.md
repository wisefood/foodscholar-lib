# Layer A — Method Formulation, Metrics, and Construction-Exploration Evolution

**Purpose of this document.** This is a detailed, self-contained briefing intended
to be handed to a separate author (e.g. a Claude "co-work" agent) to produce a
formal, diagram-rich `.docx` for a project team. It specifies (i) the problem and
design space, (ii) the formal formulation of every construction method, (iii) the
formal definition of every evaluation metric, (iv) the benchmarking procedure and
real results, (v) a detailed narrative of how the layer-construction approach
evolved and *what changed at each step and why*, and (vi) an explicit list of
diagrams to produce. Nothing here assumes prior context.

> **Note to the DOCX author.** Produce a formal academic/technical report.
> Render every table below as a proper table; produce each figure listed in
> Section 9 as a diagram; keep the register suitable for an EU-project audience.
> Where this brief says "lower is better" / "higher is better", encode that in the
> figures (e.g. arrows, colour scales). The real measured results are in Section 7
> — use those numbers, not invented ones.

---

## 1. Context and objective

**Layer A is the entry-point (filter) index of the knowledge graph.** A user
arrives with a specific question (e.g. *"does olive oil affect cardiovascular
health?"*) and selects recognizable categories ("olive oil", "cardiovascular
health") to filter the corpus. Layer A provides the food-side categories — called
**shelves** — and attaches corpus **chunks** (passages) to them.

**Operative success criterion.** Layer A is a *faceted filter index*, not a
complete taxonomy. It "works" when: **for any food a user can plausibly name, a
recognizable entry point exists and is reachable in few interactions.**
Nameability and coverage dominate; ontological completeness is secondary.

**Substrate.** Shelves are grounded in **FoodOn**, an OWL ontology of foods. Every
shelf corresponds (where possible) to a real FoodOn identifier.

---

## 2. The core problem and the design space

### 2.1 Why this is hard: FoodOn has no single browsable food tree

Two structural properties of FoodOn dominate every design decision:

1. **No single food tree.** Foods are organized along multiple parallel
   classification axes (e.g. by material vs. product, by process, by consumer
   group). A single is-a projection cannot reproduce an intuitive, mutually
   exclusive set of food groups.
2. **Partial / counter-intuitive grouping.** A substantial fraction of
   corpus-mentioned foods (~38% in our corpus) do **not** lie under the principal
   `food product` branch, and the is-a graph frequently fails to connect a
   specific food to the human category a user expects (e.g. a specific fruit may
   not connect to a usable "fruit" *food* node).

### 2.2 The central tension

Every method is a bet trading two desirable properties against each other:

| Property | Definition | Cost of maximizing it |
|---|---|---|
| **Faithfulness** | The hierarchy is FoodOn's real (is-a or relational) structure. | Can be deep, wide, or sparsely populated — poor for browsing. |
| **Navigability** | A shallow, recognizable, well-populated tree. | Can require leaving FoodOn's real structure (fabricating groupings). |

### 2.3 The two-axis design space (key framing)

Every construction method is a choice on two **independent** axes:

- **Axis A — source of hierarchy / membership:** FoodOn structure (is-a, and
  optionally non-is-a relations) **vs.** LLM semantic judgement (by label).
- **Axis B — source of coverage:** **top-down** (start at ontology roots, prune
  downward) **vs. bottom-up** (start at corpus-mentioned leaves).

Bottom-up construction guarantees coverage by construction; structural membership
preserves faithfulness. The historically-conflated assumption — that "top-down" and
"faithful" are the same thing — is false, and untangling it is the core insight
(see Section 8).

---

## 3. Evolution of the layer-construction explorations (the detailed "what changed and why")

This section is the requested detailed note. Each subsection states **what the
approach was, what problem it targeted, what it changed relative to the previous
step, and why it was kept, rejected, or superseded.**

### 3.1 v0 — Top-down support-pruned projection (the original baseline)

- **What it was.** Start at FoodOn roots, walk down the is-a graph, prune branches
  by corpus support, and **lift survivors to the nearest surviving ancestor**;
  then collapse single-child chains. (This is the `prune.py` cascade: blacklist →
  umbrella rule → whitelist → support threshold → depth-cap lift → single-child
  collapse.)
- **Problem it targeted.** Produce a browsable foods facet directly from corpus
  support over FoodOn.
- **Why it failed.** On the real corpus it produced a flat, un-navigable blob:
  - **No coverage guarantee** — ~2,198 corpus-mentioned foods were pruned out
    entirely; a user who named them got nothing.
  - **Flat mega-parents** — `nearest_surviving_ancestor` collapsed everything onto
    a few hubs (e.g. `food product` ended with 111 direct children).
  - **Junk + fragmentation** — organizational labels, auto-suffixed labels
    (`broccoli food product`), and near-duplicate variants.
- **Diagnosis.** Support-driven pruning *attacks navigation nodes precisely because
  they are navigation nodes* (aggregators have low direct support), and top-down
  pruning loses coverage. **This is method "0 — Baseline" in the benchmark.**

### 3.2 Re-tiering patch (rejected)

- **What changed.** Insert FoodOn intermediate tiers to break the wide levels.
- **Why rejected.** It made fan-out and depth *worse*, not better — the
  intermediate tiers were themselves wide and sparse. Archived.

### 3.3 Projection bake-off (backbone-first vs. structural-cut vs. multi-facet)

- **What changed.** Stop deriving the tree from support alone. Instead choose a
  **category backbone first** (designed for browsing) and let support *decorate*
  it. Compared, side by side and judged by eye: a fixed-depth structural cut, an
  auto/structural backbone (**1a**), an LLM-proposed backbone (**1b**), and a
  multi-facet DAG (**3**).
- **What it surfaced.** The decisive empirical findings: the no-single-tree fact,
  the ~38% not-under-`food product`, and high multi-home rates. It established that
  a faithful is-a projection **cannot** be both shallow and complete.
- **Status.** Methodologically pivotal but **judged by eye** — which motivated the
  metric-driven harness (3.6).

### 3.4 Bottom-up + LLM semantic grouping (merged to `main`)

- **What changed (two things at once).** (a) Switched coverage to **bottom-up**:
  start from corpus-mentioned FoodOn leaves, so a mentioned food can never
  disappear. (b) Switched membership to **LLM-by-label**: the LLM proposes ~14
  human food groups (anchored to real FoodOn ids) and assigns each leaf to a group
  **by its label, deliberately ignoring is-a ancestry**.
- **Why.** It is navigable and guarantees coverage; it directly fixes v0's flat
  blob and dropped-foods problems.
- **The accepted trade-off (and the problem with it).** Membership is **not**
  is-a-derived, so it is **not faithful and not reproducible** from structure. This
  is the method the project lead was uncomfortable with — "we disregard FoodOn's
  taxonomy." The later benchmark (Section 7) confirms it is the **least faithful**
  method (75% of placements fabricated). **This is method "grouping (main)".**

### 3.5 `1a+` — Auto backbone + controlled expansion (the mechanical faithful-and-navigable method)

- **What changed.** Keep the structural backbone (1a), but **recursively open
  deeper FoodOn tiers only where corpus support clears a floor**, while **capping
  fan-out per parent and overall depth** and **skipping low-value single-child
  chains**. Every displayed node remains a real FoodOn id.
- **Why it matters.** It is **simultaneously faithful (real is-a structure),
  bottom-up (covered via support), and navigable (bounded width/depth)** — the
  first method to hit all three mechanically, with **no LLM**. It became the strong
  baseline that any LLM method must beat. **This is method "1a+".**

### 3.6 The metric-driven bake-off harness (stop judging by eye)

- **What changed.** Replace "by eye" comparison with a **reproducible benchmark**:
  every method emits a common representation; a fixed set of metrics is computed
  identically over all methods; a single **scorecard** plus per-decision audit logs
  is produced. Overlapping exploration notebooks were consolidated into one
  evaluation notebook.
- **Why.** The trade-off (Section 2.2) cannot be adjudicated by argument; it must be
  measured. This harness is what makes method selection evidence-based.

### 3.7 The agentic MCP method (the LLM-over-graph bet)

- **What changed.** Instead of grouping labels globally (3.4) or applying fixed
  support+cap rules (3.5), an **LLM agent traverses the real FoodOn support graph**
  and makes **local** keep/collapse/reparent decisions through a read-only tool
  interface (MCP-style), acting **only on real FoodOn edges** (membership stays
  is-a-faithful) and recording a rationale per decision.
- **The sharpened question it answers.** *Does context-aware, per-node LLM
  judgement over the real graph beat `1a+`'s mechanical rules — and is it worth the
  LLM cost and non-reproducibility?*
- **Status.** The **is-a core is built and benchmarked**; acting on **non-is-a
  relations** to bridge the 38% gap (the "relation-bridging" increment) is
  specified but deferred. **This is method "agentic".**

### 3.8 Summary of what each step changed

| Step | Coverage axis | Membership axis | Net change vs. prior | Verdict |
|---|---|---|---|---|
| v0 prune | top-down | is-a | — | flat blob; superseded |
| re-tiering | top-down | is-a | add intermediate tiers | worse; rejected |
| projection bake-off | mixed | is-a (+LLM backbone) | backbone-first, by eye | pivotal finding |
| grouping (main) | bottom-up | LLM by label | guarantee coverage; drop faithfulness | navigable, unfaithful |
| 1a+ | bottom-up | is-a + support | faithful **and** navigable, no LLM | strong baseline |
| harness | — | — | measure, don't argue | enables selection |
| agentic | bottom-up | is-a (+relations, deferred) | LLM local edits on real graph | labels win only |

---

## 4. Method formulations (formal)

### 4.1 Common substrate (shared by all methods)

- **Mentioned leaves.** From the annotated corpus, `collect_leaf_chunks` maps each
  FoodOn id mentioned in a chunk (via the `foodon_ids` denormalization or a
  foods-facet entity link above a confidence floor) to the set of chunk ids that
  mention it: `leaf_chunks : foodon_id → set(chunk_id)`.
- **Support roll-up.** For tree methods, each FoodOn node receives the union of
  chunk sets of itself and all is-a descendants restricted to the `food product`
  subtree: `node_support(n) = |⋃_{d ⪯ n} leaf_chunks[d]|`.
- **Common output representation (`MethodResult`).** Every method emits:
  `edges` (parent → ordered children), `labels` (node → display label), `counts`
  (node → chunk count), `leaf_home` (mentioned leaf → the node it is reachable
  under), `home_edge_type` (leaf → `is-a` | `other-relation` | `fabricated`),
  `home_distance` (leaf → is-a steps to its home), `llm_calls`, and `audit`.

### 4.2 Method 0 — Top-down prune (baseline)

Top-down cascade over FoodOn: drop blacklisted/organizational ("umbrella") nodes
and below-threshold nodes (by with-descendant support), lift survivors to the
nearest surviving ancestor within a depth cap, collapse single-child chains.
**Membership:** is-a (faithful). **Coverage:** top-down (lossy).

### 4.3 Method 1a — Auto structural backbone

Backbone = the direct is-a children of `food product`. Each chunk's food terms are
lifted to their nearest backbone ancestor; a second tier shows backbone children
that themselves have evidence. **Two-tier, faithful, bottom-up homing.**

### 4.4 Method 1a+ — Auto backbone + controlled expansion

Backbone as in 1a, then recursive expansion governed by parameters:

- open a child tier **iff** `node_support(child) ≥ EXPAND_MIN_CHUNKS` (default 25);
- emit at most `EXPAND_MAX_CHILDREN` (default 12) highest-support children per parent;
- stop at `EXPAND_MAX_DEPTH` (default 6);
- **skip single-child filing chains**: if a node has exactly one supported child
  that is itself not directly evidenced, show the grandchild under the node.

All displayed nodes are real FoodOn ids. **Faithful + bottom-up + navigable.**

### 4.5 Method 1b — LLM-proposed backbone

The LLM proposes 12–20 intuitive top-level category *names*; each name is resolved
to a real FoodOn id (drop unresolved); the resolved set is the backbone; lift as
in 1a. **Faithfulness medium; coverage is fragile** — if few names resolve, most
leaves cannot home (observed: coverage 0.17).

### 4.6 Method "grouping (main)" — Bottom-up + LLM label grouping

Bottom-up leaves; the LLM proposes ~14 groups (each anchored to a real FoodOn id)
and **assigns each leaf to a group by its label, ignoring is-a**. Each group is a
flat shelf (`display_label` = human name); unassigned leaves are kept as their own
shelves. **Membership by label → placements are fabricated when the leaf is not in
fact an is-a descendant of the group anchor.**

### 4.7 Method 3 — Multi-facet (DAG)

Like 1a, but a chunk attaches to **all** applicable backbone categories (a small
DAG rather than a tree), to quantify multi-home overlap. **One-tier.**

### 4.8 Method "agentic" — DFS MCP editor (detailed)

**Substrate.** The FoodOn support DAG (Section 4.1), rooted at `food product`.

**Tool surface (MCP-style).**
- *Read FoodOn:* `get_node`, `get_parents/children/ancestors`, `get_relations`
  (non-is-a relations from the relation index), `search` (semantic retrieval),
  `lowest_common_ancestor`.
- *Read/write graph-under-construction:* `keep`, `collapse`, `reparent`, and
  `expand_scope` (pull a real FoodOn node the induced subgraph dropped into scope).

**Per-node decision (the "lens").** Visiting node *N* with parent *P* and supported
children *Cᵢ*, the agent is shown: *N*'s label, *P*'s label, each *Cᵢ* label +
support, *N*'s support, and candidate non-is-a relation bridges. It returns one
action over **real edges only**:
- `KEEP` — *N* becomes a shelf;
- `COLLAPSE` — *N* is redundant with *P*; its children lift to *P*;
- `REPARENT` — *N* is an organizational artifact; its children lift to *P*.

**Guards** (bound the result by construction): `min_support` 25, `max_children`
12, `max_depth` 6.

**Faithfulness dial.** Reparent/expand may use (i) is-a only, (ii) is-a + real
non-is-a FoodOn relations, or (iii) + fabricate as a last resort. **Current build =
is-a core (i);** every placement is logged by relation type. Relation-bridging (ii)
is implemented at the data level (the relation index) and surfaced in the lens but
**not yet acted upon** — the deferred increment that targets the 38% gap.

**Implementation note.** The LLM client has **no native function-calling**, so the
agent loop is a manual `generate_json` loop with an `{action, reason}` protocol;
one call per visited internal node (hence the LLM-cost characteristic).

**Relation index (prerequisite, built).** Loaded directly from the FoodOn OWL via
pronto; keeps only FOODON→FOODON object-property edges. ~8,225 FOODON terms carry
≥1 non-is-a relation; the most useful food→food relations are `derives from`
(2,816), `member of` (2,342), `has defining ingredient`, `has ingredient`, and
`has food substance analog`.

---

## 5. Metrics (formal definitions)

All metrics are computed over the common `MethodResult`. Let `M` be the set of
corpus-mentioned leaves and `Q ⊆ M` a held-out query set.

| Metric | Formula / definition | Direction | Captures |
|---|---|---|---|
| **Coverage** | `|{ℓ∈M : ℓ∈leaf_home}| / |M|` | higher | Are named foods reachable at all? (≈1.0 for any bottom-up method) |
| **Specificity** | mean & median over homed ℓ of `home_distance(ℓ)` = number of is-a steps from ℓ up to its home node (0 if ℓ is itself the node) | **lower** | How *precisely* foods are placed; the true discriminator coverage is not |
| **Findability** | over `Q`: distribution of `depth(home(ℓ))` from root — median, 90th percentile, fraction ≤ K (K=3) | lower clicks | Interaction effort to reach a food |
| **Nameability** | fraction of a sampled set of shelf labels judged "recognizable to a layperson" by an LLM judge (sample 25) | higher | Are the categories human-readable? |
| **Fan-out** | max & median `|children(n)|` over internal nodes | lower (to a point) | Navigation width |
| **Depth** | max & median depth of homed leaves' home nodes | context-dependent | Navigation depth |
| **Faithfulness** | fractional split of `home_edge_type` over homed leaves into `is-a` / `other-relation` / `fabricated` | higher within-FoodOn (is-a + other-relation); lower fabricated | Is the structure real FoodOn? |
| **Reproducibility & cost** | `llm_calls` (cost); Jaccard overlap of node-id sets across two runs (stability) | lower cost; higher stability | Price and determinism |

**Metric caveats (must be stated in the DOCX):**
1. **Coverage does not discriminate** — it is ≈1.0 for any bottom-up method because
   any ancestor in the tree counts as a home. **Specificity** was introduced
   precisely to discriminate flat-blob vs. well-tiered placement.
2. **Findability currently under-rates deep trees** — a leaf is homed to its
   *deepest* node and `K=3` is arbitrary for a depth-6 design, so intrinsically
   deep methods (1a+, agentic) score low on `%≤K` even when foods are tightly
   placed (low specificity). A tier-aware findability variant is the planned
   refinement.
3. **`llm_calls` currently counts only the agentic method**; the grouping and 1b
   methods also call the LLM but report 0 — the cost column is incomplete.
4. **Nameability is a 25-label LLM-judged sample** — small differences are within
   noise; large gaps (e.g. 0.44 vs 0.28; grouping's fabrication) are meaningful.

---

## 6. Benchmarking procedure

1. Build the shared support layer once from a single corpus + FoodOn snapshot.
2. Run every method → common `MethodResult`.
3. Sample a held-out, **stratified** query set of foods (common + rare) for
   findability.
4. Compute all metrics → assemble the **scorecard** (one row per method).
5. Render each method's tree for inspection; for LLM methods, export the
   per-decision audit (decision → rationale → edge type).

**Corpus snapshot for the reported run:** ~13,000 chunks, of which ~4,200 mention
≥1 food; ~2,950 distinct food leaves; FoodOn snapshot ~39,000 terms.

---

## 7. Results (measured)

The following scorecard is the **measured** result with a live LLM (Groq
`llama-3.1-8b-instant`). Use these exact numbers in the DOCX.

| method | coverage | find_median | find_p90 | find_%≤3 | nameability | fanout_max | depth_max | spec_mean | faith_is-a | faith_fabricated | llm_calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 — Baseline | 1.00 | 2.00 | 2.00 | 1.00 | 0.60 | 111 | 3 | 4.92 | 1.00 | 0.00 | 0 |
| 2 — Structural cut | 1.00 | 3.00 | 3.00 | 0.92 | 0.24 | 85 | 4 | 2.96 | 1.00 | 0.00 | 0 |
| 1a — Auto backbone | 1.00 | 2.00 | 2.00 | 0.99 | 0.24 | 30 | 2 | 4.07 | 1.00 | 0.00 | 0 |
| 1a+ — Controlled expansion | 1.00 | 5.00 | 6.00 | 0.16 | 0.28 | 12 | 6 | 1.50 | 1.00 | 0.00 | 0 |
| 1b — LLM backbone | 0.17 | 2.00 | 2.00 | 0.19 | 0.12 | 9 | 2 | 2.89 | 1.00 | 0.00 | 0 |
| 3 — Multi-facet | 1.00 | 1.00 | 1.00 | 0.99 | 0.60 | 10 | 1 | 5.79 | 1.00 | 0.00 | 0 |
| agentic | 1.00 | 5.00 | 6.00 | 0.18 | 0.44 | 21 | 6 | 1.52 | 1.00 | 0.00 | 542 |
| grouping (main) | 1.00 | 1.00 | 1.00 | 1.00 | 0.40 | 472 | 1 | 10.41 | 0.25 | 0.75 | 0 |

### 7.1 Interpretation

- **Faithfulness cleanly isolates `grouping`:** 75% fabricated placements (only 25%
  is-a), worst specificity (10.41), and a 472-wide top level (a large ungrouped
  tail). It is shallow (1 click) but neither faithful nor genuinely browsable.
- **`1b` is broken:** coverage 0.17 — its LLM-proposed backbone resolved too few
  FoodOn ids to home most foods. Discard.
- **`1a+` and `agentic` are structurally the best and essentially tied:** lowest
  specificity (1.50 / 1.52), 100% is-a faithful, capped fan-out (12 / 21), depth 6.
- **`1a+` vs `agentic` (the key comparison):** equal specificity, faithfulness, and
  depth. The agentic method's *only* measured advantage is **nameability (0.44 vs
  0.28)** — more recognizable labels — at the cost of **542 LLM calls** and
  run-to-run non-reproducibility. **The LLM's value-add is curation (labels), not
  structure.**
- **Do not over-penalize the deep methods on `find_%≤3`:** that 0.16–0.18 is the
  deepest-home artifact (caveat 5.2); their specificity ≈1.5 shows foods are placed
  tightly.

### 7.2 Recommendation

- Drop `1b` and `grouping` for the foods facet.
- `1a+` clears the bar on structure, faithfulness, and width — and is **free and
  deterministic**. The agentic editor does not beat it structurally.
- **Likely production choice: a hybrid — `1a+` structure + a cheap one-shot LLM
  pass that only renames the kept shelves**, capturing the agentic nameability gain
  (≈0.28→0.44) without 542 calls. The full agentic editor is justified only if its
  deferred **relation-bridging** adds coverage/placement value on the 38% gap.

---

## 8. Key conceptual takeaways (for the team)

1. **"Top-down" ≠ "faithful."** The original failure was top-down *coverage loss*,
   not faithfulness; the fix was bottom-up coverage, which can be combined with
   faithful is-a membership (that is exactly what `1a+` and the agentic method do).
2. **Coverage is necessary but not sufficient** — specificity is the metric that
   reveals whether foods are actually placed well.
3. **An LLM over the real ontology graph improves labels, not structure** — at
   least for the is-a core; the open question is whether relation-bridging changes
   that.
4. **Method selection is now evidence-based** — the scorecard, not argument,
   decides.

---

## 9. Diagrams to produce (specification for the DOCX author)

Produce the following figures (suggested types in parentheses):

1. **Two-axis design space** (2×2 quadrant): X = coverage (top-down ↔ bottom-up),
   Y = membership (is-a ↔ LLM-by-label); plot each method as a point. Highlight the
   previously-unexplored "bottom-up + is-a" quadrant where `1a+`/agentic sit.
2. **Construction pipeline** (flowchart): corpus → entity linking → `leaf_chunks` →
   support roll-up → [method] → shelves (`MethodResult`) → metrics → scorecard.
3. **The FoodOn multi-axis problem** (graph illustration): a specific food (e.g.
   "apple food product") whose is-a path does not reach a usable "fruit" food node;
   contrast with the relation edge (`derives from`) that could bridge it.
4. **`1a+` controlled expansion** (flowchart/decision tree): backbone → for each
   node, support ≥ 25? cap children ≤ 12? depth ≤ 6? skip single-child chain?
5. **Agentic DFS loop** (flowchart): visit node → build lens → LLM returns
   KEEP/COLLAPSE/REPARENT → apply over real edge → guards → recurse → emit
   `MethodResult` + audit.
6. **MCP tool surface** (component diagram): read-FoodOn tools vs.
   read/write-graph tools, with the relation index and retriever as backends.
7. **Metric illustration** (annotated small tree): show one leaf and annotate
   coverage, specificity (is-a distance), depth/findability, and home_edge_type.
8. **Scorecard heatmap** (heatmap or grouped bars): the Section 7 table, with
   colour encoding metric direction (good vs. bad).
9. **Faithfulness–navigability map** (scatter): X = specificity (reverse axis, so
   right = more specific), Y = nameability, point size = `llm_calls`, colour =
   fabricated fraction; this single chart tells the whole story (`1a+`/agentic
   top-right and cheap/faithful; `grouping` bottom-left and fabricated).
10. **Evolution timeline** (timeline/stepped diagram): v0 prune → re-tiering →
    projection bake-off → grouping (main) → `1a+` → metric harness → agentic, with
    a one-line "what changed" per step (from Section 3.8).

---

## 10. Appendix

### 10.1 Default parameters

| Parameter | Default | Role |
|---|---|---|
| `EXPAND_MIN_CHUNKS` / agentic `min_support` | 25 | Minimum support to open/keep a tier |
| `EXPAND_MAX_CHILDREN` / agentic `max_children` | 12 | Fan-out cap per parent |
| `EXPAND_MAX_DEPTH` / agentic `max_depth` | 6 | Depth cap |
| Findability query set size | ~100 | Stratified common/rare foods |
| Nameability sample | 25 | Labels judged per method |
| LLM | Groq `llama-3.1-8b-instant` | Proposal / assignment / agent / judge |

### 10.2 Software manifest (for provenance, not for the DOCX body)

- Harness: `src/foodscholar/layer_a/bakeoff/{result,metrics,scorecard}.py`.
- Agentic: `src/foodscholar/layer_a/bakeoff/agentic/{relations,support,tools,agent}.py`.
- Evaluation notebook: `notebooks/layer_a_method_bakeoff.ipynb` (regenerable from
  `scripts/build_layer_a_method_bakeoff_nb.py`).
- Unit tests: 26 covering harness + agentic components.
- Companion docs: `docs/methods_layer_a_bakeoff_brief.md`,
  `docs/methods_layer_a_rework_brief.md`, and the two implementation plans under
  `docs/superpowers/plans/`.

### 10.3 Glossary

- **Shelf** — a Layer A entry-point category (usually a real FoodOn id).
- **Leaf** — a corpus-mentioned FoodOn term.
- **Home** — the node a leaf is reachable under in a given method's tree.
- **Backbone** — a chosen set of top-level categories that the tree hangs from.
- **Faithful** — the hierarchy uses real FoodOn edges (is-a or relations), not
  invented groupings.
