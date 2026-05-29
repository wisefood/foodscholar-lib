# FoodOn structure exploration — design

A standalone, corpus-free analysis notebook that surfaces the structural
problems of the raw FoodOn ontology — the problems the Layer-A projection
exists to fix. Output is a self-contained HTML report for internal
understanding and demos.

## Motivation

Layer A projects FoodOn onto our corpus to produce a small, navigable set of
"shelves". The shelves we currently get are sparse and not very navigable.
Before tuning the projection further, we want to understand *what is wrong with
FoodOn itself* — independent of any corpus — so that each projection rule
(`max_depth` capping + lifting, single-child chain collapse, the umbrella
rule, curation) has a visible, quantified motivation.

## Audience & scope

- **Audience:** internal / our own understanding. Dense, technical, exploratory.
  Captions are terse; maximize qualitative + quantitative coverage.
- **Grounding:** pure FoodOn structure only. No corpus, no Elastic, no Neo4j.
  Fully reproducible from `data/foodon.owl` alone.
- **Viz:** interactive (collapsible/zoomable) for the headline subtree; static
  matplotlib figures for distributions/tables.

## Environment

Runs on the existing `foodscholar` conda env (Python 3.11,
`/mnt/miniconda3/envs/foodscholar/bin/python`), which has `numpy`, `pandas`,
`networkx`, `pyvis`, `matplotlib`, `graphviz`, `jinja2`, and the `foodscholar`
package installed. **No `plotly`** — interactive viz uses `pyvis` (already the
basis of the existing `data/viz/*.html` artifacts).

Reuse the project's own loader; do not re-parse the OWL:

```python
from foodscholar.ontology import load_ontology, FoodOnAPI
terms = load_ontology("data/foodon.owl", cache_path="data/foodon_cache.parquet")
api = FoodOnAPI(terms, prefix_filter=("FOODON:",))
```

## Foundation (§0)

`FoodOnAPI` exposes labels, synonyms, parents, children, ancestors,
descendants, and the obsolete flag — but **no depth**. §0 builds one
`networkx.DiGraph` (parent→child edges over FOODON: terms) and computes a
single per-node pandas frame that all downstream sections read:

| column        | meaning                                                        |
|---------------|----------------------------------------------------------------|
| `id`          | FOODON id                                                      |
| `label`       | term label                                                     |
| `depth_min`   | shortest root→node path (FoodOn is a DAG, so min ≠ max)        |
| `depth_max`   | longest root→node path                                         |
| `n_children`  | direct children                                                |
| `n_parents`   | direct parents (>1 ⇒ multi-parent / DAG-ness)                  |
| `subtree`     | transitive descendant count                                    |
| `obsolete`    | obsolete flag                                                  |

Roots = nodes with `n_parents == 0`. Depth computed by BFS/longest-path over
the DiGraph (note: a true longest-path on a DAG; cycle-guard in case the OWL
has any).

## The four problem sections

Each section = a short summary stat line + one figure + a terse caption tying
the finding to the projection rule that answers it.

### §1 — Too deep / over-specific
- depth histogram (`depth_max`)
- a few example deepest root→leaf paths rendered as label chains
- depth-vs-subtree-size scatter (deep terms are leaves, not destinations)
- **answers:** `max_depth` capping + lifting

### §2 — Single-child scaffolding
- count of parents with exactly one child
- longest single-child chains shown as label sequences
- **answers:** single-child chain collapse

### §3 — Umbrella / abstract nodes
- top terms by `n_children` (fan-out giants) and by `subtree` (the abstract
  spine: `food product`, `material entity`, …)
- **answers:** the umbrella rule

### §4 — DAG-ness & label noise
- multi-parent term count + worst offenders (`n_parents` desc)
- obsolete-term count
- label-length distribution + auto-generated-label pattern detection
  (e.g. `(... , ...)` modifier suffixes), synonym sprawl
- **answers:** *why a curated projection is needed at all*

### Headline interactive widget
One `pyvis` collapsible/zoomable subtree (rooted at an abstract node such as
`food product`) so the over-deep, single-child reality is explorable live in a
demo. Generated as a self-contained HTML fragment and embedded in the report.

## HTML assembly

A final cell collects every section's figure (matplotlib → inline base64 SVG),
summary table, and caption, plus the pyvis widget (embedded as an
`<iframe srcdoc=...>` so it is self-contained), and renders them through a
small `jinja2` template to `data/viz/foodon_report.html`. No external asset
dependencies — the file opens standalone.

## Non-goals

- No corpus overlay (explicitly chosen: pure structure).
- No changes to `src/foodscholar/**` — this is analysis only, reusing the
  existing loader/API.
- No new dependency (`plotly` is *not* added; `pyvis` covers interactivity).
- Not wired into the pytest gate — it's a notebook artifact.
