# Tuning Layer B coverage

Per-shelf Layer B themes a *subset* of a shelf's chunks — only those that land in a
Leiden community of at least `min_community_size` within their shelf's graph. Un-themed
chunks stay attached to their shelf and fully searchable; they're just not bucketed into
a sub-topic. If you want more chunks themed, loosen the per-shelf graph — at the cost of
smaller, noisier themes.

## The knobs

| Knob | Default | Loosening effect |
|---|---|---|
| `layer_b.leiden.min_community_size` | 15 | smaller communities survive — **biggest lever** |
| `layer_b.similarity.edge_threshold` | 0.55 | looser kNN edges → denser graph |
| `layer_b.similarity.require_mutual` | true | keep one-directional neighbours too |

## Let the sweep pick for you

`fs.sweep_layer_b()` runs the full Cartesian grid of the knobs above as
*non-mutating* `dry_run` builds (cheap keyword labels, `per_shelf` Pass 1),
scores each config, and returns a ranked table. Nothing is written — apply the
winner yourself and rebuild.

```python
result = fs.sweep_layer_b(facet="foods")   # full 160-config grid — slow
print(result)                              # ranked Markdown table, best first
result.best                                # winning config dict
result.to_frame()                          # pandas DataFrame for plotting
```

Scoring maximizes coverage and useful merged themes while penalizing duplicate
labels, tiny themes, and cross-shelf leakage, and keeps the theme count in a
sane band. Weights are fixed and documented in `layer_b/sweep.py`. Shrink the
grid by passing your own:

```python
result = fs.sweep_layer_b(facet="foods", grid={
    "leiden.min_community_size": [5, 8, 10],
    "similarity.edge_threshold": [0.45, 0.50],
})
```

From the CLI: `foodscholar sweep-layer-b -c config.yaml`.

## Inspect a single build

`fs.build_quality_report(facet="foods")` gives the same metrics for the
*current* persisted build plus WARN-level smells (high-lifted/low-direct
shelves, no-theme shelves, near-duplicate labels, themes that span too many
entities, labels echoing the parent shelf). It's read-only.

```python
art = fs.build_layer_b(facet="foods")          # pass1_mode defaults to per_shelf
print(fs.build_quality_report(facet="foods"))  # metrics + warnings, as Markdown
```

From the CLI: `foodscholar report-layer-b -c config.yaml`.

## Reading the result

- **`min` theme size equals `min_community_size`** — that floor is the binding
  constraint, so lowering it is the most direct way to raise coverage.
- A suggested sweep: start `(8, 0.45, False)`, then try `(10, 0.50, False)` and
  `(5, 0.45, False)`. Pick the point where coverage is acceptable *before* themes get
  too small to be meaningful.
- Coverage is a property of **per-shelf** Pass 1. The `"global"` mode themes more chunks
  but smears them across shelves — see [](../concepts/layer-b-themes.md) for why
  per-shelf is the default despite lower coverage.

```{tip}
Each run replaces the previous themes (the build clears stale ones first), so it's safe
to sweep repeatedly against the same graph.
```
