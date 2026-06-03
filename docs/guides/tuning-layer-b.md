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

## A tuning cell

Re-run this per knob set and watch `coverage` rise as median theme size falls. Keyword
labels keep it fast and free while sweeping; flip `labeling.strategy` to `"llm"` for the
final run.

```python
import statistics as _stats

cfg = fs.config.layer_b
cfg.pass1_mode = "per_shelf"
cfg.labeling.strategy = "keyword"        # fast + free while tuning

# --- knobs (defaults in comments) ---
cfg.leiden.min_community_size = 8        # default 15 — biggest lever
cfg.similarity.edge_threshold = 0.45     # default 0.55
cfg.similarity.require_mutual = False    # default True

art = fs.build_layer_b(facet="foods", dry_run=False)

# --- coverage = attached chunks that landed in a theme ---
attach = fs.graph_store.list_chunk_shelf_attachments()
foods = {s.shelf_id for s in fs.graph_store.list_shelves() if s.facet == "foods"}
attached = [cid for cid, sids in attach.items() if sids & foods]
themed = sum(1 for c in fs.chunk_store.get_many(attached) if c.theme_ids)
sizes = [t.chunk_count for t in fs.graph_store.list_themes()]

print(f"themes {art.n_themes_total}  by pass {art.n_themes_by_pass}")
print(f"coverage {themed}/{len(attached)} = {themed/len(attached):.0%}"
      f"   median theme size {int(_stats.median(sizes)) if sizes else 0}")
```

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
