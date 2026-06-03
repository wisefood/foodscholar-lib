# Layer A interactive tree + per-shelf Pass 1 — design

**Date:** 2026-06-03
**Status:** approved (brainstorm), pending spec review

## Goal

Two deliverables in one task:

1. Re-run Layer B for the `foods` facet with **per-shelf Pass 1** (`pass1_mode="per_shelf"`),
   replacing the current global-mode themes. The current global run produced 0 `merged`
   themes because global similarity communities span shelves and rarely Jaccard-match a
   single shelf's relatedness community; per-shelf Pass 1 yields shelf-scoped candidates
   that can align, so a real `merged` bucket appears.
2. A **standalone, self-contained HTML report** that renders the full Layer A shelf tree
   for `foods` with counts, where clicking a shelf reveals that shelf's themes grouped by
   origin (`merged` / `global_similarity` / `relatedness`).

## Decisions (from brainstorm)

- **Form:** standalone `.html` report written to `data/viz/` (the established viz output dir;
  also renderable inline by returning the HTML string when `output=None`).
- **Pass 1:** re-run `per_shelf`, replacing existing themes.
- **Tree scope:** show **all** shelves in the facet (227 for foods). Shelves below the
  `min_chunks_per_shelf` (=50) bar render greyed/sub-threshold with no themes.
- **Facet:** `foods` only — it is the only populated facet (others are synthetic roots).
- **Architecture:** extend the existing `viz` framework (Approach A): new `builder` function
  + new `"tree"` renderer + facade method, following the `builder → VizGraph → renderer`
  and `fs.viz.X(...).render(backend, output=...)` patterns already in the repo.

## Architecture & data flow

```
fs.build_layer_b(facet="foods")          # config.layer_b.pass1_mode="per_shelf" → replaces themes
        │
        ▼  Neo4j holds merged / global_similarity / relatedness themes
builder.layer_a_tree(fs, facet)  ──►  VizGraph
        │   • one VizNode(kind="shelf") per Shelf
        │       attrs = { chunk_count, support_direct, support_lifted, depth, foodon_id,
        │                 display_label, eligible: chunk_count >= min_chunks_per_shelf,
        │                 themes: { merged:[...], global_similarity:[...], relatedness:[...] } }
        │   • one VizEdge(kind="parent_of") per parent_shelf_id link
        ▼
TreeRenderer().render(graph, output="data/viz/layer_a_tree_foods.html")
        │   • nest flat nodes/edges into a children-tree, JSON-serialize, inject into template
        │   • output=None → return HTML string; else write file, return Path (base Renderer contract)
        ▼
fs.viz.layer_a_tree("foods").render("tree", output="data/viz/layer_a_tree_foods.html")
```

Output is self-contained: data baked in as JSON, vanilla JS/CSS, no external deps, no live
kernel. Opens in any browser, survives sharing and notebook reload.

**Theme → shelf assignment:** each theme lists `shelf_ids`; attach the theme to every shelf
in that list. With `per_shelf` Pass 1, similarity/relatedness/merged themes are single-shelf
by construction, so this is usually 1:1; cross-shelf attachment only occurs if any survive.
Each theme entry carries `{label, chunk_count, keyword_terms, discovery_pass}`.

## Components

### `builder.layer_a_tree(fs, facet="foods") -> VizGraph`
Location: `src/foodscholar/viz/builder.py` (alongside `shelf_view`, `backbone`).

- Iterate `fs.graph.shelves(facet=facet)` (same read API `backbone` uses; includes
  sub-threshold shelves). Each shelf handle `sh` exposes `sh.model` and `sh.themes()`.
- For each shelf, group `sh.themes()` by `theme.model.discovery_pass` into the three buckets
  (`merged` / `global_similarity` / `relatedness`) — no manual global indexing needed.
- One `VizNode(kind="shelf")` per shelf; `weight=chunk_count`; `facet=facet`; `attrs` per schema above.
- One `VizEdge(kind="parent_of", source=parent, target=child)` per `parent_shelf_id`.
- Return `VizGraph(title="Layer A tree — {facet}", level="L3", nodes, edges,
  attrs={"facet":..., "min_chunks_per_shelf":..., "n_shelves":..., "n_eligible":..., "n_themes":...})`.

### `TreeRenderer(Renderer)`
Location: `src/foodscholar/viz/renderers/tree_renderer.py`; `name="tree"`.
Registered in `view.py` `render()` factory and `renderers/__init__.py`.

- Reshape flat node/edge lists into nested `children` tree (roots = nodes with no incoming
  `parent_of`). JSON-serialize tree + header aggregates; inject into an HTML template.
- Honor base contract: `output is None` → return HTML string; else write file, return `Path`.
- Palette: reuse `FACET_COLORS`/`KIND_COLORS` from `renderers/base.py`; origin colors —
  `merged`=green, `global_similarity`=blue, `relatedness`=amber.

### `fs.viz.layer_a_tree(facet="foods") -> RenderableGraph`
Location: `src/foodscholar/viz/view.py`. Thin facade: `return RenderableGraph(builder.layer_a_tree(self._fs, facet))`.

## HTML UI

Two panes in a single file:

- **Left (tree, scrollable):** indented collapsible tree. Each row: expand/collapse caret
  (if children), eligibility marker (filled=eligible, hollow=has themes, ⊘=sub-threshold/greyed),
  shelf label, chunk count, `[theme count]`. Roots expanded by default.
- **Right (detail, sticky):** on shelf click — header (label, foodon_id, depth,
  chunk_count, direct/lifted support) then three labelled, color-coded origin sections
  (MERGED / SIMILARITY / RELATEDNESS), each listing its themes with label, chunk count,
  top keywords. Sub-threshold shelves show "below {min} chunk threshold — no themes."
- **Header:** facet totals (`N shelves · M eligible · K themes`). Origin filter dims
  themes not matching the selected origin.
- Vanilla JS only; static; works offline from the single file.

## Testing (pytest is the gate)

- `builder.layer_a_tree`: fake store → assert node count, `parent_of` edges, theme bucketing
  by origin, `eligible` flag at the `min_chunks_per_shelf` boundary.
- `TreeRenderer`: render a 2-shelf/3-theme `VizGraph` to string → embedded JSON round-trips,
  all three origin classes present, `output=path` writes a non-empty file. No JS execution.
- `per_shelf` Layer B re-run covered by existing `layer_b` tests; run full suite as the gate.

## Out of scope (YAGNI)

- Other facets (only `foods` is populated).
- Live/ipywidgets interactivity, search box, theme→chunk drill-down, card display.
- Comparing global vs per_shelf side by side (we replace, not compare).
