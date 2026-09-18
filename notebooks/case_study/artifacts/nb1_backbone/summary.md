# NB1 — Setup & backbone (M1) — summary

- Corpus: **34359 chunks** ({'textbook': 12194, 'guide': 1150, 'abstract': 21015}); real-embedded fraction **0.4216** (BGE cache, 14487 attached).
- Layer A: **1019 shelves** across all facets, 6 roots. Shelves per facet: {'allergies': 1, 'dietary_patterns': 1, 'foods': 402, 'nutrients': 391, 'health': 157, 'sustainability': 67}.

## Multi-facet projection
- `prefix_filter=None`: the projection admits ALL OBO terms in the cache, so the non-food facets populate from their ontologies (CHEBI/CDNO→nutrients, UBERON/PATO→health, ENVO→sustainability). 4 of 6 facets are populated; `dietary_patterns` and `allergies` stay stubs (no entity_type/prefix routes to them and the prototype NER leaves entity_type='other').
- Graph-wide before/after for the projection transforms (all facets, read-only) is in `projection_transforms.json`.

## Anchors
- **olive_oil** — `olive oil` (FOODON:03301826), depth 4, 185 chunks, fallback_used=False. Path: Foods > Plant-based foods > plant oils and fats > Refined Plant Oils > olive oil
- **legume** — `legume food product` (FOODON:00001264), depth 3, 1455 chunks, fallback_used=False. Path: Foods > Plant-based foods > fruit > bean
- **fish** — `fish food product` (FOODON:00001248), depth 3, 789 chunks, fallback_used=False. Path: Foods > Meat and Seafood > vertebrate food product > fish products
- **dietary_fibre** — `dietary fibre` (CDNO:0000005), depth 2, 1512 chunks, fallback_used=False. Path: Nutrients > Food Components > fibre

## M1 raw vs projected
- **olive_oil**: raw FoodOn depth 16 → projected depth 4 (Δ12); raw sibling fan-out 2, longest single-child run 0.
- **legume**: raw FoodOn depth 7 → projected depth 3 (Δ4); raw sibling fan-out 2, longest single-child run 0.
- **fish**: raw FoodOn depth 8 → projected depth 3 (Δ5); raw sibling fan-out 19, longest single-child run 0.
- **dietary_fibre**: raw FoodOn depth 2 → projected depth 2 (Δ0); raw sibling fan-out 12, longest single-child run 0.

## Files
env.json, corpus_info.json, anchor_shelves.json, m1_raw_vs_projected.json, projection_transforms.json, stats.json, stats.csv; figures foods_backbone.html, {key}_breadcrumb.png, m1_{key}.png, depth_hist.png, support_{key}.png; scale_comparison.{json,png}; shared_graph_manifest.json.

## Deviations / limitations
1. Second anchor re-homed from `nutrients`/'dietary fibre' to `legume food product` (foods). Original reason was that a FoodOn-only Layer A had no nutrient hierarchy; the multi-facet build now DOES populate `nutrients` (from CHEBI/CDNO), but the anchor stays on `legume` for continuity with NB2–NB4. See prereg.md.
2. annotated.parquet carries no vectors; real BGE-base vectors loaded from the npy cache instead of the dim-8 mock embedder. Coverage is **40%** of all chunks (13,767 of 34,359 — textbook/guide + anchor-subtree abstracts only); the rest are BM25-only at search time.
3. PNG via graphviz unavailable (no `dot` binary): interactive tree is HTML; analytic figures are matplotlib.
4. **Non-food facets inherit NEL mislink noise**: prefix routing sends some links to the wrong facet (e.g. ancestry/HANCESTRO terms land in `health`, disease→UBERON), and the cache's non-FOODON coverage is thin (e.g. ~200 UBERON terms vs ~12.6k UBERON links). nutrients/health/sustainability are populated but noisier and sparser than foods. See `layer_a/facet.py` route_link_to_facet.

## Acceptance
- [x] all three anchors resolved (no fallback)
- [x] backbone HTML + breadcrumb/M1 PNGs exist
- [x] M1 file shows raw-vs-projected contrast
- [x] non-food facets populated (nutrients/health/sustainability); 2 remain stubs
- [x] stats reproduce under SEED=42 on a fixed stack