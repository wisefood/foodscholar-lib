# NB3 — Retrieval flat vs graph (M3) — summary

- Build/backend: in-memory, Leiden themes from NB2 snapshot; k=10.
- Flat = `InMemoryChunkStore.search` (BM25+kNN RRF) fused with real BGE-base kNN. Graph = flat ∪ theme-siblings ∪ anchor-shelf chunks, re-ranked with a theme-membership weight (0.5).

## Deltas
- **olive_oil**: 5 graph-only chunks of 10; reasons {'paraphrase': 2, 'different-document': 3}.
- **legume**: 5 graph-only chunks of 10; reasons {'different-document': 5}.
- **fish**: 3 graph-only chunks of 10; reasons {'paraphrase': 1, 'different-document': 2}.
- **dietary_fibre**: 7 graph-only chunks of 10; reasons {'different-document': 4, 'different-food-entity': 1, 'paraphrase': 2}.

## Files
env.json, {anchor}_flat.json, {anchor}_graph.json, {anchor}_delta.json, {anchor}_compare.png.

## Deviations / limitations
- BM25 is the in-memory store's token-overlap approximation (no ES); the kNN side uses real BGE-base vectors (chunk cache + in-space query embedding).
- The delta `reason` is a heuristic (doc / FoodOn-entity / paraphrase); the narrative confirms a sample by hand.

## Acceptance
- [x] both queries produce flat + graph result sets
- [x] delta computed and each delta hit annotated
- [x] empty/noisy deltas reported, not hidden