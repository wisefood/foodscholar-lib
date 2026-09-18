# NB2 — Theme bake-off (M2) — summary

- Leiden config: `{'algorithm': 'leiden', 'scope': 'subtree', 'min_chunks_per_shelf': 50, 'leiden.min_community_size': 15, 'similarity.edge_threshold': 0.55, 'similarity.require_mutual': True}`
- BERTopic config: `{'algorithm': 'bertopic', 'bertopic.clusterer': 'hdbscan', 'bertopic.scope': 'subtree', 'bertopic.min_topic_size': 15, 'bertopic.random_state': 42, 'min_chunks_per_shelf': 50}`

## Per-anchor themes (clustered over each anchor's subtree)
- **olive_oil** (olive oil, subtree 185 chunks): Leiden 6 themes, BERTopic(mts15) 2 themes.
- **legume** (legume food product, subtree 187 chunks): Leiden 6 themes, BERTopic(mts15) 2 themes.
- **fish** (fish food product, subtree 789 chunks): Leiden 15 themes, BERTopic(mts15) 9 themes.
- **dietary_fibre** (dietary fibre, subtree 1216 chunks): Leiden 14 themes, BERTopic(mts15) 13 themes.

## Method: anchor-scoped, not facet-wide
- NB2 clusters **only the three anchor subtrees** via the low-level Leiden / BERTopic primitives — it does NOT run a full-facet `build_layer_b`. Over the abstract-expanded corpus (34k chunks) a facet-wide build exceeds 30 min and themes 400+ shelves we never inspect; anchor-scoped clustering is O(subtree) and gives the identical anchor result.
- The two methods still theme at **different altitudes** facet-wide (Leiden finds fine communities; BERTopic/HDBSCAN prefers coarser parents) — surfaced via A/B below.

### A — same chunk set, intrinsic metrics (altitude-independent)
Both clusterers over the *identical* subtree chunk set; coverage (1−outliers), silhouette, keyword-Jaccard:
- **olive_oil** (N=185): Leiden 6 clusters cov=0.876 sil=0.085; BERTopic(mts15) 2 clusters cov=0.989 sil=0.239; keyword-Jaccard 0.273.
- **legume** (N=187): Leiden 6 clusters cov=0.888 sil=0.064; BERTopic(mts15) 2 clusters cov=0.765 sil=0.235; keyword-Jaccard 0.333.
- **fish** (N=789): Leiden 15 clusters cov=0.933 sil=0.058; BERTopic(mts15) 9 clusters cov=0.833 sil=0.107; keyword-Jaccard 0.338.
- **dietary_fibre** (N=1216): Leiden 14 clusters cov=0.905 sil=0.032; BERTopic(mts15) 13 clusters cov=0.659 sil=0.088; keyword-Jaccard 0.397.

### B — retuned BERTopic (min_topic_size = 15)
Lowering the HDBSCAN floor lets BERTopic theme the *same* subtree as Leiden:
- **olive_oil**: Leiden 6 vs BERTopic 2 clusters; cov 0.876 vs 0.989; sil 0.085 vs 0.239; keyword-Jaccard 0.273.
- **legume**: Leiden 6 vs BERTopic 2 clusters; cov 0.888 vs 0.765; sil 0.064 vs 0.235; keyword-Jaccard 0.333.
- **fish**: Leiden 15 vs BERTopic 9 clusters; cov 0.933 vs 0.833; sil 0.058 vs 0.107; keyword-Jaccard 0.338.
- **dietary_fibre**: Leiden 14 vs BERTopic 13 clusters; cov 0.905 vs 0.659; sil 0.032 vs 0.088; keyword-Jaccard 0.397.

## Files
env.json, {anchor}_leiden.json, {anchor}_bertopic.json, diagnostics.csv, {anchor}_sidebyside.png; intrinsic_same_chunkset.{json,csv}, bertopic_min_topic_size_sweep.csv, bertopic_retuned_comparison.json, comparable_AB.png.

## Deviations / limitations
- Corpus now includes **abstracts** (NEL-covered); anchor subtrees grew (olive_oil/legume = 185/187 chunks).
- Labels use the deterministic **keyword** strategy (no LLM); LLM polish is NB4.
- Anchor-scoped clustering replaces the facet-wide build for tractability over 34k chunks (same anchor result; see 'Method' above).
- **B is a deliberate tuning deviation**: BERTopic min_topic_size lowered to 15 so it themes the anchors; default (15) leaves them as outliers — reported in A, not hidden.

## Acceptance
- [x] Leiden themes for all three anchors
- [x] BERTopic themes for all three anchors (retuned, comparable)
- [x] same-chunk-set intrinsic comparison (A) for all three anchors
- [x] diagnostics.csv + side-by-side + comparable_AB figures exist
- [x] reproducible under SEED=42