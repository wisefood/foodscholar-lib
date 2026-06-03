"""Run Layer A + Layer B end-to-end IN MEMORY (no Elastic/Neo4j) and print themes.

Reads data/annotated.parquet (build it first with make_annotated_parquet.py), then:
  load chunks -> build_layer_a -> attach -> embed -> build_layer_b(foods).

Embeddings: a `memory` backend uses a deterministic MOCK embedder by default, which
is INSTANT. Layer B's per-shelf gate needs chunks embedded at all, so the mock lets
both passes run — but only **Pass 2 (relatedness / entity co-occurrence)** themes are
semantically real with the mock; Pass 1 (similarity) is on toy vectors. Pass
`--real-embed` to load BGE-base for real Pass-1 themes too (slower, downloads ~440MB).

Labels use the keyword (c-TF-IDF) strategy so no real LLM is needed.

    /mnt/miniconda3/envs/foodscholar/bin/python scripts/run_layer_b_inmemory.py [--real-embed]
"""

import sys
import time
from pathlib import Path

from foodscholar.config import FoodScholarConfig
from foodscholar.facade import FoodScholar

_t0 = time.time()
def step(msg):
    print(f"[{time.time() - _t0:6.1f}s] {msg}", flush=True)

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data" / "annotated.parquet"
REAL_EMBED = "--real-embed" in sys.argv

if not SNAPSHOT.exists():
    raise SystemExit(f"{SNAPSHOT} missing — run scripts/make_annotated_parquet.py first.")

cfg = FoodScholarConfig.model_validate({
    "corpus": {"chunks_path": "tests/fixtures/sample_chunks.jsonl",
               "annotated_snapshot_path": str(SNAPSHOT)},
    "ontology": {"foodon_path": str(ROOT / "data" / "foodon.owl"),
                 "cache_path": str(ROOT / "data" / "foodon_cache.parquet"),
                 "prefix_filter": ["FOODON:"]},
    "layer_a": {"facets": ["foods"]},
    "storage": {"chunk_store": {"backend": "memory"}, "graph_store": {"backend": "memory"}},
})

fs = FoodScholar.from_config(cfg)
step("loading ontology…")
fs.attach_ontology(fs.load_ontology())
step("loading chunks from parquet…")
fs.load_chunks(str(SNAPSHOT))

if REAL_EMBED:
    from foodscholar.embedding import HFEmbedder  # type: ignore[import-not-found]

    fs.embedder = HFEmbedder("BAAI/bge-base-en-v1.5")
    step("using REAL BGE-base embedder (Pass 1 will be semantic)")
else:
    step("using MOCK embedder (Pass 2/relatedness real; Pass 1/similarity is toy)")
    # Mock vectors make Pass 1 (similarity) meaningless AND its global mutual-kNN is
    # the slow part — skip it so only Pass 2 (entity relatedness, the real one) runs.
    fs.config.layer_b.global_similarity_max_chunks = 1

step("build_layer_a…")
fs.build_layer_a()
step("attach…")
fs.attach()
step("embed…")
fs.embed()

# Offline labels (no real LLM): c-TF-IDF keywords instead of the mock LLM.
fs.config.layer_b.labeling.strategy = "keyword"
step("build_layer_b(foods)…")
art = fs.build_layer_b(facet="foods", dry_run=False)
step("done.")

print("\n=== Layer B artifact ===")
print(f"themes total       : {art.n_themes_total}")
print(f"themes by pass     : {art.n_themes_by_pass}")
print(f"shelves themed/skip: {art.n_shelves_themed}/{art.n_shelves_skipped}")

themes = sorted(fs.graph_store.list_themes(), key=lambda t: -t.chunk_count)
print(f"\n=== top {min(15, len(themes))} themes (by chunk_count) ===")
for t in themes[:15]:
    shelves = ", ".join(t.shelf_ids[:2]) + ("…" if len(t.shelf_ids) > 2 else "")
    print(f"[{t.discovery_pass:17}] {t.chunk_count:4d} chunks · {t.label}"
          + (f"  (shelves: {shelves})" if shelves else ""))
