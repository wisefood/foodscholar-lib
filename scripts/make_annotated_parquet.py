"""Build data/annotated.parquet WITHOUT Elastic/Neo4j.

The bake-off notebook (and any in-memory run) reads `data/annotated.parquet`; it
never touches Elastic. This builds that snapshot from the pre-computed NEL CSVs
using an in-memory chunk store — fast and infra-free — so you don't have to wait
on (or fight) an Elastic ingest just to see the Layer-A tree.

    /mnt/miniconda3/envs/foodscholar/bin/python scripts/make_annotated_parquet.py
"""

from pathlib import Path

from foodscholar.config import FoodScholarConfig
from foodscholar.facade import FoodScholar

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "foodscholar" / "corpus"   # chunks_*.csv
NEL_DIR = ROOT / "data" / "foodscholar" / "ner"         # nel_chunks_*.csv
SNAPSHOT = ROOT / "data" / "annotated.parquet"

cfg = FoodScholarConfig.model_validate({
    "corpus": {
        "chunks_path": str(CORPUS_DIR),
        "annotated_snapshot_path": str(SNAPSHOT),
    },
    "ontology": {
        "foodon_path": str(ROOT / "data" / "foodon.owl"),
        "cache_path": str(ROOT / "data" / "foodon_cache.parquet"),
        "prefix_filter": ["FOODON:"],
    },
    # In-memory stores: ingest's parquet snapshot is written from chunk_store.scan()
    # regardless of backend, so memory gives the same parquet with zero infra.
    "storage": {
        "chunk_store": {"backend": "memory"},
        "graph_store": {"backend": "memory"},
    },
})

fs = FoodScholar.from_config(cfg)
# nel_dir supplied → annotations come from the pre-computed CSVs (no GLiNER/HNSW).
# ignore 'abstract' to match the build_graph notebook (those chunks carry no NEL).
meta = fs.ingest(
    CORPUS_DIR,
    nel_dir=NEL_DIR,
    snapshot_path=SNAPSHOT,
    ignore_source_types={"abstract"},
)
n = sum(1 for _ in fs.chunk_store.scan())
print(f"wrote {SNAPSHOT} ({SNAPSHOT.stat().st_size/1024:.0f} KB) · {n} chunks in memory")
if meta is None:
    print("(snapshot already existed and was non-empty — delete it to rebuild)")
