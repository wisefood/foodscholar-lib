"""Run GLiNER + HNSW annotation end-to-end from a single corpus CSV/parquet.

This is the release-ready entry-point. It mirrors the validated `gliner.py`
prototype: load → annotate → upsert → optional parquet snapshot, in one call.

    python examples/02_run_annotation.py path/to/chunks.csv

Prereqs:
    pip install 'foodscholar[annotate]'
    # plus a FoodOn .owl file at data/foodon.owl (see ontology/README).
"""

from __future__ import annotations

import sys
from pathlib import Path

from foodscholar import FoodScholar


def main(chunks_path: Path) -> None:
    fs = FoodScholar.from_config(
        {
            "corpus": {
                "chunks_path": str(chunks_path),
                "annotated_snapshot_path": "data/annotated.parquet",
            },
            "ontology": {
                "foodon_path": "data/foodon.owl",
                "cache_path": "data/foodon_cache.parquet",
                "prefix_filter": ["FOODON:"],
            },
            "annotate": {
                "ner": "gliner",
                "batch_size": 16,
                "linker": {
                    "nel_backend": "hnsw",
                    "nel_encoder": "biolord",
                    "nel_min_sim": 0.70,
                },
            },
            "storage": {
                "chunk_store": {"backend": "memory"},
                "graph_store": {"backend": "memory"},
            },
        }
    )

    meta = fs.load_and_annotate(chunks_path)
    if meta is None:
        print("snapshot already exists — re-run skipped")
        return
    print(
        f"annotated {meta.record_count} chunks  "
        f"(artifact={meta.artifact_id}, config_hash={meta.config_hash})"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: 02_run_annotation.py <chunks.csv|chunks.parquet>")
    main(Path(sys.argv[1]))
