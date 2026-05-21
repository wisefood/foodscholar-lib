#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/miniconda3/envs/foodscholar/bin/python}"
CORPUS_DIR="${CORPUS_DIR:-/mnt/workspaces/foodscholar/corpus/big}"
OUT_PARQUET="${OUT_PARQUET:-/tmp/foodscholar_chunks_sample.parquet}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
export CORPUS_DIR OUT_PARQUET SAMPLE_SIZE

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" - <<'PY'
from collections import Counter
from itertools import islice
from pathlib import Path
import os

from foodscholar import FoodScholar
from foodscholar.corpus import ChunkAnnotation, iter_chunks, load_chunks, merge_annotations
from foodscholar.corpus import write_chunks_parquet
from foodscholar.layer_a import shelf_id_for_foodon
from foodscholar.ontology import FoodOnAPI, load_ontology

corpus_dir = Path(os.environ["CORPUS_DIR"])
out_parquet = Path(os.environ["OUT_PARQUET"])
sample_size = int(os.environ["SAMPLE_SIZE"])

if corpus_dir.exists():
    source = corpus_dir
else:
    source = Path("tests/fixtures/corpus_chunks.csv")
    print(f"Corpus dir not found, using fixture: {source}")

count = 0
by_type: Counter[str] = Counter()
for chunk in iter_chunks(source):
    count += 1
    by_type[chunk.source_type] += 1

print("streamed_chunks", count)
print("by_type", dict(by_type))

sample = list(islice(iter_chunks(source), sample_size))
written = write_chunks_parquet(sample, out_parquet)
restored = load_chunks(out_parquet)
assert len(restored) == written
assert restored[0].source_metadata == sample[0].source_metadata
print("parquet_round_trip", written, str(out_parquet))

fs = FoodScholar.in_memory()
fs.upsert_chunks(restored)
batch_count = sum(1 for _ in fs.chunk_store.iter_chunks(batch_size=250))
print("in_memory_chunks", len(fs.chunk_store.scan()))
print("in_memory_batches", batch_count)

first_id = restored[0].chunk_id
updated = merge_annotations(
    fs.chunk_store,
    [ChunkAnnotation(chunk_id=first_id, foodon_ids=["TEST:0000008"])],
)
assert updated == 1
assert fs.chunk_store.get(first_id).foodon_ids == ["TEST:0000008"]
print("annotation_merge_ok", first_id)

fs.attach_ontology(
    FoodOnAPI(load_ontology(Path("tests/fixtures/mini_foodon.obo")), prefix_filter=None)
)
fs.config.layer_a.min_support = 1
meta = fs.build_layer_a()
assert meta.record_count >= 1
assert fs.graph_store.get_shelf(shelf_id_for_foodon("TEST:0000008")) is not None
print("layer_a_ok", meta.record_count)
PY
