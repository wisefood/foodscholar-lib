#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/miniconda3/envs/foodscholar/bin/python}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m pytest \
  tests/unit/test_corpus_loader.py \
  tests/unit/test_annotation_merge.py \
  tests/unit/test_layer_a.py \
  tests/unit/test_in_memory_stores.py \
  tests/unit/test_io_models.py \
  tests/unit/test_facade.py \
  tests/unit/test_cli.py \
  tests/unit/test_facade_ontology.py
