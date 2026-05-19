"""NER + entity linking + embedding phase.

Public surface:
  KeywordNER / SciFoodNERAdapter    — NER protocol implementations
  ThreeTierLinker                   — Linker protocol implementation
  HashEmbedder / HFEmbedder / SourceTypeRouter — Embedder implementations
  run / dry_run                     — phase orchestration

Most users do not import from here directly — `fs.annotate()` and friends on
the FoodScholar facade are the canonical entry points.
"""

from foodscholar.annotate.dense_index import DenseIndex
from foodscholar.annotate.embedder import (
    HashEmbedder,
    HFEmbedder,
    SapBERTEmbedder,
    SourceTypeRouter,
)
from foodscholar.annotate.linker import ThreeTierLinker
from foodscholar.annotate.ner import KeywordNER, SciFoodNERAdapter, simplify_label
from foodscholar.annotate.runner import dry_run, run

__all__ = [
    "DenseIndex",
    "HFEmbedder",
    "HashEmbedder",
    "KeywordNER",
    "SapBERTEmbedder",
    "SciFoodNERAdapter",
    "SourceTypeRouter",
    "ThreeTierLinker",
    "dry_run",
    "run",
    "simplify_label",
]
