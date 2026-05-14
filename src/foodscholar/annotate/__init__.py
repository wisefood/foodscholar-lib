"""NER + entity linking + embedding phase.

Public surface:
  KeywordNER / SciFoodNERAdapter    — NER protocol implementations
  ThreeTierLinker                   — Linker protocol implementation
  HashEmbedder / HFEmbedder / SourceTypeRouter — Embedder implementations
  run / dry_run                     — phase orchestration

Most users do not import from here directly — `fs.annotate()` and friends on
the FoodScholar facade are the canonical entry points.
"""

from foodscholar.annotate.embedder import HashEmbedder, HFEmbedder, SourceTypeRouter
from foodscholar.annotate.linker import ThreeTierLinker
from foodscholar.annotate.ner import KeywordNER, SciFoodNERAdapter
from foodscholar.annotate.runner import dry_run, run

__all__ = [
    "HFEmbedder",
    "HashEmbedder",
    "KeywordNER",
    "SciFoodNERAdapter",
    "SourceTypeRouter",
    "ThreeTierLinker",
    "dry_run",
    "run",
]
