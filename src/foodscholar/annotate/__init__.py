"""NER + entity linking + embedding phase.

Public surface:
  GLinerNER                            — `NER` protocol implementation
  HNSWLinker                           — `Linker` protocol implementation
  HNSWNELIndex / ElasticNELIndex       — NEL index backends
  HashEmbedder / HFEmbedder /
    SapBERTEmbedder / SourceTypeRouter — Embedder implementations
  run / dry_run                        — phase orchestration

Most users do not import from here directly — `fs.annotate()` and friends on
the FoodScholar facade are the canonical entry points.
"""

from foodscholar.annotate.embedder import (
    HashEmbedder,
    HFEmbedder,
    SapBERTEmbedder,
    SourceTypeRouter,
)
from foodscholar.annotate.gliner_ner import GLinerNER
from foodscholar.annotate.linker import HNSWLinker
from foodscholar.annotate.nel_index import (
    ENCODER_IDS,
    ElasticNELIndex,
    HNSWNELIndex,
    NELIndex,
)
from foodscholar.annotate.runner import dry_run, run

__all__ = [
    "ENCODER_IDS",
    "ElasticNELIndex",
    "GLinerNER",
    "HFEmbedder",
    "HNSWLinker",
    "HNSWNELIndex",
    "HashEmbedder",
    "NELIndex",
    "SapBERTEmbedder",
    "SourceTypeRouter",
    "dry_run",
    "run",
]
