"""Evaluation gates per BRIEF §17.

  - linker: coverage / accuracy on a JSONL gold set (see `evaluate`).
"""

from foodscholar.evaluation.linker import GoldRecord, LinkerEvalReport
from foodscholar.evaluation.linker import evaluate as evaluate_linker
from foodscholar.evaluation.linker import load_gold as load_linker_gold

__all__ = [
    "GoldRecord",
    "LinkerEvalReport",
    "evaluate_linker",
    "load_linker_gold",
]
