"""Layer C — theme summarization (extractive Stage 1 → LLM Stage 2)."""

from foodscholar.layer_c.benchmark import benchmark_facet, benchmark_theme
from foodscholar.layer_c.builder import build_layer_c
from foodscholar.layer_c.models import LayerCReport, MethodResult, Stage1Output

__all__ = [
    "LayerCReport",
    "MethodResult",
    "Stage1Output",
    "benchmark_facet",
    "benchmark_theme",
    "build_layer_c",
]
