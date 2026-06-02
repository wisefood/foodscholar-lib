"""Layer-A method bake-off harness: a common MethodResult + pure metrics.

See docs/methods_layer_a_bakeoff_brief.md. Every construction method emits a
MethodResult via an adapter; metrics consume only that struct so methods are
scored on identical footing.
"""

from foodscholar.layer_a.bakeoff.result import MethodResult, node_depths

__all__ = ["MethodResult", "node_depths"]
