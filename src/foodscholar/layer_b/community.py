"""Community detection runner shared by both Layer B passes.

V1 ships **Leiden only**. HDBSCAN is documented as a fallback in
`layer_b_construction_brief.md` §4.3 but cut from v1 per the implementation
plan: the precomputed-distance hack on the relatedness graph wasn't a valid
metric, and the similarity-pass kNN graph is already a natural fit for
modularity-based Leiden.

`run_leiden(graph, cfg)` takes a weighted igraph and returns a list of
communities (sets of vertex indices) filtered by `min_community_size`. With
a fixed `cfg.random_state` it's deterministic across runs — the load-bearing
guarantee for audit cross-store parity.

leidenalg + python-igraph are lazy-imported (gated by the `[clustering]`
extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import igraph as ig

    from foodscholar.config import LeidenConfig


def run_leiden(
    graph: ig.Graph,
    cfg: LeidenConfig,
) -> list[set[int]]:
    """Return Leiden communities (sets of vertex indices) above
    `cfg.min_community_size`.

    Empty graphs and graphs with no edges return `[]` — leidenalg would
    otherwise emit one singleton per isolated vertex, all of which fail the
    size filter anyway.
    """
    import leidenalg as la

    if graph.vcount() == 0 or graph.ecount() == 0:
        return []
    weights = graph.es["weight"] if "weight" in graph.es.attributes() else None
    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=cfg.resolution,
        n_iterations=cfg.n_iterations,
        seed=cfg.random_state,
    )
    return [set(c) for c in partition if len(c) >= cfg.min_community_size]


# Convenience alias for forward compatibility — when callers want to
# dispatch by algorithm name later, they can call `run(algorithm, graph, cfg)`.
# Today only Leiden is supported.
def run(algorithm: str, graph: Any, cfg: Any) -> list[set[int]]:
    if algorithm == "leiden":
        return run_leiden(graph, cfg)
    raise NotImplementedError(
        f"Layer B algorithm {algorithm!r} is not supported in v1 — "
        f"only 'leiden' ships in this version. See layer_b_plan.md."
    )
