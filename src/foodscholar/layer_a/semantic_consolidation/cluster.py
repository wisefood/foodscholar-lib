"""Group candidate pairs into clusters for batch judging.

A candidate pair is a weighted edge (weight = cosine similarity); the connected
components of that graph are the clusters. Judging one cluster per LLM call lets
the model resolve transitive structure itself — a 5-shelf cereal cluster that
would otherwise fragment into ~7 pairwise verdicts becomes one clean decision.

But at a loose threshold the graph chains transitively into a single hairball
(A~B~C~…) spanning most of the facet — too big to judge in one call and enough
to blow the model's JSON budget. So any component larger than `max_cluster_size`
is split: the weakest edges are dropped one at a time until every piece fits.
This breaks a hairball into its dense cores (the genuinely tight groups) while
shedding the marginal edges that merely bridged them.

Singletons (shelves left with no surviving edge) are not clusters and aren't
judged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from foodscholar.layer_a.semantic_consolidation.models import CandidatePair


def cluster_candidates(
    candidates: list[CandidatePair], max_cluster_size: int = 12
) -> list[list[str]]:
    """Connected components of the candidate graph, each capped at
    `max_cluster_size`.

    Returns a list of clusters, each a sorted list of shelf ids (sorted for
    deterministic prompts → reproducible runs). Only components with at least
    one surviving edge are returned.
    """
    graph: nx.Graph = nx.Graph()
    for pair in candidates:
        graph.add_edge(pair.shelf_a, pair.shelf_b, weight=pair.cosine_similarity)

    clusters: list[list[str]] = []
    for component in nx.connected_components(graph):
        sub = graph.subgraph(component).copy()
        clusters.extend(_split_to_cap(sub, max_cluster_size))
    return sorted(clusters)


def _split_to_cap(graph: nx.Graph, cap: int) -> list[list[str]]:
    """Split one component until every piece has at most `cap` nodes.

    Repeatedly removes the lowest-weight edge from any oversized component and
    re-evaluates connectivity. Pieces that fit are emitted; singletons (a node
    isolated by edge removal) are dropped — a shelf with no surviving similar
    neighbour is no longer a merge candidate.
    """
    if cap < 2:  # degenerate; nothing can be judged as a group
        return []
    out: list[list[str]] = []
    queue: list[nx.Graph] = [graph]
    while queue:
        g = queue.pop()
        nodes = list(g.nodes)
        if len(nodes) <= cap:
            if len(nodes) >= 2:
                out.append(sorted(nodes))
            continue
        # Drop the weakest edge, then re-split the resulting components.
        u, v, _ = min(g.edges(data="weight"), key=lambda e: e[2])
        g.remove_edge(u, v)
        for comp in nx.connected_components(g):
            queue.append(g.subgraph(comp).copy())
    return out
