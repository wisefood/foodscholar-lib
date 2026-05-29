"""Semantic shelf consolidation — embedding + LLM-as-judge merge pass.

Public entry point is `consolidate()`. It runs as a standalone phase after
`fs.attach()` (see `FoodScholar.semantic_consolidate`):

  embed → candidate pairs → cluster (connected components) → judge each cluster
  in one LLM call → enforce the block-list → apply confirmed N-way merges.

Returns the new shelf list plus a `ConsolidationArtifact` audit log. See
CONSOLIDATION.md.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from foodscholar.layer_a.semantic_consolidation.apply import (
    apply_groups,
    bucket_groups,
)
from foodscholar.layer_a.semantic_consolidation.candidates import find_candidates
from foodscholar.layer_a.semantic_consolidation.cluster import cluster_candidates
from foodscholar.layer_a.semantic_consolidation.embed import embed_shelves
from foodscholar.layer_a.semantic_consolidation.judge import judge_clusters
from foodscholar.layer_a.semantic_consolidation.models import (
    CandidatePair,
    ClusterDecision,
    ConsolidationArtifact,
    MergeGroup,
    ShelfEmbedding,
)
from foodscholar.logging import get_logger
from foodscholar.versioning import new_artifact_id

if TYPE_CHECKING:
    from foodscholar.config import SemanticConsolidationConfig
    from foodscholar.io.graph import Shelf
    from foodscholar.ontology import FoodOnAPI
    from foodscholar.storage.protocols import ChunkStore, Embedder, LLMClient

_log = get_logger("foodscholar.semantic_consolidation")

__all__ = [
    "CandidatePair",
    "ClusterDecision",
    "ConsolidationArtifact",
    "MergeGroup",
    "ShelfEmbedding",
    "consolidate",
]


def consolidate(
    shelves: list[Shelf],
    chunk_store: ChunkStore,
    ontology: FoodOnAPI,
    embedder: Embedder,
    llm: LLMClient,
    cfg: SemanticConsolidationConfig,
    config_hash: str,
    *,
    facet: str = "foods",
    dry_run: bool = False,
) -> tuple[list[Shelf], ConsolidationArtifact]:
    """Embed, cluster candidates, judge, and (unless `dry_run`) apply merges.

    Returns ``(new_shelves, artifact)``. With `dry_run` the shelves come back
    unchanged but the artifact still carries every decision for inspection.
    With `cfg.judge_enabled` False the judge is skipped (no LLM calls) and
    nothing is applied.
    """
    started_at = datetime.now(UTC).isoformat()

    embeddings = embed_shelves(shelves, ontology, embedder, cfg)
    by_id = {s.shelf_id: s for s in shelves}
    candidates, filtered = find_candidates(embeddings, by_id, cfg)
    clusters = cluster_candidates(candidates, cfg.max_cluster_size)

    decisions: list[ClusterDecision] = []
    applied: list[MergeGroup] = []
    uncertain: list[MergeGroup] = []
    blocked: list[MergeGroup] = []

    if cfg.judge_enabled and clusters:
        decisions = judge_clusters(
            clusters, by_id, ontology, chunk_store, llm, cfg
        )
        applied, uncertain, blocked = bucket_groups(decisions, by_id, cfg)

    new_shelves = (
        shelves if dry_run else apply_groups(shelves, applied, cfg)
    )

    artifact = ConsolidationArtifact(
        run_id=new_artifact_id("semantic-consolidation"),
        config_hash=config_hash,
        facet=facet,
        embedder_id=embedder.model_id,
        llm_id=llm.model_id,
        candidate_count=len(candidates),
        cluster_count=len(clusters),
        cluster_decisions=decisions,
        applied_groups=applied,
        uncertain_groups=uncertain,
        blocked_groups=blocked,
        filtered_pairs=filtered,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
    )
    _log.info(
        "semantic_consolidation.done",
        facet=facet,
        candidates=len(candidates),
        clusters=len(clusters),
        applied_groups=len(applied),
        shelves_removed=artifact.shelves_removed,
        uncertain=len(uncertain),
        blocked=len(blocked),
        dry_run=dry_run,
    )
    return new_shelves, artifact
