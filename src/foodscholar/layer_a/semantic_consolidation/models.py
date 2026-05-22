"""Pydantic data contracts for the semantic consolidation pass.

Mirrors CONSOLIDATION.md §4, adapted to the real `Shelf` (which carries no
`definition`/`synonyms`/`chunk_ids` — those are fetched from the ontology and
chunk store on demand).

Judging is **cluster-based**, not pairwise: candidate pairs are grouped into
connected components, and the judge sees a whole cluster in one call. It
returns `merge_groups` (each a set of members that collapse to one shelf) and
`keep_alone` (members that stay distinct). This lets the model resolve
transitive structure itself ("A~B~C are all the same; D is different") instead
of us reconstructing it from fragmented pairwise verdicts.

Every run produces one `ConsolidationArtifact`: the full audit log.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

ShelfId = str


class ShelfEmbedding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shelf_id: ShelfId
    foodon_id: str  # never None — synthetic facet roots are excluded upstream
    text: str  # the concatenated "label | syn1 | syn2" that got embedded
    embedding: list[float]
    embedder_id: str


class CandidatePair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shelf_a: ShelfId
    shelf_b: ShelfId
    cosine_similarity: float
    filtered_reason: str | None = None
    """If set, a pre-LLM filter excluded this pair (the reason). None means it
    survives to the judge."""


class MergeGroup(BaseModel):
    """A set of shelves the judge says collapse into one."""

    model_config = ConfigDict(extra="forbid")
    members: list[ShelfId]
    canonical_name: str  # the judge's suggested label for the merged shelf
    confidence: float  # 0..1
    rationale: str
    blocked_pairs: list[tuple[ShelfId, ShelfId]] = []
    """Member pairs vetoed by the permanent block-list. If non-empty the group
    was split before application (see apply.py)."""


class ClusterDecision(BaseModel):
    """The judge's full verdict on one candidate cluster (one LLM call)."""

    model_config = ConfigDict(extra="forbid")
    cluster_members: list[ShelfId]
    merge_groups: list[MergeGroup]
    keep_alone: list[ShelfId]
    llm_id: str
    prompt_version: str
    decided_at: str  # ISO timestamp


class ConsolidationArtifact(BaseModel):
    """The full audit log of one consolidation run."""

    model_config = ConfigDict(extra="forbid")
    run_id: str
    config_hash: str
    facet: str
    embedder_id: str
    llm_id: str
    candidate_count: int  # pairs surviving the pre-LLM filters
    cluster_count: int  # connected components judged
    cluster_decisions: list[ClusterDecision]
    applied_groups: list[MergeGroup]  # confirmed (conf >= threshold, not blocked)
    uncertain_groups: list[MergeGroup]  # below confidence → not applied
    blocked_groups: list[MergeGroup]  # vetoed by the permanent block-list
    filtered_pairs: list[CandidatePair]  # excluded pre-LLM, kept for audit
    started_at: str
    finished_at: str

    @property
    def shelves_removed(self) -> int:
        """How many shelves the applied groups fold away (members minus one
        canonical per group)."""
        return sum(max(0, len(g.members) - 1) for g in self.applied_groups)

    def __str__(self) -> str:  # human-readable summary for notebook printing
        lines = [
            f"SemanticConsolidation[{self.facet}] run={self.run_id}",
            f"  embedder={self.embedder_id}  llm={self.llm_id}",
            f"  candidates={self.candidate_count}  "
            f"clusters={self.cluster_count}  "
            f"filtered_pre_llm={len(self.filtered_pairs)}",
            f"  applied_groups={len(self.applied_groups)} "
            f"(-{self.shelves_removed} shelves)  "
            f"uncertain={len(self.uncertain_groups)}  "
            f"blocked={len(self.blocked_groups)}",
        ]
        for g in self.applied_groups:
            members = " + ".join(g.members)
            lines.append(
                f"    MERGE [{members}] -> {g.canonical_name!r} "
                f"(conf={g.confidence:.2f}): {g.rationale}"
            )
        return "\n".join(lines)
