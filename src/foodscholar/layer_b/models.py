"""Pydantic models internal to Layer B.

`Theme` itself lives in `foodscholar.io.graph` because it's persisted and
cross-cuts the storage layer. The models here are intermediate (candidate,
merge decision) or run-level (artifact, audit) — they don't go through the
stores.

See layer_b_construction_brief.md §3.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from foodscholar.io.chunk import ChunkId
from foodscholar.io.graph import Facet

DiscoveryPass = Literal["similarity", "relatedness", "merged", "global_similarity"]
DiscoveredBy = Literal["leiden", "hdbscan"]


class ThemeCandidate(BaseModel):
    """One community emitted by a single pass.

    Not persisted — fed into `merge.merge_candidates` which decides which
    candidates pair into `discovery_pass="merged"` themes vs. pass through
    as single-pass themes.
    """

    model_config = ConfigDict(extra="forbid")

    pass_name: Literal["similarity", "relatedness", "global_similarity"]
    chunk_ids: set[ChunkId]
    foodon_ids: set[str] = Field(default_factory=set)
    centroid_embedding: list[float] | None = None
    discovered_by: DiscoveredBy = "leiden"


class MergeDecision(BaseModel):
    """Audit log for one (sim_cand, rel_cand) pairing — kept or discarded.

    Stored in the run artifact so we can answer "why didn't these merge?"
    without re-running the pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    similarity_candidate_idx: int
    relatedness_candidate_idx: int
    chunk_jaccard: float
    entity_jaccard: float
    combined_similarity: float
    merged: bool


class LayerBArtifact(BaseModel):
    """Run-level metadata for one `fs.build_layer_b()` invocation."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    facet: Facet
    config_hash: str
    n_shelves_themed: int = 0
    n_shelves_skipped: int = 0
    n_themes_total: int = 0
    n_themes_by_pass: dict[DiscoveryPass, int] = Field(default_factory=dict)
    leiden_seed: int
    started_at: str
    finished_at: str


class LayerBAuditReport(BaseModel):
    """Cross-store audit gates for Layer B (per brief §10).

    `parity` is the agreement rate between Neo4j `THEME_OF` edges and the
    Elastic `theme_ids` denorm. `dangling_edges` counts `theme_ids` on
    chunks that reference a theme_id with no corresponding `(:Theme)` node.
    `empty_themes` counts themes with `chunk_count > 0` but zero attached
    chunks at audit time. `passed` is True iff all three CRITICAL gates pass
    (parity == 1.0, no dangling, no empty).

    The WARN gates (target_themes_per_shelf range, both passes contribute
    something, merged-rate inside band) live on the report as informational
    fields — they don't flip `passed` to False but they're emitted in the
    notebook for tuning.
    """

    model_config = ConfigDict(extra="forbid")

    parity: float = 0.0
    dangling_edges: int = 0
    empty_themes: int = 0
    n_themes: int = 0
    n_themed_shelves: int = 0
    by_pass: dict[str, int] = Field(default_factory=dict)
    merged_rate: float = 0.0

    @property
    def passed(self) -> bool:
        return (
            self.parity == 1.0
            and self.dangling_edges == 0
            and self.empty_themes == 0
        )
