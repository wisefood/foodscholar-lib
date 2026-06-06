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
from foodscholar.io.graph import Facet, Theme

DiscoveryPass = Literal["relatedness", "merged", "global_similarity"]
DiscoveredBy = Literal["leiden", "hdbscan"]


class ThemeCandidate(BaseModel):
    """One community emitted by a single pass.

    Not persisted — fed into `merge.merge_candidates` which decides which
    candidates pair into `discovery_pass="merged"` themes vs. pass through
    as single-pass themes.
    """

    model_config = ConfigDict(extra="forbid")

    pass_name: Literal["relatedness", "global_similarity"]
    chunk_ids: set[ChunkId]
    foodon_ids: set[str] = Field(default_factory=set)
    centroid_embedding: list[float] | None = None
    discovered_by: DiscoveredBy = "leiden"
    origin_shelf_id: str | None = None
    """The shelf this candidate's graph was built from, when the pass ran
    per-shelf. Set for per-shelf Pass 1 (and Pass 2, which is always per-shelf);
    `None` for global Pass 1, where the community spans shelves and has no single
    origin. When set, the theme attaches to exactly this shelf rather than the
    union of its member chunks' shelves (which over-attaches via lifted chunks)."""


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
    themes_preview: list[Theme] | None = None
    """Themes computed this run, surfaced ONLY on `dry_run=True` builds so a
    caller (the tuning sweep / quality scorer) can score a config without
    persisting it or re-reading the store. `None` on persisted (non-dry-run)
    builds — read the live store instead."""
    theme_chunk_ids_preview: dict[str, list[ChunkId]] | None = None
    """theme_id -> member chunk_ids for the dry-run themes (the `Theme` model
    only carries `chunk_count`, not membership). Surfaced alongside
    `themes_preview` so the sweep scorer can compute coverage / orphan counts
    without the persisted `theme_ids` denorm. `None` on non-dry-run builds."""


class LayerBAuditReport(BaseModel):
    """Cross-store audit gates for Layer B (per brief §10).

    `parity` is the agreement rate between Neo4j `THEME_OF` edges and the
    Elastic `theme_ids` denorm. `dangling_edges` counts `theme_ids` on
    chunks that reference a theme_id with no corresponding `(:Theme)` node.
    `empty_themes` counts themes with `chunk_count > 0` but zero attached
    chunks at audit time. `orphan_themes` counts themes with `shelf_ids=[]`
    — these themes are unreachable from any shelf in the UI.
    `passed` is True iff all four CRITICAL gates pass (parity == 1.0, no
    dangling, no empty, no orphan themes).

    The WARN gates (target_themes_per_shelf range, both passes contribute
    something, merged-rate inside band) live on the report as informational
    fields — they don't flip `passed` to False but they're emitted in the
    notebook for tuning.
    """

    model_config = ConfigDict(extra="forbid")

    parity: float = 0.0
    dangling_edges: int = 0
    empty_themes: int = 0
    orphan_themes: int = 0
    """Themes with shelf_ids=[] — unreachable from any shelf in the UI."""
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
            and self.orphan_themes == 0
        )


WarningKind = Literal[
    "high_lifted_low_direct",
    "shelf_no_themes",
    "mostly_single_pass",
    "near_duplicate_labels",
    "theme_spans_many_entities",
    "theme_label_equals_parent",
]


class LayerBWarning(BaseModel):
    """One WARN-level smell surfaced by `build_quality_report()`.

    Purely informational — warnings never gate a build. `shelf_id` / `theme_id`
    locate the offending element (either or both may be None for facet-wide
    warnings).
    """

    model_config = ConfigDict(extra="forbid")

    kind: WarningKind
    message: str
    shelf_id: str | None = None
    theme_id: str | None = None


class LayerBQualityReport(BaseModel):
    """Read-only WARN-level quality + tuning report for one facet of Layer B.

    Pairs with `audit_layer_b()` (CRITICAL invariants) but answers a different
    question: not "is the build correct" but "is the build *good* / well-tuned".
    Every field is derived from the live shelves, themes, and attachments; the
    report mutates nothing. `__str__` renders Markdown for notebook viewing.

    Theme sources map the persisted `discovery_pass` values onto the brief's
    vocabulary: ``similarity_only`` := ``global_similarity``, ``relatedness_only``
    := ``relatedness``, ``merged`` := ``merged``.
    """

    model_config = ConfigDict(extra="forbid")

    facet: Facet

    # --- shelf structure ---
    n_shelves: int = 0
    max_depth: int = 0
    median_depth: float = 0.0
    max_fanout: int = 0
    shelves_zero_direct_support: int = 0
    direct_to_lifted_ratio: float = 0.0
    chunks_per_shelf_min: int = 0
    chunks_per_shelf_median: float = 0.0
    chunks_per_shelf_max: int = 0

    # --- theme stats ---
    n_themes: int = 0
    theme_coverage: float = 0.0
    """Fraction of attached chunks that landed in ≥1 theme."""
    n_merged: int = 0
    n_similarity_only: int = 0
    n_relatedness_only: int = 0
    n_duplicate_label_themes: int = 0
    n_tiny_themes: int = 0
    n_orphan_chunks: int = 0
    n_cross_shelf_leakage: int = 0

    warnings: list[LayerBWarning] = Field(default_factory=list)

    def __str__(self) -> str:
        lines: list[str] = []
        a = lines.append
        a(f"# Layer B quality report — facet `{self.facet}`\n")

        a("## Shelf structure\n")
        a(f"- shelves: **{self.n_shelves}**")
        a(f"- depth: max **{self.max_depth}**, median **{self.median_depth:.1f}**")
        a(f"- max fanout: **{self.max_fanout}**")
        a(f"- shelves with zero direct support: **{self.shelves_zero_direct_support}**")
        a(f"- direct/lifted support ratio: **{self.direct_to_lifted_ratio:.2f}**")
        a(
            f"- chunks per shelf: min **{self.chunks_per_shelf_min}**, "
            f"median **{self.chunks_per_shelf_median:.1f}**, "
            f"max **{self.chunks_per_shelf_max}**\n"
        )

        a("## Themes\n")
        a(f"- themes: **{self.n_themes}**")
        a(f"- coverage: **{self.theme_coverage:.0%}** of attached chunks themed")
        a(
            f"- by source: merged **{self.n_merged}**, "
            f"similarity-only **{self.n_similarity_only}**, "
            f"relatedness-only **{self.n_relatedness_only}**"
        )
        a(f"- duplicate-label themes: **{self.n_duplicate_label_themes}**")
        a(f"- tiny themes: **{self.n_tiny_themes}**")
        a(f"- orphan (un-themed) chunks: **{self.n_orphan_chunks}**")
        a(f"- cross-shelf leakage themes: **{self.n_cross_shelf_leakage}**\n")

        a("## Warnings\n")
        if not self.warnings:
            a("_none_\n")
        else:
            for w in self.warnings:
                loc = " ".join(
                    p for p in (
                        f"shelf=`{w.shelf_id}`" if w.shelf_id else "",
                        f"theme=`{w.theme_id}`" if w.theme_id else "",
                    ) if p
                )
                a(f"- **{w.kind}** {loc} — {w.message}".rstrip())
        return "\n".join(lines)
