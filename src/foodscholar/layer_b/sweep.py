"""Layer B tuning sweep + config scoring.

`sweep_layer_b(fs, *, facet, grid=None)` runs a Cartesian grid of Layer B
configs as *non-mutating* `dry_run` builds, computes the quality metrics for
each from the build's `themes_preview`, scores each config, and returns a
ranked `SweepResult`. Every build forces cheap keyword labels and the
production `per_shelf` Pass 1, so a sweep is fast and deterministic and never
writes the store. The user applies the winning config separately.

Scoring uses fixed, documented weights (module constants) that encode the
brief's preference order: maximize coverage and useful merged themes; minimize
duplicate labels, tiny noisy themes, and cross-shelf leakage; keep theme count
manageable.

See docs/superpowers/specs/2026-06-05-layer-b-quality-tuning-design.md §4.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from foodscholar.layer_b.models import LayerBQualityReport
from foodscholar.layer_b.quality import compute_quality_metrics

if TYPE_CHECKING:
    from foodscholar.facade import FoodScholar

# --- default Cartesian grid (160 configs) -----------------------------------
DEFAULT_GRID: dict[str, list[Any]] = {
    "leiden.min_community_size": [5, 8, 10, 15],
    "leiden.resolution": [0.6, 0.8, 1.0, 1.2, 1.5],
    "similarity.edge_threshold": [0.40, 0.45, 0.50, 0.55],
    "similarity.require_mutual": [True, False],
}

# --- scoring weights (documented, fixed for v1) -----------------------------
# Each component is normalized to [0, 1] before weighting; the rewards are
# added and the penalties subtracted. Weights reflect the brief's priority:
# coverage and useful merges are the goal, noise is the cost.
W_COVERAGE = 1.0          # + themed fraction of attached chunks
W_MERGED = 0.5            # + merged themes / total
W_DUP_LABELS = 0.5        # - duplicate-label themes / total
W_TINY = 0.5              # - tiny themes / total
W_LEAKAGE = 1.0          # - cross-shelf-leakage themes / total
W_COUNT_OVERAGE = 0.3     # - theme-count overage past the target band


def _set_path(obj: Any, dotted: str, value: Any) -> None:
    """Set a dotted attribute path on a (mutable) pydantic model in place."""
    *parents, leaf = dotted.split(".")
    for p in parents:
        obj = getattr(obj, p)
    setattr(obj, leaf, value)


def _theme_count_band(report: LayerBQualityReport, cfg: Any) -> tuple[int, int]:
    """Acceptable total-theme band, derived from the per-shelf targets scaled
    by the facet's shelf count. A config whose total theme count falls outside
    this band is penalized in `score_report`."""
    lo = cfg.audit.target_themes_per_shelf_min
    hi = cfg.audit.target_themes_per_shelf_max * max(report.n_shelves, 1)
    return lo, max(hi, lo)


def score_report(report: LayerBQualityReport, cfg: Any) -> float:
    """Weighted score for one config's quality report. Higher is better."""
    n = report.n_themes or 1

    coverage = report.theme_coverage
    merged_frac = report.n_merged / n
    dup_rate = report.n_duplicate_label_themes / n
    tiny_rate = report.n_tiny_themes / n
    leak_rate = report.n_cross_shelf_leakage / n

    lo, hi = _theme_count_band(report, cfg)
    if report.n_themes < lo:
        overage = (lo - report.n_themes) / lo
    elif report.n_themes > hi:
        overage = (report.n_themes - hi) / hi
    else:
        overage = 0.0
    overage = min(1.0, overage)

    return (
        W_COVERAGE * coverage
        + W_MERGED * merged_frac
        - W_DUP_LABELS * dup_rate
        - W_TINY * tiny_rate
        - W_LEAKAGE * leak_rate
        - W_COUNT_OVERAGE * overage
    )


class SweepRow(BaseModel):
    """One config's result in a sweep."""

    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any]
    score: float
    report: LayerBQualityReport


class SweepResult(BaseModel):
    """Ranked result of a Layer B tuning sweep (best first)."""

    model_config = ConfigDict(extra="forbid")

    facet: str
    rows: list[SweepRow] = Field(default_factory=list)

    @property
    def best(self) -> dict[str, Any] | None:
        return self.rows[0].config if self.rows else None

    def to_frame(self):  # type: ignore[no-untyped-def]
        """Flatten to a pandas DataFrame (one row per config). Lazy import so
        pandas is only required when this is called."""
        import pandas as pd

        records = []
        for r in self.rows:
            rec: dict[str, Any] = {**r.config, "score": r.score}
            rep = r.report
            rec.update(
                coverage=rep.theme_coverage,
                n_themes=rep.n_themes,
                merged=rep.n_merged,
                similarity_only=rep.n_similarity_only,
                relatedness_only=rep.n_relatedness_only,
                dup_labels=rep.n_duplicate_label_themes,
                tiny=rep.n_tiny_themes,
                leakage=rep.n_cross_shelf_leakage,
            )
            records.append(rec)
        return pd.DataFrame.from_records(records)

    def __str__(self) -> str:
        lines = [f"# Layer B sweep — facet `{self.facet}` ({len(self.rows)} configs)\n"]
        if not self.rows:
            lines.append("_no configs evaluated_")
            return "\n".join(lines)
        keys = list(self.rows[0].config.keys())
        header = ["rank", "score", "cov", "themes", "merged", "dup", "tiny", "leak", *keys]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for i, r in enumerate(self.rows, 1):
            rep = r.report
            cells = [
                str(i),
                f"{r.score:.3f}",
                f"{rep.theme_coverage:.0%}",
                str(rep.n_themes),
                str(rep.n_merged),
                str(rep.n_duplicate_label_themes),
                str(rep.n_tiny_themes),
                str(rep.n_cross_shelf_leakage),
                *[str(r.config[k]) for k in keys],
            ]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)


def sweep_layer_b(
    fs: FoodScholar,
    *,
    facet: str = "foods",
    grid: dict[str, list[Any]] | None = None,
) -> SweepResult:
    """Run a non-mutating Cartesian sweep over `grid` and return ranked results.

    `grid` maps dotted `layer_b` config paths to candidate values; the default
    is the full 160-config Cartesian product (see `DEFAULT_GRID`). Each combo
    runs `fs.build_layer_b(dry_run=True)` with keyword labels and `per_shelf`
    Pass 1, scores the resulting metrics, and is recorded as a `SweepRow`.
    Nothing is persisted.
    """
    from foodscholar.layer_b.builder import build_layer_b as _build

    grid = grid or DEFAULT_GRID
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]

    synth_root = f"facet:{facet}"
    shelves = [
        s
        for s in fs.graph_store.list_shelves()
        if s.facet == facet and s.shelf_id != synth_root
    ]
    facet_shelf_ids = {s.shelf_id for s in shelves}
    attachments = fs.graph_store.list_chunk_shelf_attachments()

    original = fs.config.layer_b
    rows: list[SweepRow] = []
    try:
        for combo in itertools.product(*value_lists):
            config = dict(zip(keys, combo, strict=True))
            trial = original.model_copy(deep=True)
            trial.pass1_mode = "per_shelf"
            trial.labeling.strategy = "keyword"
            for path, value in config.items():
                _set_path(trial, path, value)
            fs.config.layer_b = trial

            art = _build(fs, facet=facet, dry_run=True)
            themes = [
                t
                for t in (art.themes_preview or [])
                if any(sid in facet_shelf_ids for sid in t.shelf_ids)
            ]
            theme_chunks = art.theme_chunk_ids_preview or {}
            themed_chunk_ids = {
                cid for t in themes for cid in theme_chunks.get(t.theme_id, [])
            }
            report = compute_quality_metrics(
                shelves, themes, attachments, themed_chunk_ids, trial, facet
            )
            rows.append(
                SweepRow(config=config, score=score_report(report, trial), report=report)
            )
    finally:
        fs.config.layer_b = original

    rows.sort(key=lambda r: r.score, reverse=True)
    return SweepResult(facet=facet, rows=rows)
