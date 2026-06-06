# Layer B quality, warnings, and tuning — design

**Date:** 2026-06-05
**Status:** approved (brainstorming) — pending spec review
**Goal:** make FoodScholar's FoodOn-based graph more intuitive, less noisy, and
easier to tune via five focused interventions on the Layer B subsystem.

---

## 1. Background

Layer B discovers per-shelf "themes" (Leiden communities) over chunks attached
to FoodOn shelves. Two passes run — Pass 1 (embedding similarity) and Pass 2
(entity relatedness) — and a merge step pairs them. Pass 1 can scope its kNN
graph globally (one graph over the whole facet) or per-shelf.

Today the config defaults Pass 1 to `"global"`, but the docs already describe
`"per_shelf"` as the production default and several of them set it manually to
compensate. Build output is also hard to reason about: there is a CRITICAL
invariant audit (`audit_layer_b`) but no WARN-level *quality* report, no
warnings for common smells, and no tuning sweep. This design closes those gaps.

Relevant existing code:
- `src/foodscholar/config.py` — `LayerBConfig`, `LeidenConfig`,
  `SimilarityConfig`, `LayerBAuditConfig`.
- `src/foodscholar/layer_b/builder.py` — `build_layer_b()` orchestrator.
- `src/foodscholar/layer_b/audit.py` + `models.py` — CRITICAL invariant audit.
- `src/foodscholar/io/graph.py` — `Shelf` (has `depth`, `parent_shelf_id`,
  `support_direct`, `support_lifted`, `chunk_count`) and `Theme` (has
  `shelf_ids`, `chunk_count`, `discovery_pass`, `label`, `keyword_terms`,
  `foodon_id_signature`).
- `src/foodscholar/facade.py` — `fs.audit()`, `fs.quality_report()`,
  `fs.build_layer_b()` (the patterns the new surfaces mirror).
- `src/foodscholar/cli/main.py` — thin CLI over the facade.

**Design principle:** the quality report and sweep are *read-only / non-mutating*
tuning aids, kept cleanly separate from `audit_layer_b()`, which alone gates
CRITICAL build invariants.

---

## 2. Intervention 1 — make `per_shelf` the default

**Change:** in `LayerBConfig` (`config.py`), flip `pass1_mode` default from
`"global"` to `"per_shelf"`. Rewrite the docstring so `per_shelf` is the
production default (single-shelf themes, no cross-shelf smear) and `global` is
described as experimental — used only to discover cross-shelf bridges.

**Ripple:**
- `build_layer_b()` already branches on `pass1_mode`; **no logic change**.
- `global_similarity_max_chunks` is now dormant by default (it only guards
  global mode). Note this in its docstring; keep the field.
- Docs that set `cfg.pass1_mode = "per_shelf"` *only to compensate* for the old
  default — remove that override. Keep it where it is illustrative of the knob.
  Audit: `docs/concepts/layer-b-themes.md`, `docs/guides/tuning-layer-b.md`,
  `docs/guides/visualization.md`, and any plan/script that toggles it.
- Test: assert the new default; scan Layer B tests for any implicitly relying
  on `global` and fix.

---

## 3. Intervention 2 + 3 — build-quality report and warnings

### 3.1 Surfaces

- **New module** `src/foodscholar/layer_b/quality.py`:
  `build_quality_report(chunk_store, graph_store, cfg, *, facet="foods") ->
  LayerBQualityReport`. Read-only; reads shelves, themes, and attachments.
- **New Pydantic models** in `src/foodscholar/layer_b/models.py`:
  `LayerBQualityReport` (metrics + `warnings: list[LayerBWarning]`) and
  `LayerBWarning` (`kind: str`, `shelf_id: str | None`, `theme_id: str | None`,
  `message: str`). `LayerBQualityReport.__str__` renders Markdown for notebook
  viewing (mirrors `QualityReport` / `LayerBAuditReport`).
- **Facade:** `fs.build_quality_report(facet="foods") -> LayerBQualityReport`.
- **CLI:** `foodscholar report-layer-b -c config.yaml` prints the report.

### 3.2 Metrics (all derivable from existing models)

Shelf structure (from `list_shelves()` filtered to `facet`, non-synth):
- number of shelves
- max depth, median depth (`Shelf.depth`)
- max fanout (max children count via `parent_shelf_id`)
- shelves with zero direct support (`support_direct == 0`)
- direct-vs-lifted support ratio (sum `support_direct` / sum `support_lifted`,
  guarded for zero denominator)
- chunks-per-shelf distribution (min / median / max of `Shelf.chunk_count`)

Theme stats (from `list_themes()` filtered to `facet` + attachments):
- theme coverage % — attached chunks with ≥1 `theme_id` over total attached
- themes by source — `merged`, `similarity_only` (:= `discovery_pass ==
  "global_similarity"`), `relatedness_only` (:= `discovery_pass ==
  "relatedness"`)
- duplicate labels — count of (shelf, label) groups with >1 theme sharing an
  identical lowercased label
- tiny themes — themes with `chunk_count < leiden.min_community_size`
- orphan chunks — attached chunks with zero `theme_ids`
- cross-shelf leakage — themes with `len(shelf_ids) > 1` (≈0 in per_shelf mode)

### 3.3 Warnings

A `LayerBWarning` is appended for each condition below. Thresholds come from an
**extended `LayerBAuditConfig`** (documented defaults, YAML-overridable):

| Warning kind | Condition | Threshold field (default) |
|---|---|---|
| `high_lifted_low_direct` | shelf with `support_lifted` high but `support_direct` low | `lifted_to_direct_ratio_max` (e.g. 4.0) applied when `support_direct < direct_support_floor` (e.g. 3) |
| `shelf_no_themes` | facet shelf eligible by `min_chunks_per_shelf` but produced 0 themes | — (uses `min_chunks_per_shelf`) |
| `mostly_single_pass` | within a shelf, fraction of `similarity_only` OR `relatedness_only` themes exceeds a cap | `single_pass_share_max` (e.g. 0.90) |
| `near_duplicate_labels` | two themes in a shelf with token-set Jaccard ≥ threshold on lowercased labels | `dup_label_jaccard_min` (e.g. 0.80) |
| `theme_spans_many_entities` | theme `len(foodon_id_signature)` over a cap | `max_entity_span` (e.g. 8) |
| `theme_label_equals_parent` | theme label equals (case-insensitive) its parent shelf's label/display_label | — |

Near-duplicate detection uses **token-set Jaccard** on lowercased,
whitespace-split labels — no new dependency.

### 3.4 Config additions

Add to `LayerBAuditConfig`: `lifted_to_direct_ratio_max`,
`direct_support_floor`, `single_pass_share_max`, `dup_label_jaccard_min`,
`max_entity_span`. Each with a documented default and a docstring line. None
flip any CRITICAL `passed` flag — these are tuning signal only.

---

## 4. Intervention 4 + 5 — sweep and scoring

### 4.1 Builder dry-run change (prerequisite, approved)

`build_layer_b(dry_run=True)` currently returns only a `LayerBArtifact`
(counts). The sweep needs per-theme detail (chunk membership, shelf_ids,
discovery_pass, label) to score a config *without persisting*. We will **surface
the computed themes from the dry-run path** so a scorer can read them from the
return value rather than re-reading the store.

Implementation: `build_layer_b()` returns the `LayerBArtifact` as today; the
dry-run-computed `themes: list[Theme]` and the attachment/coverage inputs are
attached to the returned object (a new optional field on `LayerBArtifact`, e.g.
`themes_preview: list[Theme] | None`, populated only on `dry_run=True`). The
persisted (non-dry-run) path leaves it `None`. This keeps the public return type
stable and avoids a second store round-trip during sweeps.

### 4.2 Sweep

**New module** `src/foodscholar/layer_b/sweep.py`:
`sweep_layer_b(fs, *, facet="foods", grid=None) -> SweepResult`.

- Default grid = **full Cartesian** (160 configs):
  - `leiden.min_community_size ∈ {5, 8, 10, 15}`
  - `leiden.resolution ∈ {0.6, 0.8, 1.0, 1.2, 1.5}`
  - `similarity.edge_threshold ∈ {0.40, 0.45, 0.50, 0.55}`
  - `similarity.require_mutual ∈ {true, false}`
- `grid=` overrides the default (caller may shrink to e.g. the brief's 3 named
  configs).
- Each config: deep-copy `fs.config.layer_b`, force `labeling.strategy =
  "keyword"` (cheap, deterministic) and `pass1_mode = "per_shelf"`, run
  `build_layer_b(dry_run=True)`, compute metrics from `themes_preview`, compute
  the score, record a row `{config, metrics, score}`.
- Non-mutating: every build is `dry_run=True`; the store is never written. The
  user applies the winning config separately.

### 4.3 Scoring (fixed, documented weights)

`SweepResult` holds rows; `.best` returns the top-scoring config dict;
`.to_frame()` returns a DataFrame; `__str__` prints a ranked Markdown table
(best first).

Score = weighted sum of normalized [0,1] components (weights are module
constants, documented):
- **+ coverage** — themed fraction of attached chunks (maximize)
- **+ useful merged fraction** — merged themes / total themes (maximize)
- **− duplicate-label rate** — dup-label themes / total (minimize)
- **− tiny-theme rate** — tiny themes / total (minimize)
- **− cross-shelf leakage rate** — leaked themes / total (minimize)
- **− theme-count overage** — penalty for theme count outside a target band
  (derived from `target_themes_per_shelf_*` × themed shelves)

Each component is normalized before weighting so the score is comparable across
configs. The winning config is the one that maximizes coverage and useful merged
themes while minimizing dup/tiny/leakage and keeping theme count manageable —
exactly the brief's preference order.

### 4.4 CLI

`foodscholar sweep-layer-b -c config.yaml` runs the default grid and prints the
ranked table. (Slow: 160 dry-run builds. Documented.)

---

## 5. Tests

- New default: `LayerBConfig().pass1_mode == "per_shelf"`.
- Metrics: each computed against a small in-memory fixture (memory chunk +
  graph stores) with known shelves/themes/attachments.
- Warnings: each `kind` fires on a crafted case and stays silent otherwise.
- Scoring monotonicity: a strictly-better config (higher coverage, fewer
  dup/tiny/leak) scores strictly higher.
- Sweep: runs over a tiny 2×2 grid against the fixture, returns ranked rows,
  `.best` is the top row, never writes the store.
- Builder dry-run: `themes_preview` is populated on `dry_run=True` and `None`
  otherwise; persisted output unchanged.

The project pytest suite is the completion gate.

---

## 6. Docs

- `docs/guides/tuning-layer-b.md` — replace the hand-rolled coverage cell with
  `fs.build_quality_report()` and the `sweep-layer-b` command; keep the knob
  table.
- Note the new `report-layer-b` / `sweep-layer-b` CLI commands wherever Layer B
  commands are listed.
- Remove now-redundant `pass1_mode="per_shelf"` overrides (see §2).

---

## 7. Out of scope (YAGNI)

- Renaming the persisted `discovery_pass` enum to `*_only` (we map names in the
  report instead).
- Config-driven scoring weights (fixed for v1).
- LLM labels during sweeps (keyword only — cheap and deterministic).
- Persisting/auto-applying a swept config (sweep is non-mutating; user applies
  the winner).
