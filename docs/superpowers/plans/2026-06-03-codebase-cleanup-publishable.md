# Codebase cleanup → testable & publishable — Plan

> **For agentic workers:** this is a staged *cleanup/consolidation* plan, not a feature TDD plan.
> The gate after every stage is the same: the full unit suite must pass **in the `foodscholar`
> conda env** (see Stage 0). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring branch `layer-a-method-bakeoff` to a state that a fresh clone can install,
test, and run — then publish (merge to `main`). Remove the method-bake-off scaffolding now
that the **1a+ backbone projection + aliasing** method is the decided production path.

**Why now:** the chosen Layer A method (`backbone.py`, `alias.py`) is *untracked* — a clone or
CI would fail to build Layer A. The repo also carries a parallel bake-off harness, a superseded
tree renderer, duplicate build notebooks, and one real test failure. This plan resolves all of it.

---

## Current state (measured 2026-06-03)

**Test health** — `foodscholar` env (Python 3.11, numpy 2.4.4): **545 passed, 1 failed, 1 skipped**.
The same suite in `base` (Python 3.13, numpy 1.26.4) is uninstallable — numpy/igraph/leidenalg
fail to import. **Tests must run in the `foodscholar` env.**

**The one failure:** `tests/unit/test_layer_b_label.py::test_label_by_keywords_filters_ocr_codes_and_id_leakage`
— `label_by_keywords` lets a 2-char token (`mg`) through; the filter must drop tokens < 3 chars.
Real bug in live Layer B code.

**Branch:** `layer-a-method-bakeoff` is **32 commits ahead of `main`, 0 behind**, pushed to origin.

**Untracked — LIVE production code (must be committed):**
- `src/foodscholar/layer_a/backbone.py` — the chosen 1a+ backbone method (dispatched from
  `builder.py:147` when `config.projection == "backbone"`, the default).
- `src/foodscholar/layer_a/alias.py` — the LLM aliasing pass (`build_layer_a` calls it).
- `tests/unit/test_layer_a_backbone.py`, `tests/unit/test_layer_a_alias.py` — their tests.
- `scripts/make_annotated_parquet.py` — builds `data/annotated.parquet`; referenced by the
  offline path of `graph_build.ipynb`.
- `scripts/run_layer_b_inmemory.py` — infra-free Layer A+B smoke run.

**Untracked — DEAD/superseded (remove):**
- `src/foodscholar/viz/shelf_tree.py` + `tests/unit/test_shelf_tree.py` — superseded by the
  canonical `fs.viz.layer_a_tree().render("tree")` (decided last session). No production refs.
- `src/foodscholar/layer_a/bakeoff/agentic/alias.py` + `tests/unit/test_agentic_alias.py` —
  part of the bake-off harness (see Decision A).

**Modified/deleted — bake-off churn:** `src/foodscholar/layer_a/bakeoff/**` (metrics, result,
agentic/relations; agent.py deleted), `tests/unit/test_agentic_*`, plus tracked edits to
`config.py`, `builder.py`, `storage/elastic.py`, `docker-compose.yaml`, and three notebooks.

**Two build notebooks (duplication):**
- `notebooks/build_graph.ipynb` (tracked) — older; references `shelf_tree`; bottom-up grouping path.
- `notebooks/graph_build.ipynb` (committed this session) — canonical; backbone + `layer_a_tree`.
- `scripts/build_graph_clean_nb.py` *generates* `graph_build.ipynb` but its docstring is **stale**
  (says "projection prune + shelf_tree"; the notebook now uses backbone + layer_a_tree).

---

## Decisions required before execution

These are judgment calls I should not make for you. Each blocks the stage noted.

**Decision A — the bake-off harness (`src/foodscholar/layer_a/bakeoff/`):** delete, or keep as
research provenance? It's how the backbone method was chosen (`metrics.py`, `scorecard.py`,
`result.py`, `agentic/*`), used only by `scripts/build_layer_a_method_bakeoff_nb.py`,
`notebooks/layer_a_method_bakeoff.ipynb`, and `tests/unit/test_bakeoff_*`. Not imported by any
production path. Options: **(A1)** delete it and its tests/notebook/script; **(A2)** move it under
`research/` or `docs/bakeoff/` (kept, but out of the shipped package); **(A3)** keep in place.
*Recommendation: A2 — a publishable research repo benefits from reproducible method-selection
evidence, but it shouldn't sit in the importable package.* Blocks Stage 3.

**Decision B — alternate Layer A paths (`prune.py` fallback + `grouping.py` bottom-up):** the
backbone method is default, but `prune` is still the `projection != "backbone"` fallback and
`grouping` runs when `bottom_up_grouping.enabled`. Keep both as supported alternates, or strip to
backbone-only for a smaller publishable surface? *Recommendation: keep `prune` (cheap, real
fallback, exercised by tests) and keep `grouping` only if a notebook/config still uses it;
otherwise remove. Verify usage in Stage 4.* Blocks Stage 4.

**Decision C — the two build notebooks:** keep both (they demo different methods), or make
`graph_build.ipynb` canonical and remove/retire `build_graph.ipynb`? *Recommendation: make
`graph_build.ipynb` canonical; if `build_graph.ipynb` only differs by the bottom-up method and
shelf_tree, retire it (and its generator references).* Blocks Stage 5.

**Decision D — publish target:** merge `layer-a-method-bakeoff` → `main` via PR at the end, or
keep iterating on the branch? *Recommendation: open a PR to `main` once Stages 0–6 are green.*
Blocks Stage 7.

---

## Stage 0 — Lock the test environment (gate definition)

- [ ] **0.1** Confirm the gate command works:
  `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit -q`
  Expected today: `1 failed, 545 passed, 1 skipped`.
- [ ] **0.2** Record the env requirement where contributors will see it. Add to `README` (or
  `CONTRIBUTING.md`): "Run tests in the `foodscholar` conda env (Python 3.11). The `base` env's
  numpy is incompatible with 3.13." If a `pyproject.toml`/`environment.yml` pins deps, verify it
  declares `numpy>=2`, `python-igraph`, `leidenalg`, `hnswlib`, `sentence-transformers` under the
  right extras.
- [ ] **0.3** Commit: `chore(test): document foodscholar env as the test gate`.

## Stage 1 — Fix the one real failure (TDD: test already exists and fails)

**Files:** `src/foodscholar/layer_b/label.py`, `tests/unit/test_layer_b_label.py` (already failing).

- [ ] **1.1** Run the failing test, confirm it fails on the `< 3 chars` assertion:
  `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest "tests/unit/test_layer_b_label.py::test_label_by_keywords_filters_ocr_codes_and_id_leakage" -v`
- [ ] **1.2** In `label_by_keywords` (`src/foodscholar/layer_b/label.py`), find the token/keyword
  filter and add a minimum-length guard so every whitespace-token of a kept term has `len >= 3`
  (drop terms containing any sub-3-char token, matching the test's intent alongside the existing
  OCR/ID filters). Read the function first; reuse its existing filter predicate rather than adding
  a parallel one.
- [ ] **1.3** Re-run 1.1 → PASS. Then full suite → `546 passed, 1 skipped`.
- [ ] **1.4** Commit: `fix(layer_b): drop sub-3-char tokens from keyword labels`.

## Stage 2 — Commit the untracked LIVE code (publishable blocker #1)

- [ ] **2.1** Sanity-check these are the live path: `grep -n "import.*backbone\|import.*alias"
  src/foodscholar/layer_a/builder.py` shows both imported.
- [ ] **2.2** Run their tests green in the foodscholar env:
  `... -m pytest tests/unit/test_layer_a_backbone.py tests/unit/test_layer_a_alias.py -q`.
- [ ] **2.3** `git add` the live files + tests:
  `src/foodscholar/layer_a/backbone.py src/foodscholar/layer_a/alias.py
   tests/unit/test_layer_a_backbone.py tests/unit/test_layer_a_alias.py
   scripts/make_annotated_parquet.py scripts/run_layer_b_inmemory.py`
- [ ] **2.4** Commit: `feat(layer_a): track backbone projection + aliasing (the production method)`.
- [ ] **2.5** Verify a clean checkout builds: in a throwaway worktree of HEAD,
  `... -c "from foodscholar.layer_a.backbone import build_backbone_shelves; from foodscholar.layer_a.alias import alias_shelves; print('importable')"`.

## Stage 3 — Remove the superseded tree + (per Decision A) the bake-off harness

- [ ] **3.1** Remove the superseded inline tree (decided last session):
  `git rm src/foodscholar/viz/shelf_tree.py tests/unit/test_shelf_tree.py` (use `rm` for the
  untracked ones). Confirm zero references first:
  `grep -rn "shelf_tree\|write_shelf_tree_html" src notebooks scripts tests` → only the files
  being removed (and the already-updated `graph_build.ipynb` should show none).
- [ ] **3.2** Per **Decision A**, either:
  - **A1 delete:** `git rm -r src/foodscholar/layer_a/bakeoff tests/unit/test_bakeoff_*.py
    tests/unit/test_agentic_*.py scripts/build_layer_a_method_bakeoff_nb.py`;
    `git rm notebooks/layer_a_method_bakeoff.ipynb`. Remove `alias.py`/`test_agentic_alias.py`
    (untracked) with `rm`.
  - **A2 archive:** `git mv src/foodscholar/layer_a/bakeoff research/bakeoff` and move its tests +
    notebook + script under `research/`; update imports in those tests to the new path; ensure
    `research/` is excluded from the package build (`pyproject` packages/`find` config).
  - **A3 keep:** no action; ensure the bake-off tests pass in the foodscholar env.
- [ ] **3.3** Full suite green (count drops by the removed bake-off/shelf_tree tests under A1).
- [ ] **3.4** Commit: `chore(layer_a): remove method bake-off harness + superseded shelf_tree`
  (or `refactor: move bake-off to research/` for A2).

## Stage 4 — Resolve alternate Layer A paths (per Decision B)

- [ ] **4.1** Map live usage: `grep -rn "from foodscholar.layer_a.prune\|grouping\|projection\|
  bottom_up_grouping" src notebooks scripts`.
- [ ] **4.2** If keeping `prune` (recommended): leave as-is — it's the `projection != "backbone"`
  fallback and tested. If removing `grouping`: confirm no notebook/config sets
  `bottom_up_grouping.enabled = True` (note `build_graph.ipynb` does — resolve under Decision C
  first), then `git rm src/foodscholar/layer_a/grouping.py` and its config/tests, and drop the
  `BottomUpGroupingConfig` block from `config.py`.
- [ ] **4.3** Full suite green. Commit: `refactor(layer_a): <kept/removed> alternate construction paths`.

## Stage 5 — Consolidate build notebooks (per Decision C)

- [ ] **5.1** If retiring `build_graph.ipynb`: `git rm notebooks/build_graph.ipynb`; update any
  README/docs links to point at `graph_build.ipynb`.
- [ ] **5.2** Refresh the generator: update `scripts/build_graph_clean_nb.py`'s stale docstring
  (and any prune/shelf_tree cell-emitting code) so re-running it reproduces the *current*
  `graph_build.ipynb` (backbone + `fs.viz.layer_a_tree`). Then run it and `git diff` the notebook
  to confirm no unintended drift.
- [ ] **5.3** Commit: `docs(nb): single canonical graph-build notebook + matching generator`.

## Stage 6 — Final hygiene + full-clone verification

- [ ] **6.1** `ruff check src tests scripts` → clean (fix inline).
- [ ] **6.2** Confirm no lingering references to removed modules:
  `grep -rn "bakeoff\|shelf_tree\|agentic" src` → none (or only `research/` under A2).
- [ ] **6.3** Stale-data check: regenerate the demo artifact end-to-end against live stores
  (`python scripts/build_layer_a_tree.py`) and confirm `themes by pass` shows non-zero `merged`.
- [ ] **6.4** Throwaway-worktree clone test: from a fresh `git worktree` of HEAD, install in the
  foodscholar env and run `pytest tests/unit -q` → all green. This proves "fresh clone is testable."
- [ ] **6.5** Commit any doc/README updates: `docs: refresh README for publishable state`.

## Stage 7 — Publish (per Decision D)

- [ ] **7.1** Push branch: `git push origin layer-a-method-bakeoff`.
- [ ] **7.2** Open PR `layer-a-method-bakeoff → main` via `gh pr create`, body summarizing: backbone
  method now tracked, bake-off harness <removed/archived>, tree consolidated, 546 tests green in
  the foodscholar env. (Confirm with you before creating — outward-facing action.)
- [ ] **7.3** After review/merge, delete the branch if desired.

---

## Coverage check (self-review)

- Publishable blocker (untracked live method) → Stage 2. ✓
- Testable blocker (wrong env) → Stage 0; real failure → Stage 1. ✓
- DRY (two trees) → Stage 3; (two notebooks) → Stage 5. ✓
- Dead code (bake-off) → Stage 3 / Decision A. ✓
- Alternate-path surface → Stage 4 / Decision B. ✓
- Publish → Stage 7 / Decision D. ✓
- Every stage gated by the foodscholar-env suite; clone-from-scratch proven in 6.4.
