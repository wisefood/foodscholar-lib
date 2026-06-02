# Layer-A Method Bake-off Harness (Plan A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing "by-eye" projection bake-off into a metric-driven benchmark: a small importable harness that scores any Layer-A construction method on coverage, findability, nameability, fan-out, depth, faithfulness, and reproducibility, then wire a scorecard over the already-built methods (`0, 1a, 1a+, 1b`, + the merged grouping method) into the bake-off notebook.

**Architecture:** A new `src/foodscholar/layer_a/bakeoff/` package defines one common representation — `MethodResult` (the display tree + per-leaf homing + per-leaf membership edge-type) — that every method emits via an *adapter*, plus pure metric functions that consume only a `MethodResult`. Metrics are unit-tested against hand-built results (no ontology); adapters are tested against the `mini_foodon.obo` fixture. The bake-off notebook's build script imports the harness and renders a scorecard above the existing tree columns. This plan is the decision-ready gate from [`docs/methods_layer_a_bakeoff_brief.md`](../../methods_layer_a_bakeoff_brief.md) §4–§5; the agentic MCP method (§3) is a separate Plan B, written only if `1a+` doesn't clear the bar.

**Tech Stack:** Python 3.11, pydantic v2, pytest, dataclasses. LLM (for nameability) via `LLMClient.generate_json`. Tests use a fake LLM (no network). Env: `/mnt/miniconda3/envs/foodscholar/bin/python`.

---

## File Structure

- **Create** `src/foodscholar/layer_a/bakeoff/__init__.py` — package exports.
- **Create** `src/foodscholar/layer_a/bakeoff/result.py` — `MethodResult` dataclass + `node_depths` helper + adapters `from_children_map` (projection methods) and `from_shelves` (prune/grouping).
- **Create** `src/foodscholar/layer_a/bakeoff/metrics.py` — pure metric functions: `coverage`, `fan_out`, `tree_depth`, `findability`, `faithfulness`, `reproducibility`, `nameability`, plus a `sample_query_leaves` helper.
- **Create** `src/foodscholar/layer_a/bakeoff/scorecard.py` — `build_scorecard` (one row per method) + `render_scorecard_markdown` / `render_scorecard_html`.
- **Create** `tests/unit/test_bakeoff_result.py`, `tests/unit/test_bakeoff_metrics.py`, `tests/unit/test_bakeoff_scorecard.py`.
- **Modify** `scripts/build_projection_bakeoff_nb.py` → add scorecard + grouping column, rename to `scripts/build_layer_a_method_bakeoff_nb.py`; regenerate as `notebooks/layer_a_method_bakeoff.ipynb`.
- **Move** `notebooks/retier_layer_a.ipynb`, `notebooks/entrypoint_audit.ipynb` (and their build scripts) to `notebooks/archive/` / `scripts/archive/`.

### Core type (defined here so every task agrees on it)

```python
@dataclass
class MethodResult:
    name: str
    root: str                              # root node id
    edges: dict[str, list[str]]            # parent node id -> ordered child node ids
    labels: dict[str, str]                 # node id -> display label
    counts: dict[str, int]                 # node id -> chunk count
    leaf_home: dict[str, str]              # mentioned-leaf foodon id -> home node id
    home_edge_type: dict[str, str]         # leaf id -> 'is-a' | 'other-relation' | 'fabricated'
    llm_calls: int = 0
    audit: list[dict] = field(default_factory=list)
```

`leaf_home`/`home_edge_type` are the membership record: which node a corpus-mentioned leaf is reachable under, and *how* (faithful is-a ancestry, a non-is-a FoodOn relation, or a fabricated/label assignment). All metrics read only this struct.

---

## Conventions for the worker

- Run tests with: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest`.
- The test ontology fixture is `tests/fixtures/mini_foodon.obo`, loaded via `FoodOnAPI(load_ontology(path), prefix_filter=None)` (see `tests/unit/test_layer_a.py:10-12`). Its terms: `TEST:0000001 food product` → `0000002 plant food` → `0000004 fruit` → `0000006 apple`, `0000007 olive` → `0000008 olive oil`; `0000004` also → none else; `plant food` → `0000005 vegetable`, `0000009 peanut`; `food product` → `0000003 animal food` → `0000011 dairy product`; `0000010 legacy term` is an orphan root.
- `FoodOnAPI` methods used: `id_to_label(id)->str|None`, `id_to_ancestors(id)->list[str]` (closed transitive), `is_subclass_of(child, anc)->bool` (includes self), `__contains__`.
- `Shelf` (`src/foodscholar/io/graph.py:24-35`): `shelf_id, label, display_label, facet, depth, foodon_id, parent_shelf_id, chunk_count, support_direct, support_lifted, see_also`.
- `LLMClient.generate_json(prompt, schema, max_tokens=1024) -> dict` — may raise; handle defensively.
- Commit after every task.

---

### Task 1: Package skeleton + `MethodResult` + `node_depths`

**Files:**
- Create: `src/foodscholar/layer_a/bakeoff/__init__.py`
- Create: `src/foodscholar/layer_a/bakeoff/result.py`
- Test: `tests/unit/test_bakeoff_result.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_bakeoff_result.py`:

```python
from foodscholar.layer_a.bakeoff.result import MethodResult, node_depths


def _toy() -> MethodResult:
    # root -> A -> A1 ; root -> B
    return MethodResult(
        name="toy",
        root="root",
        edges={"root": ["A", "B"], "A": ["A1"]},
        labels={"root": "Foods", "A": "Fruit", "A1": "Apple", "B": "Dairy"},
        counts={"root": 3, "A": 2, "A1": 2, "B": 1},
        leaf_home={"A1": "A1", "B": "B"},
        home_edge_type={"A1": "is-a", "B": "is-a"},
    )


def test_node_depths_bfs_from_root():
    d = node_depths(_toy())
    assert d == {"root": 0, "A": 1, "B": 1, "A1": 2}


def test_node_depths_ignores_unreachable():
    r = _toy()
    r.edges["orphan"] = ["x"]  # not reachable from root
    d = node_depths(r)
    assert "orphan" not in d and "x" not in d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_result.py -v`
Expected: FAIL with `ModuleNotFoundError: foodscholar.layer_a.bakeoff.result`.

- [ ] **Step 3: Implement the type + helper**

Create `src/foodscholar/layer_a/bakeoff/__init__.py`:

```python
"""Layer-A method bake-off harness: a common MethodResult + pure metrics.

See docs/methods_layer_a_bakeoff_brief.md. Every construction method emits a
MethodResult via an adapter; metrics consume only that struct so methods are
scored on identical footing.
"""

from foodscholar.layer_a.bakeoff.result import MethodResult, node_depths

__all__ = ["MethodResult", "node_depths"]
```

Create `src/foodscholar/layer_a/bakeoff/result.py`:

```python
"""Common representation for a constructed Layer-A method + tree helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class MethodResult:
    name: str
    root: str
    edges: dict[str, list[str]]
    labels: dict[str, str]
    counts: dict[str, int]
    leaf_home: dict[str, str]
    home_edge_type: dict[str, str]
    llm_calls: int = 0
    audit: list[dict] = field(default_factory=list)


def node_depths(result: MethodResult) -> dict[str, int]:
    """BFS depth of every node reachable from `root` (root = 0). Cycle-safe."""
    depths: dict[str, int] = {result.root: 0}
    queue: deque[str] = deque([result.root])
    while queue:
        node = queue.popleft()
        for child in result.edges.get(node, []):
            if child not in depths:
                depths[child] = depths[node] + 1
                queue.append(child)
    return depths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_result.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_a/bakeoff/__init__.py src/foodscholar/layer_a/bakeoff/result.py tests/unit/test_bakeoff_result.py
git commit -m "feat(bakeoff): MethodResult + node_depths tree helper"
```

---

### Task 2: Structural metrics — `coverage`, `fan_out`, `tree_depth`

**Files:**
- Create: `src/foodscholar/layer_a/bakeoff/metrics.py`
- Test: `tests/unit/test_bakeoff_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_bakeoff_metrics.py`:

```python
from foodscholar.layer_a.bakeoff.metrics import coverage, fan_out, tree_depth
from foodscholar.layer_a.bakeoff.result import MethodResult


def _toy() -> MethodResult:
    return MethodResult(
        name="toy", root="root",
        edges={"root": ["A", "B"], "A": ["A1", "A2"]},
        labels={"root": "Foods", "A": "Fruit", "A1": "Apple", "A2": "Olive", "B": "Dairy"},
        counts={"root": 4, "A": 3, "A1": 2, "A2": 1, "B": 1},
        leaf_home={"A1": "A1", "A2": "A2", "B": "B"},
        home_edge_type={"A1": "is-a", "A2": "is-a", "B": "is-a"},
    )


def test_coverage_fraction_of_mentioned_leaves_homed():
    r = _toy()
    # 3 of 4 mentioned leaves are homed (X is not)
    assert coverage(r, {"A1", "A2", "B", "X"}) == 0.75


def test_fan_out_max_and_median_over_internal_nodes():
    mx, med = fan_out(_toy())
    assert mx == 2          # root and A both have 2 children
    assert med == 2.0


def test_tree_depth_max_and_median():
    mx, med = tree_depth(_toy())
    assert mx == 2          # A1/A2 at depth 2
    assert med == 2.0       # depths: root0, A1, B1, A1=2, A2=2 -> median of [0,1,1,2,2]=1 ... see impl note
```

> Note: `tree_depth` is computed over **homed leaf nodes** (what the user actually lands on), not all nodes — so the median in the assertion is over `{A1:2, A2:2, B:1}` → max 2, median 2.0. Adjust the assertion comment if you compute differently; the test value `2.0` assumes leaf-node depths.

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/foodscholar/layer_a/bakeoff/metrics.py`:

```python
"""Pure metric functions over a MethodResult. No ontology, no I/O."""

from __future__ import annotations

import statistics

from foodscholar.layer_a.bakeoff.result import MethodResult, node_depths


def coverage(result: MethodResult, mentioned_leaves: set[str]) -> float:
    """Fraction of mentioned leaves that are homed under some node."""
    if not mentioned_leaves:
        return 0.0
    homed = sum(1 for fid in mentioned_leaves if fid in result.leaf_home)
    return homed / len(mentioned_leaves)


def fan_out(result: MethodResult) -> tuple[int, float]:
    """(max, median) children over internal (non-leaf) nodes."""
    sizes = [len(kids) for kids in result.edges.values() if kids]
    if not sizes:
        return 0, 0.0
    return max(sizes), float(statistics.median(sizes))


def tree_depth(result: MethodResult) -> tuple[int, float]:
    """(max, median) depth over the nodes users land on (homed leaf homes)."""
    depths = node_depths(result)
    home_depths = [
        depths[home] for home in result.leaf_home.values() if home in depths
    ]
    if not home_depths:
        return 0, 0.0
    return max(home_depths), float(statistics.median(home_depths))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_a/bakeoff/metrics.py tests/unit/test_bakeoff_metrics.py
git commit -m "feat(bakeoff): coverage / fan_out / tree_depth metrics"
```

---

### Task 3: `findability` + `sample_query_leaves`

**Files:**
- Modify: `src/foodscholar/layer_a/bakeoff/metrics.py`
- Test: `tests/unit/test_bakeoff_metrics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_bakeoff_metrics.py`:

```python
from foodscholar.layer_a.bakeoff.metrics import findability, sample_query_leaves


def test_findability_clicks_from_root():
    r = _toy()  # depths: A1=2, A2=2, B=1
    out = findability(r, ["A1", "A2", "B"], k=2)
    assert out["median_clicks"] == 2.0       # [2,2,1] -> median 2
    assert out["p90_clicks"] == 2
    assert out["pct_within_k"] == 1.0        # all <= 2
    assert out["pct_reachable"] == 1.0


def test_findability_unreachable_leaf_counts_against_reachable():
    r = _toy()
    out = findability(r, ["A1", "X"], k=2)   # X not homed
    assert out["pct_reachable"] == 0.5
    assert out["pct_within_k"] == 0.5        # only A1 within k


def test_sample_query_leaves_is_deterministic_and_stratified():
    freq = {"a": 100, "b": 50, "c": 1, "d": 1, "e": 1}
    s1 = sample_query_leaves(freq, n=4)
    s2 = sample_query_leaves(freq, n=4)
    assert s1 == s2                          # deterministic (no RNG)
    assert "a" in s1 and "c" in s1           # both a common and a rare leaf included
    assert len(s1) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_metrics.py -k "findability or sample_query" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Add to `metrics.py`:

```python
def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round((pct / 100) * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def findability(result: MethodResult, query_leaves: list[str], *, k: int) -> dict:
    """For each query leaf, clicks = depth of its home node from root.

    Unreachable leaves (no home) are excluded from the click stats but counted
    against pct_reachable and pct_within_k.
    """
    depths = node_depths(result)
    clicks: list[int] = []
    reachable = 0
    for fid in query_leaves:
        home = result.leaf_home.get(fid)
        if home is not None and home in depths:
            reachable += 1
            clicks.append(depths[home])
    total = len(query_leaves) or 1
    sorted_clicks = sorted(clicks)
    within_k = sum(1 for c in clicks if c <= k)
    return {
        "median_clicks": float(statistics.median(sorted_clicks)) if sorted_clicks else 0.0,
        "p90_clicks": _percentile([float(c) for c in sorted_clicks], 90),
        "pct_within_k": within_k / total,
        "pct_reachable": reachable / total,
    }


def sample_query_leaves(leaf_freq: dict[str, int], *, n: int) -> list[str]:
    """Deterministic stratified sample of leaves: take the most frequent half
    and the least frequent half so both common and rare foods are tested.
    No RNG (reproducible across runs)."""
    if n <= 0 or not leaf_freq:
        return []
    by_freq = sorted(leaf_freq, key=lambda fid: (-leaf_freq[fid], fid))
    if n >= len(by_freq):
        return by_freq
    head = n // 2
    tail = n - head
    return by_freq[:head] + by_freq[-tail:]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_metrics.py -k "findability or sample_query" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_a/bakeoff/metrics.py tests/unit/test_bakeoff_metrics.py
git commit -m "feat(bakeoff): findability metric + deterministic stratified query sampler"
```

---

### Task 4: `faithfulness` tally + `reproducibility`

**Files:**
- Modify: `src/foodscholar/layer_a/bakeoff/metrics.py`
- Test: `tests/unit/test_bakeoff_metrics.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from foodscholar.layer_a.bakeoff.metrics import faithfulness, reproducibility


def test_faithfulness_tallies_home_edge_types():
    r = _toy()
    r.home_edge_type = {"A1": "is-a", "A2": "is-a", "B": "fabricated"}
    f = faithfulness(r)
    assert f["is-a"] == 2 / 3
    assert f["fabricated"] == 1 / 3
    assert f["other-relation"] == 0.0


def test_reproducibility_jaccard_of_node_sets():
    a = _toy()
    b = _toy()
    assert reproducibility(a, b) == 1.0      # identical node sets
    b.edges = {"root": ["A"], "A": ["A1"]}   # drop B and A2
    assert reproducibility(a, b) < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_metrics.py -k "faithfulness or reproducibility" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Add to `metrics.py`:

```python
def faithfulness(result: MethodResult) -> dict[str, float]:
    """Fraction of homed leaves whose membership edge is is-a / other-relation /
    fabricated. is-a + other-relation = 'within FoodOn'; fabricated = invented."""
    cats = {"is-a": 0, "other-relation": 0, "fabricated": 0}
    for etype in result.home_edge_type.values():
        if etype in cats:
            cats[etype] += 1
    total = sum(cats.values()) or 1
    return {k: v / total for k, v in cats.items()}


def _all_nodes(result: MethodResult) -> set[str]:
    nodes = {result.root, *result.edges.keys()}
    for kids in result.edges.values():
        nodes.update(kids)
    return nodes


def reproducibility(a: MethodResult, b: MethodResult) -> float:
    """Jaccard similarity of the two runs' node-id sets (1.0 = identical)."""
    na, nb = _all_nodes(a), _all_nodes(b)
    union = na | nb
    if not union:
        return 1.0
    return len(na & nb) / len(union)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_metrics.py -k "faithfulness or reproducibility" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_a/bakeoff/metrics.py tests/unit/test_bakeoff_metrics.py
git commit -m "feat(bakeoff): faithfulness edge-type tally + reproducibility Jaccard"
```

---

### Task 5: `nameability` (LLM-judged, fake LLM in tests)

**Files:**
- Modify: `src/foodscholar/layer_a/bakeoff/metrics.py`
- Test: `tests/unit/test_bakeoff_metrics.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from foodscholar.layer_a.bakeoff.metrics import nameability


class _FakeLLM:
    model_id = "fake"

    def __init__(self, verdicts):
        self._verdicts = verdicts  # dict label -> bool

    def generate(self, prompt, max_tokens=1024):
        return ""

    def generate_json(self, prompt, schema, max_tokens=1024):
        return {"verdicts": [
            {"label": lbl, "recognizable": ok} for lbl, ok in self._verdicts.items()
        ]}


def test_nameability_fraction_recognizable():
    r = _toy()  # shelf labels: Fruit, Apple, Olive, Dairy (root 'Foods' excluded)
    llm = _FakeLLM({"Apple": True, "Dairy": True, "Fruit": True, "Olive": False})
    score = nameability(r, llm, sample=10)
    assert score == 0.75   # 3 of 4 recognizable


def test_nameability_zero_when_llm_raises():
    class Boom(_FakeLLM):
        def generate_json(self, prompt, schema, max_tokens=1024):
            raise RuntimeError("no llm")
    assert nameability(_toy(), Boom({}), sample=10) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_metrics.py -k nameability -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Add to `metrics.py`:

```python
def nameability(result: MethodResult, llm, *, sample: int) -> float:
    """Fraction of a deterministic sample of shelf labels an LLM judges
    'recognizable to a layperson'. Excludes the root. Returns 0.0 if the LLM
    errors (so a broken judge never inflates the score)."""
    labels = sorted(
        {lbl for nid, lbl in result.labels.items() if nid != result.root}
    )[:sample]
    if not labels:
        return 0.0
    schema = {
        "type": "object",
        "properties": {"verdicts": {"type": "array", "items": {
            "type": "object",
            "properties": {"label": {"type": "string"}, "recognizable": {"type": "boolean"}},
            "required": ["label", "recognizable"],
        }}},
        "required": ["verdicts"],
    }
    prompt = (
        "For each food-category label, would a layperson browsing a nutrition "
        "site recognize it as a food group / food (true) or is it jargon / an "
        "organizational artifact (false)?\nLabels:\n"
        + "\n".join(f"  - {lbl}" for lbl in labels)
        + '\n\nReturn JSON {"verdicts": [{"label": "...", "recognizable": true}]}.'
    )
    try:
        obj = llm.generate_json(prompt, schema, max_tokens=2048)
    except Exception:
        return 0.0
    verdict = {
        v.get("label"): bool(v.get("recognizable"))
        for v in (obj or {}).get("verdicts", [])
    }
    ok = sum(1 for lbl in labels if verdict.get(lbl) is True)
    return ok / len(labels)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_metrics.py -k nameability -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_a/bakeoff/metrics.py tests/unit/test_bakeoff_metrics.py
git commit -m "feat(bakeoff): nameability metric (LLM-judged, defensive)"
```

---

### Task 6: `from_children_map` adapter (projection methods → MethodResult)

**Files:**
- Modify: `src/foodscholar/layer_a/bakeoff/result.py`
- Test: `tests/unit/test_bakeoff_result.py`

The projection columns (`1a`, `1a+`, `1b`) build an explicit `children_map` (parent id → child ids) over real FoodOn ids plus `counts`/`labels`. This adapter wraps that and computes `leaf_home`/`home_edge_type` from the ontology: a mentioned leaf homes to the **deepest** tree node that is an is-a ancestor-or-self of it.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_bakeoff_result.py`:

```python
from pathlib import Path

from foodscholar.layer_a.bakeoff.result import from_children_map
from foodscholar.ontology import FoodOnAPI, load_ontology


def _mini_foodon() -> FoodOnAPI:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"
    return FoodOnAPI(load_ontology(path), prefix_filter=None)


def test_from_children_map_homes_leaves_to_deepest_tree_ancestor():
    api = _mini_foodon()
    # tree: food product -> {plant food, animal food}; plant food kept as a shelf
    children = {
        "TEST:0000001": ["TEST:0000002", "TEST:0000003"],
        "TEST:0000002": [],
        "TEST:0000003": [],
    }
    counts = {"TEST:0000001": 3, "TEST:0000002": 2, "TEST:0000003": 1}
    labels = {fid: api.id_to_label(fid) for fid in counts}
    mentioned = {"TEST:0000006", "TEST:0000011"}  # apple, dairy product
    r = from_children_map(
        "1a", root="TEST:0000001", children_map=children, counts=counts,
        labels=labels, ontology=api, mentioned_leaves=mentioned,
    )
    # apple homes under plant food (deepest kept ancestor), dairy under animal food
    assert r.leaf_home["TEST:0000006"] == "TEST:0000002"
    assert r.leaf_home["TEST:0000011"] == "TEST:0000003"
    # projection edges are all real is-a
    assert r.home_edge_type["TEST:0000006"] == "is-a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_result.py -k from_children_map -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Add to `result.py` (add `from typing import TYPE_CHECKING` + ontology import guard at top):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foodscholar.ontology import FoodOnAPI


def _deepest_tree_home(
    leaf: str, tree_nodes: set[str], ontology: FoodOnAPI
) -> str | None:
    """The most-specific tree node that is an is-a ancestor-or-self of `leaf`."""
    if leaf in tree_nodes:
        return leaf
    candidates = [a for a in ontology.id_to_ancestors(leaf) if a in tree_nodes]
    if not candidates:
        return None
    # deepest = the candidate with the longest ancestor chain (closest to leaf)
    return max(candidates, key=lambda a: len(ontology.id_to_ancestors(a)))


def from_children_map(
    name: str,
    *,
    root: str,
    children_map: dict[str, list[str]],
    counts: dict[str, int],
    labels: dict[str, str],
    ontology: FoodOnAPI,
    mentioned_leaves: set[str],
) -> MethodResult:
    """Wrap a projection method's explicit FoodOn children_map. Leaves home to
    their deepest kept is-a ancestor (membership is is-a → faithful)."""
    tree_nodes = {root, *children_map.keys()}
    for kids in children_map.values():
        tree_nodes.update(kids)
    leaf_home: dict[str, str] = {}
    home_edge_type: dict[str, str] = {}
    for leaf in mentioned_leaves:
        home = _deepest_tree_home(leaf, tree_nodes, ontology)
        if home is not None:
            leaf_home[leaf] = home
            home_edge_type[leaf] = "is-a"
    return MethodResult(
        name=name, root=root,
        edges={p: list(kids) for p, kids in children_map.items()},
        labels=dict(labels), counts=dict(counts),
        leaf_home=leaf_home, home_edge_type=home_edge_type,
    )
```

Add `from_children_map` to `__init__.py`'s imports/`__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_result.py -k from_children_map -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_a/bakeoff/result.py src/foodscholar/layer_a/bakeoff/__init__.py tests/unit/test_bakeoff_result.py
git commit -m "feat(bakeoff): from_children_map adapter (projection methods)"
```

---

### Task 7: `from_shelves` adapter (prune + grouping → MethodResult)

**Files:**
- Modify: `src/foodscholar/layer_a/bakeoff/result.py`
- Test: `tests/unit/test_bakeoff_result.py`

`prune()` and `build_grouped_shelves()` emit `list[Shelf]`. Tree edges come from `parent_shelf_id`; node id = `foodon_id` (root shelf has none → use its `shelf_id`). Homing: a leaf homes to a shelf if it's in that shelf's `see_also` (grouping assignment) OR is an is-a descendant of the shelf's `foodon_id`. `home_edge_type` is `is-a` when the leaf is genuinely a subclass of the home's `foodon_id`, else `fabricated` (label grouping placed it there with no is-a relation).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_bakeoff_result.py`:

```python
from foodscholar.io.graph import Shelf
from foodscholar.layer_a.bakeoff.result import from_shelves


def test_from_shelves_groups_mark_nonancestor_membership_fabricated():
    api = _mini_foodon()
    root = Shelf(shelf_id="facet:foods", label="Foods", facet="foods", depth=0)
    # group shelf anchored at 'fruit' but (by label-grouping) also claims 'peanut',
    # which is NOT an is-a descendant of fruit -> fabricated membership.
    fruit = Shelf(
        shelf_id="foodon:TEST:0000004", label="fruit", display_label="Fruits",
        facet="foods", depth=1, foodon_id="TEST:0000004",
        parent_shelf_id="facet:foods", chunk_count=3,
        see_also=["TEST:0000006", "TEST:0000009"],  # apple (is-a), peanut (not)
    )
    r = from_shelves("grouping", [root, fruit], ontology=api,
                     mentioned_leaves={"TEST:0000006", "TEST:0000009"})
    assert r.root == "facet:foods"
    assert r.edges["facet:foods"] == ["TEST:0000004"]
    assert r.leaf_home["TEST:0000006"] == "TEST:0000004"
    assert r.home_edge_type["TEST:0000006"] == "is-a"        # apple ⊂ fruit
    assert r.home_edge_type["TEST:0000009"] == "fabricated"  # peanut ⊄ fruit
    # display label preferred over FoodOn label
    assert r.labels["TEST:0000004"] == "Fruits"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_result.py -k from_shelves -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Add to `result.py`:

```python
def _shelf_node_id(shelf) -> str:
    return shelf.foodon_id or shelf.shelf_id


def from_shelves(
    name: str,
    shelves: list,  # list[Shelf]
    *,
    ontology: FoodOnAPI,
    mentioned_leaves: set[str],
) -> MethodResult:
    """Wrap a list[Shelf] (prune / grouping output) into a MethodResult."""
    by_shelf_id = {s.shelf_id: s for s in shelves}
    node_of = {s.shelf_id: _shelf_node_id(s) for s in shelves}

    root = next((s for s in shelves if s.parent_shelf_id is None), None)
    root_id = node_of[root.shelf_id] if root else (shelves[0].shelf_id if shelves else "")

    edges: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    counts: dict[str, int] = {}
    for s in shelves:
        nid = node_of[s.shelf_id]
        labels[nid] = s.display_label or s.label
        counts[nid] = s.chunk_count
        if s.parent_shelf_id is not None and s.parent_shelf_id in node_of:
            edges.setdefault(node_of[s.parent_shelf_id], []).append(nid)

    # homing: see_also membership first (grouping), else deepest is-a ancestor shelf
    shelf_nodes = {node_of[s.shelf_id] for s in shelves if s.foodon_id}
    leaf_home: dict[str, str] = {}
    home_edge_type: dict[str, str] = {}
    for leaf in mentioned_leaves:
        home_shelf = next(
            (s for s in shelves if leaf in s.see_also), None
        )
        if home_shelf is not None:
            home_node = node_of[home_shelf.shelf_id]
            leaf_home[leaf] = home_node
            anchor = home_shelf.foodon_id
            home_edge_type[leaf] = (
                "is-a" if anchor and ontology.is_subclass_of(leaf, anchor) else "fabricated"
            )
            continue
        home = _deepest_tree_home(leaf, shelf_nodes, ontology)
        if home is not None:
            leaf_home[leaf] = home
            home_edge_type[leaf] = "is-a"
    return MethodResult(
        name=name, root=root_id, edges=edges, labels=labels, counts=counts,
        leaf_home=leaf_home, home_edge_type=home_edge_type,
    )
```

Add `from_shelves` to `__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_result.py -k from_shelves -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_a/bakeoff/result.py src/foodscholar/layer_a/bakeoff/__init__.py tests/unit/test_bakeoff_result.py
git commit -m "feat(bakeoff): from_shelves adapter (prune + grouping, fabricated membership flagged)"
```

---

### Task 8: Scorecard assembler + renderers

**Files:**
- Create: `src/foodscholar/layer_a/bakeoff/scorecard.py`
- Test: `tests/unit/test_bakeoff_scorecard.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_bakeoff_scorecard.py`:

```python
from foodscholar.layer_a.bakeoff.result import MethodResult
from foodscholar.layer_a.bakeoff.scorecard import build_scorecard, render_scorecard_markdown


def _r(name) -> MethodResult:
    return MethodResult(
        name=name, root="root",
        edges={"root": ["A", "B"], "A": ["A1"]},
        labels={"root": "Foods", "A": "Fruit", "A1": "Apple", "B": "Dairy"},
        counts={"root": 3, "A": 2, "A1": 2, "B": 1},
        leaf_home={"A1": "A1", "B": "B"},
        home_edge_type={"A1": "is-a", "B": "is-a"},
    )


def test_build_scorecard_one_row_per_method():
    rows = build_scorecard(
        [_r("1a"), _r("1a+")],
        mentioned_leaves={"A1", "B"},
        query_leaves=["A1", "B"],
        k=2,
        llm=None,
    )
    assert [row["method"] for row in rows] == ["1a", "1a+"]
    assert rows[0]["coverage"] == 1.0
    assert rows[0]["faithfulness_is_a"] == 1.0
    assert "nameability" in rows[0]            # None when llm omitted
    assert rows[0]["nameability"] is None


def test_render_scorecard_markdown_has_header_and_rows():
    rows = build_scorecard([_r("1a")], mentioned_leaves={"A1", "B"},
                           query_leaves=["A1"], k=2, llm=None)
    md = render_scorecard_markdown(rows)
    assert "| method |" in md
    assert "1a" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_scorecard.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/foodscholar/layer_a/bakeoff/scorecard.py`:

```python
"""Assemble + render the method scorecard."""

from __future__ import annotations

from foodscholar.layer_a.bakeoff import metrics as M
from foodscholar.layer_a.bakeoff.result import MethodResult

_COLUMNS = [
    "method", "coverage", "find_median", "find_p90", "find_pct_within_k",
    "nameability", "fanout_max", "depth_max",
    "faithfulness_is_a", "faithfulness_fabricated", "llm_calls",
]


def build_scorecard(
    results: list[MethodResult],
    *,
    mentioned_leaves: set[str],
    query_leaves: list[str],
    k: int,
    llm=None,
    nameability_sample: int = 25,
) -> list[dict]:
    rows: list[dict] = []
    for r in results:
        find = M.findability(r, query_leaves, k=k)
        faith = M.faithfulness(r)
        fo_max, _ = M.fan_out(r)
        d_max, _ = M.tree_depth(r)
        rows.append({
            "method": r.name,
            "coverage": M.coverage(r, mentioned_leaves),
            "find_median": find["median_clicks"],
            "find_p90": find["p90_clicks"],
            "find_pct_within_k": find["pct_within_k"],
            "nameability": (M.nameability(r, llm, sample=nameability_sample)
                            if llm is not None else None),
            "fanout_max": fo_max,
            "depth_max": d_max,
            "faithfulness_is_a": faith["is-a"],
            "faithfulness_fabricated": faith["fabricated"],
            "llm_calls": r.llm_calls,
        })
    return rows


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def render_scorecard_markdown(rows: list[dict]) -> str:
    header = "| " + " | ".join(_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    body = "\n".join(
        "| " + " | ".join(_fmt(row.get(c)) for c in _COLUMNS) + " |"
        for row in rows
    )
    return f"{header}\n{sep}\n{body}"


def render_scorecard_html(rows: list[dict]) -> str:
    head = "".join(f"<th>{c}</th>" for c in _COLUMNS)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(row.get(c))}</td>" for c in _COLUMNS) + "</tr>"
        for row in rows
    )
    return f"<table class='scorecard'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_scorecard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_a/bakeoff/scorecard.py tests/unit/test_bakeoff_scorecard.py
git commit -m "feat(bakeoff): scorecard assembler + markdown/html renderers"
```

---

### Task 9: Wire the harness into the bake-off notebook (+ grouping column, rename)

**Files:**
- Modify → rename: `scripts/build_projection_bakeoff_nb.py` → `scripts/build_layer_a_method_bakeoff_nb.py`
- Output: `notebooks/layer_a_method_bakeoff.ipynb`

This task is notebook-generation (no pytest TDD); verification is "the build script runs and the scorecard cell is present." First read the tail of `scripts/build_projection_bakeoff_nb.py` to see how `COLUMNS` are finalized/rendered and where the notebook is written, then append the scorecard.

- [ ] **Step 1: Rename the build script**

```bash
git mv scripts/build_projection_bakeoff_nb.py scripts/build_layer_a_method_bakeoff_nb.py
```

- [ ] **Step 2: Point the output at the new notebook name**

In `scripts/build_layer_a_method_bakeoff_nb.py`, find the output path (search for `projection_bakeoff.ipynb`) and change it to `layer_a_method_bakeoff.ipynb`. Update the notebook's title markdown cell to "Layer-A Method Bake-off".

- [ ] **Step 3: Add a grouping-method column**

After the existing method columns are built (search for the last `COLUMNS.append(`), add a cell that runs `build_grouped_shelves` on the same chunks and converts it via `from_shelves`, so the merged method appears alongside the projection ones. Append this cell (mirrors the existing `code(...)`/`cells.append` pattern):

```python
cells.append(
    code(
        '''# Merged bottom-up + LLM grouping method (from main) as a bake-off column.
from foodscholar.config import BottomUpGroupingConfig
from foodscholar.layer_a.grouping import build_grouped_shelves
from foodscholar.layer_a.bakeoff.result import from_shelves

_grouping_cfg = BottomUpGroupingConfig(enabled=True)
_grouping_shelves = build_grouped_shelves(
    iter(chunks), api, _grouping_cfg, facet="foods",
    min_link_confidence=0.0, llm=fs.llm,
)
GROUPING_RESULT = from_shelves(
    "grouping (main)", _grouping_shelves, ontology=api,
    mentioned_leaves=set(TERM_DOC_FREQ),
)
print(f"grouping column: {len(_grouping_shelves)} shelves")'''
    )
)
```

- [ ] **Step 4: Add the scorecard cell (above the tree columns render)**

Add a cell that converts the projection `COLUMNS`' underlying children-maps into `MethodResult`s and builds the scorecard. The projection columns store their trees as rendered HTML, so capture each method's `children`/`counts`/`labels`/`root` into a registry as they're built. Add this near the top helpers cell (after `COLUMNS = []`):

```python
RESULTS = []  # list[MethodResult] for the scorecard
```

and in each `*_backbone_column(...)` / `controlled_backbone_column(...)` function, right before `COLUMNS.append({...})`, register a result:

```python
    from foodscholar.layer_a.bakeoff.result import from_children_map
    RESULTS.append(from_children_map(
        title.split(" —")[0], root=ROOT, children_map=children,
        counts=counts, labels=labels, ontology=api,
        mentioned_leaves=set(TERM_DOC_FREQ),
    ))
```

Then append the scorecard cell after all columns + the grouping cell:

```python
cells.append(
    code(
        '''# ---- Scorecard: every method on the same metrics ----------------------------
from IPython.display import HTML, Markdown
from foodscholar.layer_a.bakeoff.metrics import sample_query_leaves
from foodscholar.layer_a.bakeoff.scorecard import build_scorecard, render_scorecard_html

ALL_RESULTS = RESULTS + [GROUPING_RESULT]
MENTIONED = set(TERM_DOC_FREQ)
QUERY_LEAVES = sample_query_leaves(dict(TERM_DOC_FREQ), n=100)

SCORECARD = build_scorecard(
    ALL_RESULTS, mentioned_leaves=MENTIONED, query_leaves=QUERY_LEAVES,
    k=3, llm=fs.llm, nameability_sample=25,
)
display(HTML(render_scorecard_html(SCORECARD)))'''
    )
)
```

(If `ROOT` differs per projection function, use that function's own root-id variable — the auto/controlled functions use `"__controlled_backbone_root__"` and similar; pass whatever each function already uses as its tree root.)

- [ ] **Step 5: Regenerate the notebook and verify it builds**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python scripts/build_layer_a_method_bakeoff_nb.py`
Expected: writes `notebooks/layer_a_method_bakeoff.ipynb` with no error.

Verify the scorecard cell is present:
Run: `grep -c "build_scorecard" notebooks/layer_a_method_bakeoff.ipynb`
Expected: `>= 1`.

> Full execution (running every cell) needs the live corpus + FoodOn and is a **manual** check — run the notebook top-to-bottom in the foodscholar env and confirm the scorecard renders. Note in the commit if executed.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_layer_a_method_bakeoff_nb.py notebooks/layer_a_method_bakeoff.ipynb
git rm notebooks/projection_bakeoff.ipynb
git commit -m "feat(bakeoff): scorecard + grouping column in renamed layer_a_method_bakeoff notebook"
```

---

### Task 10: Archive superseded notebooks

**Files:**
- Move: `notebooks/retier_layer_a.ipynb`, `notebooks/entrypoint_audit.ipynb` → `notebooks/archive/`
- Move: `scripts/build_retier_layer_a_nb.py`, `scripts/build_entrypoint_audit_nb.py` → `scripts/archive/`

- [ ] **Step 1: Move the superseded notebooks + their builders**

```bash
mkdir -p notebooks/archive scripts/archive
git mv notebooks/retier_layer_a.ipynb notebooks/archive/retier_layer_a.ipynb
git mv notebooks/entrypoint_audit.ipynb notebooks/archive/entrypoint_audit.ipynb
git mv scripts/build_retier_layer_a_nb.py scripts/archive/build_retier_layer_a_nb.py
git mv scripts/build_entrypoint_audit_nb.py scripts/archive/build_entrypoint_audit_nb.py
```

- [ ] **Step 2: Leave a one-line pointer**

Create `notebooks/archive/README.md`:

```markdown
# Archived Layer-A exploration notebooks

Superseded by `notebooks/layer_a_method_bakeoff.ipynb` (the single
method-evaluation entry point). Kept for provenance; not maintained.

- `retier_layer_a.ipynb` — re-tiering experiment (rejected; made fan-out worse).
- `entrypoint_audit.ipynb` — early entry-point audit, folded into the bake-off metrics.
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/archive scripts/archive
git commit -m "chore(bakeoff): archive superseded retier + entrypoint-audit notebooks"
```

---

### Task 11: Full-suite gate + brief cross-check

**Files:** none (verification task)

- [ ] **Step 1: Run the whole unit suite**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit -q`
Expected: all PASS. Note (do not fix here) any pre-existing failures unrelated to this change.

- [ ] **Step 2: Lint the new package**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m ruff check src/foodscholar/layer_a/bakeoff/`
Expected: clean (fix any issues in the new files).

- [ ] **Step 3: Cross-check against the brief**

Confirm §4 metrics all exist (coverage, findability, nameability, fan-out, depth, faithfulness, reproducibility) and the scorecard renders them; confirm §5 consolidation done (one bake-off notebook, others archived). The agentic method (§3) is intentionally **out of scope** — that's Plan B.

- [ ] **Step 4: Commit**

```bash
git commit --allow-empty -m "test(bakeoff): full-suite gate green; harness scores all built methods incl. 1a+"
```

---

## Self-Review Notes

- **Spec coverage (brief §4/§5):** coverage/fan-out/depth (Task 2), findability + query set (Task 3), faithfulness + reproducibility (Task 4), nameability (Task 5), the two adapters that feed all methods in (Tasks 6–7), scorecard render (Task 8), notebook wiring + grouping column + rename (Task 9), notebook consolidation/archive (Task 10), gate (Task 11). §3 agentic method deliberately deferred to Plan B (gated on this scorecard).
- **Type consistency:** `MethodResult` fields (`edges`, `labels`, `counts`, `leaf_home`, `home_edge_type`, `llm_calls`) are used identically across Tasks 1–8; `from_children_map`/`from_shelves` signatures match their callers in Task 9; `_deepest_tree_home` is defined in Task 6 and reused in Task 7; scorecard `_COLUMNS` keys match the dict keys `build_scorecard` emits.
- **Not in scope (deliberate):** the agentic MCP method, the throwaway relation index, the production port of the winning method, and any prompt-tuning of the nameability judge — follow-ups, not this plan.
- **Risk:** Task 9 depends on the exact variable names inside the existing projection-column functions (`children`, `counts`, `labels`, the per-function root id). The worker MUST read `scripts/build_layer_a_method_bakeoff_nb.py` before editing and bind to whatever those functions actually name their tree maps — the registry snippet assumes `children`/`counts`/`labels` (confirmed present in the `controlled_backbone_column` for `1a+` at `b866e45`).
