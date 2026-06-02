# Layer-A Agentic MCP Construction Method (Plan B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the agentic construction method as a bake-off column — an LLM agent walks the FoodOn support DAG top-down and makes local `KEEP` / `COLLAPSE` / `REPARENT` / `EXPAND` decisions through a read-only MCP-style tool layer (including scope expansion over real non-is-a FoodOn relations), producing a `MethodResult` that plugs into the existing scorecard.

**Architecture:** A new sub-package `src/foodscholar/layer_a/bakeoff/agentic/` with three focused modules: `relations.py` (a throwaway relation index loaded straight from `foodon.owl` via pronto — the production loader keeps is-a only), `tools.py` (read-only graph queries + expansion candidates over the ontology + relation index), and `agent.py` (the DFS loop that drives an `LLMClient` via a manual `generate_json` action protocol — there is **no** native tool-calling — applies guards, logs per-edge relation type + a decision audit, and emits a `MethodResult`). A small `specificity` metric is added so deep vs. flat trees are compared on placement quality, then the method is wired as a GROQ-gated column in `layer_a_method_bakeoff.ipynb`. Builds on Plan A's harness ([2026-06-02-layer-a-method-bakeoff-harness.md](2026-06-02-layer-a-method-bakeoff-harness.md)); design is §3 of [methods_layer_a_bakeoff_brief.md](../../methods_layer_a_bakeoff_brief.md).

**Tech Stack:** Python 3.11, pronto (OWL relation extraction), pydantic, pytest. LLM via `LLMClient.generate_json` (manual `{action, args}` loop, no native tool-calling). Tests use a scripted fake LLM + a tiny relations fixture (no network). Env: `/mnt/miniconda3/envs/foodscholar/bin/python`.

---

## Context the worker needs (verified facts)

- **Non-is-a relations exist and load fast.** `pronto.Ontology("data/foodon.owl", import_depth=0)` loads ~39,682 terms in ~10s. Each `term.relationships` is a dict `{Relationship: <iterable of Term>}`; `Relationship` has `.id` (e.g. `"RO:0001000"`) and `.name` (e.g. `"derives from"`). 8,225 FOODON terms have ≥1 relationship. Most useful food→food relations: `RO:0001000` (derives from), `RO:0002350` (member of), `FOODON:00001563` (has defining ingredient), `FOODON:00002420` (has ingredient), `FOODON:00001301` (has food substance analog). Some targets are non-FOODON (`in taxon`→NCBITaxon, `has quality`→PATO) — **keep only FOODON→FOODON edges**.
- **No native tool-calling.** `LLMClient` (`src/foodscholar/storage/protocols.py`) exposes only `generate(prompt, max_tokens)` and `generate_json(prompt, schema, max_tokens) -> dict`. The agent loop must put the chosen action in the JSON it returns.
- **Reuse from Plan A:** `MethodResult` (`src/foodscholar/layer_a/bakeoff/result.py`) — fields `name, root, edges, labels, counts, leaf_home, home_edge_type, llm_calls, audit`. `collect_leaf_chunks(chunks, ontology, *, facet, min_link_confidence) -> dict[str, set[str]]` (`src/foodscholar/layer_a/grouping.py`). `FoodOnAPI`: `id_to_label`, `id_to_children`, `id_to_parents`, `id_to_ancestors`, `is_subclass_of`, `search(q, limit=)`, `__contains__`.
- **Run tests:** `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest`. End every commit message with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **mini_foodon.obo** (`tests/fixtures/`): `TEST:0000001 food product` → `0000002 plant food` → (`0000004 fruit` → `0000006 apple`; `0000007 olive` → `0000008 olive oil`), `0000005 vegetable`, `0000009 peanut`; `0000001` → `0000003 animal food` → `0000011 dairy product`.

---

### Task 1: Throwaway relation index from the OWL

**Files:**
- Create: `src/foodscholar/layer_a/bakeoff/agentic/__init__.py` (empty package marker)
- Create: `src/foodscholar/layer_a/bakeoff/agentic/relations.py`
- Create: `tests/fixtures/mini_foodon_relations.obo`
- Test: `tests/unit/test_agentic_relations.py`

- [ ] **Step 1: Create the fixture**

Create `tests/fixtures/mini_foodon_relations.obo`:

```
format-version: 1.2
ontology: test

[Typedef]
id: RO:0001000
name: derives from

[Term]
id: FOODON:0000001
name: food product

[Term]
id: FOODON:0000010
name: mammal

[Term]
id: FOODON:0000011
name: mammalian meat food product
is_a: FOODON:0000001 ! food product
relationship: RO:0001000 FOODON:0000010 ! derives from

[Term]
id: NCBITaxon:9606
name: Homo sapiens

[Term]
id: FOODON:0000012
name: human food
is_a: FOODON:0000001 ! food product
relationship: RO:0001000 NCBITaxon:9606 ! derives from
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/test_agentic_relations.py`:

```python
from pathlib import Path

from foodscholar.layer_a.bakeoff.agentic.relations import Relation, load_relation_index

_FIX = Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon_relations.obo"


def test_load_relation_index_keeps_foodon_to_foodon_edges_only():
    idx = load_relation_index(_FIX)
    # FOODON:0000011 -derives from-> FOODON:0000010 is kept
    rels = idx["FOODON:0000011"]
    assert rels == [Relation(rel_id="RO:0001000", rel_name="derives from",
                             target_id="FOODON:0000010")]


def test_load_relation_index_drops_non_foodon_targets():
    idx = load_relation_index(_FIX)
    # FOODON:0000012 -derives from-> NCBITaxon:9606 is dropped (target not FOODON)
    assert "FOODON:0000012" not in idx
```

- [ ] **Step 3: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_agentic_relations.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement**

Create `src/foodscholar/layer_a/bakeoff/agentic/__init__.py` (empty file with a docstring):

```python
"""Agentic (MCP-style) Layer-A construction method for the bake-off (Plan B)."""
```

Create `src/foodscholar/layer_a/bakeoff/agentic/relations.py`:

```python
"""Throwaway non-is-a relation index loaded straight from the FoodOn OWL.

The production ontology loader keeps is-a only; the agentic method needs FoodOn's
object-property relations (derives_from, member_of, has_ingredient, …) to bridge
gaps the is-a graph can't. This loads them once via pronto. Only FOODON→FOODON
edges are kept (targets like NCBITaxon/PATO are dropped). Prototype-only — if the
bake-off shows relations help, fold this into the real loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Relation:
    rel_id: str       # e.g. "RO:0001000"
    rel_name: str     # e.g. "derives from"
    target_id: str    # a FOODON id


def load_relation_index(
    owl_path: str | Path, *, keep_prefix: str = "FOODON:"
) -> dict[str, list[Relation]]:
    """Map each FOODON term id -> its non-is-a relations to other FOODON terms."""
    import pronto

    ont = pronto.Ontology(str(owl_path), import_depth=0)
    index: dict[str, list[Relation]] = {}
    for term in ont.terms():
        if term.id is None or not term.id.startswith(keep_prefix):
            continue
        rels: list[Relation] = []
        for rel, targets in (getattr(term, "relationships", None) or {}).items():
            rel_name = rel.name or rel.id
            for target in targets:
                if target.id is None or not target.id.startswith(keep_prefix):
                    continue
                rels.append(Relation(rel.id, rel_name, target.id))
        if rels:
            index[term.id] = rels
    return index
```

- [ ] **Step 5: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_agentic_relations.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/foodscholar/layer_a/bakeoff/agentic/ tests/fixtures/mini_foodon_relations.obo tests/unit/test_agentic_relations.py
git commit -m "feat(agentic): throwaway FoodOn relation index (FOODON->FOODON, from OWL)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Support roll-up over the food-product subtree

**Files:**
- Create: `src/foodscholar/layer_a/bakeoff/agentic/support.py`
- Test: `tests/unit/test_agentic_support.py`

The agent needs, per FoodOn node, the set of chunks mentioning it or any descendant (so it can judge whether a tier is worth keeping). Mirrors notebook `1a+`'s `NODE_CHUNKS`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_agentic_support.py`:

```python
from pathlib import Path

from foodscholar.layer_a.bakeoff.agentic.support import rollup_support
from foodscholar.ontology import FoodOnAPI, load_ontology


def _mini() -> FoodOnAPI:
    return FoodOnAPI(load_ontology(Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"),
                     prefix_filter=None)


def test_rollup_support_aggregates_descendant_chunks():
    api = _mini()
    leaf_chunks = {"TEST:0000006": {"c1"}, "TEST:0000008": {"c2", "c3"}}  # apple, olive oil
    node_chunks = rollup_support(leaf_chunks, api, root="TEST:0000001")
    # fruit (0000004) is an ancestor of both apple and olive oil
    assert node_chunks["TEST:0000004"] == {"c1", "c2", "c3"}
    # plant food + food product roll up everything too
    assert node_chunks["TEST:0000001"] == {"c1", "c2", "c3"}
    # apple keeps only its own
    assert node_chunks["TEST:0000006"] == {"c1"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_agentic_support.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/foodscholar/layer_a/bakeoff/agentic/support.py`:

```python
"""Per-node chunk support over the food-product subtree (descendant roll-up)."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foodscholar.ontology import FoodOnAPI


def rollup_support(
    leaf_chunks: dict[str, set[str]], ontology: FoodOnAPI, *, root: str
) -> dict[str, set[str]]:
    """node id -> set of chunk ids mentioning it or any is-a descendant.

    Only nodes that are `root` or a subclass of `root` are rolled up onto."""
    node_chunks: dict[str, set[str]] = defaultdict(set)
    for leaf, chunk_ids in leaf_chunks.items():
        if leaf not in ontology:
            continue
        targets = [leaf] + [
            a for a in ontology.id_to_ancestors(leaf)
            if a == root or ontology.is_subclass_of(a, root)
        ]
        for node in targets:
            node_chunks[node].update(chunk_ids)
    return dict(node_chunks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_agentic_support.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_a/bakeoff/agentic/support.py tests/unit/test_agentic_support.py
git commit -m "feat(agentic): rollup_support (per-node descendant chunk support)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Read-only graph tools (the MCP surface)

**Files:**
- Create: `src/foodscholar/layer_a/bakeoff/agentic/tools.py`
- Test: `tests/unit/test_agentic_tools.py`

A read-only query layer the agent calls: children with support, non-is-a relation targets (for bridging), label/synonym lookup, and a lowest-common-ancestor util. Mutations (keep/collapse/reparent) are applied by the agent loop (Task 4), not here — keeping tools pure and testable.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_agentic_tools.py`:

```python
from pathlib import Path

from foodscholar.layer_a.bakeoff.agentic.relations import Relation
from foodscholar.layer_a.bakeoff.agentic.tools import GraphTools
from foodscholar.ontology import FoodOnAPI, load_ontology


def _mini() -> FoodOnAPI:
    return FoodOnAPI(load_ontology(Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"),
                     prefix_filter=None)


def _tools(api):
    node_support = {"TEST:0000004": 30, "TEST:0000005": 5, "TEST:0000006": 20}
    relation_index = {"TEST:0000006": [Relation("RO:0001000", "derives from", "TEST:0000004")]}
    return GraphTools(api, relation_index, node_support=node_support, min_support=10)


def test_supported_children_filters_by_min_support():
    api = _mini()
    tools = _tools(api)
    # plant food's children: fruit(30), vegetable(5), peanut(0) -> only fruit clears min 10
    kids = tools.supported_children("TEST:0000002")
    assert kids == ["TEST:0000004"]


def test_relation_targets_returns_foodon_bridges():
    api = _mini()
    tools = _tools(api)
    assert tools.relation_targets("TEST:0000006") == [
        ("RO:0001000", "derives from", "TEST:0000004")
    ]


def test_lowest_common_ancestor_of_apple_and_olive_is_fruit():
    api = _mini()
    tools = _tools(api)
    assert tools.lowest_common_ancestor(["TEST:0000006", "TEST:0000007"]) == "TEST:0000004"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_agentic_tools.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/foodscholar/layer_a/bakeoff/agentic/tools.py`:

```python
"""Read-only MCP-style graph tools the agent queries while building the tree."""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.layer_a.bakeoff.agentic.relations import Relation

if TYPE_CHECKING:
    from foodscholar.ontology import FoodOnAPI


class GraphTools:
    """Read-only queries over FoodOn + the relation index + node support."""

    def __init__(
        self,
        ontology: FoodOnAPI,
        relation_index: dict[str, list[Relation]],
        *,
        node_support: dict[str, int],
        min_support: int,
        retriever=None,
    ) -> None:
        self._o = ontology
        self._rel = relation_index
        self._support = node_support
        self._min = min_support
        self._retriever = retriever

    def support(self, fid: str) -> int:
        return self._support.get(fid, 0)

    def label(self, fid: str) -> str:
        return self._o.id_to_label(fid) or fid

    def supported_children(self, fid: str) -> list[str]:
        """Direct is-a children whose rolled-up support clears the floor,
        most-supported first."""
        kids = [
            c for c in self._o.id_to_children(fid)
            if c in self._o and self.support(c) >= self._min
        ]
        return sorted(kids, key=lambda c: -self.support(c))

    def relation_targets(self, fid: str) -> list[tuple[str, str, str]]:
        """Non-is-a FoodOn relation targets (rel_id, rel_name, target_id)."""
        return [(r.rel_id, r.rel_name, r.target_id) for r in self._rel.get(fid, [])]

    def search(self, query: str, *, k: int = 8) -> list[str]:
        """Find candidate FoodOn ids for a concept (retriever if given, else label search)."""
        if self._retriever is not None:
            return [c.id for c in self._retriever.retrieve(query, k=k)]
        return self._o.search(query, limit=k)

    def lowest_common_ancestor(self, ids: list[str]) -> str | None:
        """Deepest FoodOn node that is an ancestor-or-self of every id in `ids`."""
        if not ids:
            return None
        sets = []
        for fid in ids:
            sets.append({fid, *self._o.id_to_ancestors(fid)})
        common = set.intersection(*sets) if sets else set()
        if not common:
            return None
        return max(common, key=lambda a: len(self._o.id_to_ancestors(a)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_agentic_tools.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_a/bakeoff/agentic/tools.py tests/unit/test_agentic_tools.py
git commit -m "feat(agentic): read-only GraphTools (supported children, relation bridges, LCA)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: The agent loop → MethodResult

**Files:**
- Create: `src/foodscholar/layer_a/bakeoff/agentic/agent.py`
- Test: `tests/unit/test_agentic_agent.py`

The DFS editor. Starting from `root`, it visits each node, shows the LLM a **lens** (node label, parent label, supported children with labels+support, sample relation bridges), and applies the returned action over real edges, with guards. Emits a `MethodResult` (so it plugs straight into the scorecard). `home_edge_type` records `is-a` for is-a placements and `other-relation` for relation-bridged ones; the agent never fabricates.

Action protocol (manual, since no native tool-calling) — the LLM returns:
`{"action": "KEEP"|"COLLAPSE"|"REPARENT", "reason": "..."}` for the current node.
- `KEEP`: node becomes a shelf under its current parent; recurse into supported children (subject to depth/fan-out caps).
- `COLLAPSE`: node is redundant; its supported children attach to its parent instead (skip the node as a shelf).
- `REPARENT`: node is an organizational umbrella; same tree effect as COLLAPSE for this prototype (children lift to parent) but logged distinctly in the audit.

Guards: stop recursing past `max_depth`; keep at most `max_children` highest-support children per node; only descend into children with support ≥ `min_support` (already enforced by `supported_children`).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_agentic_agent.py`:

```python
from pathlib import Path

from foodscholar.layer_a.bakeoff.agentic.agent import build_agentic_result
from foodscholar.ontology import FoodOnAPI, load_ontology


def _mini() -> FoodOnAPI:
    return FoodOnAPI(load_ontology(Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"),
                     prefix_filter=None)


class ScriptedLLM:
    """Returns a KEEP/COLLAPSE/REPARENT action per node, keyed by node label."""
    model_id = "scripted"

    def __init__(self, by_label):
        self._by_label = by_label
        self.calls = 0

    def generate(self, prompt, max_tokens=1024):
        return ""

    def generate_json(self, prompt, schema, max_tokens=1024):
        self.calls += 1
        # The lens prompt always names the current node as: NODE: <label>
        line = next(ln for ln in prompt.splitlines() if ln.startswith("NODE: "))
        label = line[len("NODE: "):].strip()
        return {"action": self._by_label.get(label, "KEEP"), "reason": "test"}


def test_agent_keeps_supported_tiers_and_homes_leaves_is_a():
    api = _mini()
    # apple+olive oil mentioned; fruit tier well-supported.
    leaf_chunks = {"TEST:0000006": {"c1"}, "TEST:0000008": {"c2", "c3"}}
    llm = ScriptedLLM({})  # default KEEP everywhere
    result = build_agentic_result(
        leaf_chunks, api, relation_index={}, llm=llm,
        root="TEST:0000001", min_support=1, max_depth=6, max_children=12,
    )
    assert result.name == "agentic"
    # every mentioned leaf is homed, all via is-a (no fabrication)
    assert set(result.leaf_home) == {"TEST:0000006", "TEST:0000008"}
    assert set(result.home_edge_type.values()) == {"is-a"}
    # the agent made at least one LLM call and recorded an audit trail
    assert result.llm_calls == llm.calls > 0
    assert result.audit


def test_agent_collapse_lifts_children_to_parent():
    api = _mini()
    leaf_chunks = {"TEST:0000006": {"c1"}}  # apple under fruit under plant food
    # COLLAPSE 'fruit' -> apple's nearest kept ancestor becomes 'plant food'
    llm = ScriptedLLM({"fruit": "COLLAPSE"})
    result = build_agentic_result(
        leaf_chunks, api, relation_index={}, llm=llm,
        root="TEST:0000001", min_support=1, max_depth=6, max_children=12,
    )
    # fruit is not a node in the tree (collapsed); apple still homed
    assert "TEST:0000004" not in result.edges  # fruit has no kept children edge
    assert "TEST:0000006" in result.leaf_home
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_agentic_agent.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/foodscholar/layer_a/bakeoff/agentic/agent.py`:

```python
"""The agentic DFS editor: LLM makes local KEEP/COLLAPSE/REPARENT decisions over
the real FoodOn support DAG, emitting a MethodResult for the bake-off scorecard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.layer_a.bakeoff.agentic.support import rollup_support
from foodscholar.layer_a.bakeoff.agentic.tools import GraphTools
from foodscholar.layer_a.bakeoff.result import MethodResult
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.ontology import FoodOnAPI

_log = get_logger("foodscholar.layer_a.bakeoff.agentic")

_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["KEEP", "COLLAPSE", "REPARENT"]},
        "reason": {"type": "string"},
    },
    "required": ["action"],
}


def _lens(tools: GraphTools, node: str, parent: str | None) -> str:
    kids = tools.supported_children(node)
    kid_lines = "\n".join(f"  - {tools.label(c)} ({tools.support(c)} chunks)" for c in kids)
    bridges = tools.relation_targets(node)[:5]
    bridge_lines = "\n".join(f"  - {rn} -> {tools.label(t)}" for _, rn, t in bridges)
    return (
        f"NODE: {tools.label(node)}\n"
        f"PARENT: {tools.label(parent) if parent else '(root)'}\n"
        f"SUPPORT: {tools.support(node)} chunks\n"
        f"CHILDREN:\n{kid_lines or '  (none)'}\n"
        f"RELATIONS:\n{bridge_lines or '  (none)'}\n\n"
        "You are curating a browsable food category tree from FoodOn. Decide this "
        "node's role:\n"
        "- KEEP: it's a recognizable food category worth a shelf.\n"
        "- COLLAPSE: it's redundant with its parent; lift its children up.\n"
        "- REPARENT: it's an organizational artifact; lift its children to the parent.\n"
        'Return JSON {"action": "KEEP|COLLAPSE|REPARENT", "reason": "..."}.'
    )


def build_agentic_result(
    leaf_chunks: dict[str, set[str]],
    ontology: FoodOnAPI,
    *,
    relation_index: dict,
    llm,
    root: str,
    min_support: int = 25,
    max_depth: int = 6,
    max_children: int = 12,
    retriever=None,
) -> MethodResult:
    """Run the DFS editor and return a MethodResult."""
    node_support = {n: len(cs) for n, cs in rollup_support(leaf_chunks, ontology, root=root).items()}
    tools = GraphTools(ontology, relation_index, node_support=node_support,
                       min_support=min_support, retriever=retriever)

    edges: dict[str, list[str]] = {}
    labels: dict[str, str] = {root: tools.label(root)}
    counts: dict[str, int] = {root: node_support.get(root, 0)}
    audit: list[dict] = []
    calls = [0]

    def ask(node: str, parent: str | None) -> str:
        calls[0] += 1
        try:
            obj = llm.generate_json(_lens(tools, node, parent), _ACTION_SCHEMA, max_tokens=256)
        except Exception as exc:  # defensive: a failed call defaults to KEEP
            _log.warning("agentic.action_failed", node=node, error=str(exc))
            obj = {"action": "KEEP"}
        action = (obj or {}).get("action", "KEEP")
        if action not in {"KEEP", "COLLAPSE", "REPARENT"}:
            action = "KEEP"
        audit.append({"node": node, "label": tools.label(node), "action": action,
                      "reason": (obj or {}).get("reason", "")})
        return action

    def visit(node: str, kept_parent: str, depth: int) -> None:
        # `kept_parent` is the nearest ancestor that is a real shelf.
        for child in tools.supported_children(node)[:max_children]:
            action = ask(child, node) if depth < max_depth else "KEEP"
            if action == "KEEP":
                edges.setdefault(kept_parent, []).append(child)
                labels[child] = tools.label(child)
                counts[child] = tools.support(child)
                if depth + 1 < max_depth:
                    visit(child, child, depth + 1)
            else:  # COLLAPSE / REPARENT: skip child as a shelf, lift its children
                if depth + 1 < max_depth:
                    visit(child, kept_parent, depth + 1)

    visit(root, root, 0)

    # Home each mentioned leaf to the deepest kept node that is an is-a ancestor-or-self.
    kept = {root, *(c for kids in edges.values() for c in kids)}
    leaf_home: dict[str, str] = {}
    home_edge_type: dict[str, str] = {}
    for leaf in leaf_chunks:
        if leaf not in ontology:
            continue
        cands = [a for a in ([leaf] + ontology.id_to_ancestors(leaf)) if a in kept]
        if cands:
            home = max(cands, key=lambda a: len(ontology.id_to_ancestors(a)))
            leaf_home[leaf] = home
            home_edge_type[leaf] = "is-a"
    return MethodResult(
        name="agentic", root=root, edges=edges, labels=labels, counts=counts,
        leaf_home=leaf_home, home_edge_type=home_edge_type,
        llm_calls=calls[0], audit=audit,
    )
```

> Scope note: `EXPAND` / relation-bridging (`other-relation` edges) is intentionally **not** wired into this first loop — it needs the relation index threaded into reparent-target selection and is the natural follow-up once the KEEP/COLLAPSE/REPARENT core is proven on the scorecard. The relation index + tools (`relation_targets`) are built and surfaced in the lens so the LLM sees bridges; acting on them is the next increment. Document this in the commit.

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_agentic_agent.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_a/bakeoff/agentic/agent.py tests/unit/test_agentic_agent.py
git commit -m "feat(agentic): DFS editor loop (KEEP/COLLAPSE/REPARENT) -> MethodResult

Relation bridges shown in the lens; acting on them (EXPAND/other-relation edges)
is the next increment once the core is scored.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `specificity` metric (fair comparison for deep vs flat trees)

**Files:**
- Modify: `src/foodscholar/layer_a/bakeoff/result.py` (add `home_distance` field + populate in both adapters)
- Modify: `src/foodscholar/layer_a/bakeoff/agentic/agent.py` (populate `home_distance`)
- Modify: `src/foodscholar/layer_a/bakeoff/metrics.py` (add `specificity`)
- Test: `tests/unit/test_bakeoff_metrics.py`

The Plan-A run showed `coverage` is ~1.0 for every method (any ancestor counts as "homed"), so it doesn't discriminate. `specificity` measures how *close* each leaf's home is to the leaf — the number of is-a steps between them. Low = placed specifically (good); high = dumped under a distant generic ancestor (bad). This is the metric that separates a flat blob from a well-tiered tree.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_bakeoff_metrics.py`:

```python
from foodscholar.layer_a.bakeoff.metrics import specificity


def test_specificity_mean_and_median_distance():
    r = MethodResult(
        name="toy", root="root", edges={"root": ["A"]},
        labels={"root": "Foods", "A": "Fruit"}, counts={},
        leaf_home={"x": "A", "y": "root"}, home_edge_type={"x": "is-a", "y": "is-a"},
    )
    r.home_distance = {"x": 1, "y": 3}  # x placed 1 step away, y dumped 3 steps up
    mean, med = specificity(r)
    assert mean == 2.0
    assert med == 2.0


def test_specificity_zero_when_no_distances():
    r = MethodResult(name="t", root="root", edges={}, labels={}, counts={},
                     leaf_home={}, home_edge_type={})
    assert specificity(r) == (0.0, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_metrics.py -k specificity -v`
Expected: FAIL — `MethodResult` has no `home_distance`, and `specificity` is undefined.

- [ ] **Step 3: Add `home_distance` to `MethodResult`**

In `src/foodscholar/layer_a/bakeoff/result.py`, add a field (after `home_edge_type`):

```python
    home_distance: dict[str, int] = field(default_factory=dict)  # leaf -> is-a steps to its home
```

Populate it in both adapters. In `from_children_map`, where `home` is found, also record distance:

```python
        if home is not None:
            leaf_home[leaf] = home
            home_edge_type[leaf] = "is-a"
            home_distance[leaf] = 0 if home == leaf else len(
                [a for a in ontology.id_to_ancestors(leaf)
                 if a == home or ontology.is_subclass_of(a, home)]
            )
```

Add `home_distance: dict[str, int] = {}` initialization near `leaf_home` in each adapter and pass `home_distance=home_distance` to the `MethodResult(...)` constructor. Do the same in `from_shelves` (compute distance for the is-a branch; for the `see_also`/fabricated branch use distance 1 when the leaf is a subclass of the anchor, else `len(ontology.id_to_ancestors(leaf))` as a "far" sentinel).

> Distance definition: number of the leaf's ancestors that are themselves descendants-or-equal of the home node — i.e. how many is-a steps from leaf up to home. `0` when the leaf *is* the home.

In `agent.py`, in `build_agentic_result`, populate it alongside `leaf_home`:

```python
            leaf_home[leaf] = home
            home_edge_type[leaf] = "is-a"
            home_distance[leaf] = 0 if home == leaf else len(
                [a for a in ontology.id_to_ancestors(leaf)
                 if a == home or ontology.is_subclass_of(a, home)]
            )
```

(add `home_distance: dict[str, int] = {}` and pass it to the `MethodResult(...)`).

- [ ] **Step 4: Implement `specificity`**

Add to `metrics.py`:

```python
def specificity(result: MethodResult) -> tuple[float, float]:
    """(mean, median) is-a distance from each homed leaf to its home node.
    Lower = leaves placed at specific categories; higher = dumped under generic
    ancestors. Complements coverage (which is ~1.0 for any bottom-up method)."""
    dists = [float(d) for d in result.home_distance.values()]
    if not dists:
        return 0.0, 0.0
    return float(statistics.mean(dists)), float(statistics.median(dists))
```

- [ ] **Step 5: Add specificity to the scorecard**

In `src/foodscholar/layer_a/bakeoff/scorecard.py`, add `"spec_mean"` to `_COLUMNS` (after `depth_max`) and in `build_scorecard` add:

```python
        spec_mean, _ = M.specificity(r)
```
and `"spec_mean": spec_mean,` to the row dict.

- [ ] **Step 6: Run tests**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_bakeoff_metrics.py tests/unit/test_bakeoff_result.py tests/unit/test_bakeoff_scorecard.py -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/foodscholar/layer_a/bakeoff/ tests/unit/test_bakeoff_metrics.py
git commit -m "feat(bakeoff): specificity metric (leaf->home is-a distance) + home_distance

Coverage was non-discriminating (~1.0 everywhere); specificity separates a flat
blob from a well-tiered tree. Populated by both adapters and the agentic method.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Wire the agentic column into the bake-off notebook

**Files:**
- Modify: `scripts/build_layer_a_method_bakeoff_nb.py`
- Output: `notebooks/layer_a_method_bakeoff.ipynb`

Add a GROQ-gated column (like `1b`) that builds the relation index, runs the agent, and appends its `MethodResult` to `RESULTS`. Read the script's `§0` cell first to confirm the `ROOT`/`fs`/`api`/`chunks`/`HAVE_GROQ`/`collect_leaf_chunks` names in scope.

- [ ] **Step 1: Add the agentic cell**

Insert a new `cells.append(code('''...'''))` block right before the grouping cell (search for `# The merged bottom-up + LLM grouping method`):

```python
cells.append(
    code(
        '''# Agentic MCP method (Plan B) — GROQ-gated, like 1b.
AGENTIC_RESULT = None
if not HAVE_GROQ:
    print("GROQ_API_KEY not set — skipping agentic method.")
else:
    from foodscholar.layer_a.grouping import collect_leaf_chunks
    from foodscholar.layer_a.bakeoff.agentic.relations import load_relation_index
    from foodscholar.layer_a.bakeoff.agentic.agent import build_agentic_result

    _leaf_chunks = collect_leaf_chunks(iter(chunks), api, facet="foods", min_link_confidence=0.0)
    _rel_index = load_relation_index(str(ROOT / "data/foodon.owl"))
    print(f"relation index: {len(_rel_index)} FOODON terms with non-is-a relations")
    AGENTIC_RESULT = build_agentic_result(
        _leaf_chunks, api, relation_index=_rel_index, llm=fs.llm,
        root=FOOD_PRODUCT, min_support=25, max_depth=6, max_children=12,
    )
    RESULTS.append(AGENTIC_RESULT)
    print(f"agentic: {len(AGENTIC_RESULT.edges)} internal nodes, "
          f"{AGENTIC_RESULT.llm_calls} llm calls, {len(AGENTIC_RESULT.leaf_home)} leaves homed")'''
    )
)
```

- [ ] **Step 2: Regenerate + verify the build**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python scripts/build_layer_a_method_bakeoff_nb.py`
Expected: writes `notebooks/layer_a_method_bakeoff.ipynb` with no error.
Run: `grep -c "build_agentic_result" notebooks/layer_a_method_bakeoff.ipynb`
Expected: `>= 1`.

> Full execution requires `GROQ_API_KEY` + the live corpus and is a **manual** check: set the key, run the notebook top-to-bottom, confirm the agentic column joins the scorecard with `llm_calls > 0` and a non-empty audit. Note in the commit whether executed.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_layer_a_method_bakeoff_nb.py notebooks/layer_a_method_bakeoff.ipynb
git commit -m "feat(agentic): agentic method as a GROQ-gated bake-off column

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Full-suite gate + brief update

**Files:**
- Modify: `docs/methods_layer_a_bakeoff_brief.md`

- [ ] **Step 1: Run the whole unit suite + lint**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit -q`
Expected: all PASS except the known pre-existing `test_layer_b_label.py::test_label_by_keywords_filters_ocr_codes_and_id_leakage` (unrelated; fails on `main` too — do not fix here).
Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m ruff check src/foodscholar/layer_a/bakeoff/`
Expected: clean.

- [ ] **Step 2: Update the brief**

In `docs/methods_layer_a_bakeoff_brief.md`, in the §2 methods table change the agentic row status from **to build** to **built (is-a core; relation-bridging next)**, and add a one-line note under §3 that the throwaway relation index is implemented (`derives from` / `member of` / `has ingredient` etc., FOODON→FOODON, ~8.2k terms). Add `specificity` to the §4 metric table (leaf→home is-a distance; the discriminating complement to coverage).

- [ ] **Step 3: Commit**

```bash
git add docs/methods_layer_a_bakeoff_brief.md
git commit -m "docs(agentic): brief reflects built agentic core, relation index, specificity metric

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage (brief §3):** relation index from OWL (Task 1) — keeps FOODON→FOODON only; support roll-up (Task 2); read-only MCP tools incl. relation bridges + LCA (Task 3); the DFS KEEP/COLLAPSE/REPARENT loop emitting a `MethodResult` (Task 4); the fair-comparison `specificity` metric (Task 5); notebook column (Task 6); gate + brief (Task 7). **Deliberately deferred:** acting on relation bridges (`EXPAND` / `other-relation` edges) and the `expand_scope` mutation — surfaced to the LLM in the lens but applied in a follow-up increment once the is-a core is scored; called out in Task 4. This keeps Plan B shippable and the agentic column comparable on the scorecard now.
- **Type consistency:** `MethodResult` gains `home_distance` (Task 5) consumed by `specificity`; `GraphTools(ontology, relation_index, *, node_support, min_support, retriever=None)` constructed identically in Task 3 tests and Task 4's `build_agentic_result`; `Relation(rel_id, rel_name, target_id)` used in Tasks 1/3; `build_agentic_result(...)` signature matches its Task 6 caller.
- **Risk:** Task 4's loop is the crux; the scripted-LLM tests cover KEEP and COLLAPSE on the real mini fixture. The `home_distance` definition (count of leaf-ancestors that are descendants-or-equal of the home) is used identically in both adapters and the agent — verify it matches when implementing Task 5.
- **Gate reminder:** Plan B is being built ahead of the scorecard verdict at the user's direction; its value is the agentic column it adds to that scorecard.
