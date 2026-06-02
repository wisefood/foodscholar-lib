# Layer-A Bottom-Up Grouping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the prototype's "bottom-up + LLM semantic grouping" foods-facet construction into `src/foodscholar/layer_a/` as an opt-in path, producing flat, recognizable group shelves while leaving the existing top-down `prune()` path untouched.

**Architecture:** A new `layer_a/grouping.py` builds shelves for a facet by (1) collecting per-leaf chunk evidence, (2) proposing ~14 human food groups via the LLM (each resolved to a real FoodOn id), (3) assigning each leaf to a group by label via the LLM, (4) emitting one flat `Shelf` per group plus kept-leaf shelves for unassigned leaves. Each leaf's FoodOn id is recorded in its group shelf's `see_also`, so the existing `attach.py` resolver routes leaf chunks to group shelves with **no attach changes**. Activated per-facet by a new `bottom_up_grouping` config block; `_build_facet` branches on it. The LLM is threaded into `build_layer_a` (today it's only used post-attach).

**Tech Stack:** Python 3.11, pydantic v2, pytest. LLM via `LLMClient.generate_json` (Groq `llama-3.1-8b-instant`), mirroring `layer_a/semantic_consolidation` conventions. Tests run on in-memory stores with a fake LLM (no network).

---

## File Structure

- **Create** `src/foodscholar/layer_a/grouping.py` — the new builder: leaf evidence, group proposal, leaf→group assignment, shelves-from-groups. Pure functions + one orchestrator `build_grouped_shelves(...)`.
- **Modify** `src/foodscholar/io/graph.py` — add `display_label: str | None` to `Shelf`.
- **Modify** `src/foodscholar/config.py` — add `BottomUpGroupingConfig` and a `bottom_up_grouping` field on `FacetConfig` + `LayerAConfig`; expose via `resolve_facet`.
- **Modify** `src/foodscholar/layer_a/builder.py` — thread `llm` through `build_layer_a`/`build_shelves`/`_build_facet`; branch to grouping when enabled.
- **Modify** `src/foodscholar/facade.py` — pass `self.llm` into `build_layer_a`.
- **Create** `tests/unit/test_layer_a_grouping.py` — unit tests for the new module with a fake LLM.
- **Modify** `tests/unit/test_layer_a.py` — one regression test that the default (grouping disabled) path is unchanged.

A `Group` is an internal dataclass in `grouping.py` (not persisted directly); it becomes one `Shelf`.

---

## Conventions for the worker

- Run tests with the project env: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest`.
- `Shelf` is a pydantic `BaseModel` in `src/foodscholar/io/graph.py`.
- `FoodOnAPI` (`src/foodscholar/ontology`) methods used: `id_to_label(fid)->str|None`, `id_to_synonyms(fid, include_related=False)->list[str]`, `id_to_ancestors(fid)->list[str]`, `name_to_id(name)->str|None`, `search(q, limit=)->list[str]`, `is_subclass_of(child, anc)->bool`, `__contains__`.
- `shelf_id_for_foodon(term_id) -> "foodon:{term_id}"` is in `layer_a/prune.py` (import it).
- `route_link_to_facet(link) -> Facet|None` is in `layer_a/facet.py`.
- `Chunk` has `chunk_id: str`, `foodon_ids: list[str]`, `entity_links: list[EntityLink]` (each link has `ontology_id: str`).
- `LLMClient.generate_json(prompt, schema, max_tokens=1024) -> dict` — may raise; handle defensively.

---

### Task 1: Add `display_label` to the Shelf model

**Files:**
- Modify: `src/foodscholar/io/graph.py:25-35`
- Test: `tests/unit/test_layer_a_grouping.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_layer_a_grouping.py`:

```python
from foodscholar.io.graph import Shelf


def test_shelf_has_optional_display_label():
    s = Shelf(shelf_id="foodon:X", label="plant fruit food product", facet="foods", depth=1)
    assert s.display_label is None
    s2 = Shelf(
        shelf_id="foodon:X", label="plant fruit food product", facet="foods",
        depth=1, display_label="Fruits",
    )
    assert s2.display_label == "Fruits"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py::test_shelf_has_optional_display_label -v`
Expected: FAIL with `TypeError`/`ValidationError` (unexpected keyword `display_label`).

- [ ] **Step 3: Add the field**

In `src/foodscholar/io/graph.py`, add one line to `Shelf` (after `label`):

```python
class Shelf(BaseModel):
    shelf_id: ShelfId
    label: str
    display_label: str | None = None  # human-facing name for grouped shelves; None → use label
    facet: Facet
    depth: int
    foodon_id: str | None = None
    parent_shelf_id: ShelfId | None = None
    chunk_count: int = 0
    support_direct: int = 0
    support_lifted: int = 0
    see_also: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full layer_a suite to confirm no regression**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a.py -q`
Expected: all PASS (new optional field is backward-compatible).

- [ ] **Step 6: Commit**

```bash
git add src/foodscholar/io/graph.py tests/unit/test_layer_a_grouping.py
git commit -m "feat(layer_a): add optional Shelf.display_label for grouped shelves"
```

---

### Task 2: Add BottomUpGroupingConfig

**Files:**
- Modify: `src/foodscholar/config.py` (add class near `LayerAConfig`, add field to `FacetConfig` and `LayerAConfig`, extend `resolve_facet`)
- Test: `tests/unit/test_layer_a_grouping.py`

First read `src/foodscholar/config.py` around `LayerAConfig` (≈302-413), `FacetConfig`, and `resolve_facet` / `_ResolvedFacetConfig` to match the existing override pattern exactly.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_layer_a_grouping.py`:

```python
from foodscholar.config import BottomUpGroupingConfig, LayerAConfig


def test_bottom_up_grouping_defaults_disabled():
    cfg = LayerAConfig()
    resolved = cfg.resolve_facet("foods")
    assert resolved.bottom_up_grouping.enabled is False


def test_bottom_up_grouping_per_facet_override_enables():
    cfg = LayerAConfig(facet_overrides={"foods": {"bottom_up_grouping": {"enabled": True}}})
    assert cfg.resolve_facet("foods").bottom_up_grouping.enabled is True
    assert cfg.resolve_facet("health").bottom_up_grouping.enabled is False


def test_bottom_up_grouping_config_fields():
    c = BottomUpGroupingConfig(enabled=True)
    assert c.model == "llama-3.1-8b-instant"
    assert c.assign_batch_size == 60
    assert c.min_leaf_support == 1
    assert c.frozen_groups is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py -k bottom_up -v`
Expected: FAIL with `ImportError: cannot import name 'BottomUpGroupingConfig'`.

- [ ] **Step 3: Implement the config**

In `src/foodscholar/config.py`, add the class (place it just above `class LayerAConfig`):

```python
class FrozenGroup(BaseModel):
    """A pre-reviewed group: human display name + the real FoodOn anchor ids."""
    display_name: str
    anchor_foodon_ids: list[str] = Field(default_factory=list)


class BottomUpGroupingConfig(BaseModel):
    """Bottom-up + LLM-grouping foods construction (opt-in, per facet).

    When enabled for a facet, `build_grouped_shelves` replaces the top-down
    prune path: every corpus-mentioned leaf is kept (coverage), the LLM proposes
    ~`n_groups` human food groups anchored to real FoodOn ids, and each leaf is
    assigned to a group by label. If `frozen_groups` is set, the proposal step
    is skipped and that reviewed set is used (reproducible, no proposal LLM call).
    """
    enabled: bool = False
    model: str = "llama-3.1-8b-instant"
    n_groups: int = 14
    assign_batch_size: int = 60
    min_leaf_support: int = 1  # leaf must have >= this many chunks to be kept
    frozen_groups: list[FrozenGroup] | None = None
```

Add a field to `FacetConfig` (the per-facet override model — make it `| None` so "unset" merges from global):

```python
    bottom_up_grouping: BottomUpGroupingConfig | None = None
```

Add a field to `LayerAConfig` (the global default):

```python
    bottom_up_grouping: BottomUpGroupingConfig = Field(default_factory=BottomUpGroupingConfig)
```

In `_ResolvedFacetConfig`, add:

```python
    bottom_up_grouping: BottomUpGroupingConfig
```

In `resolve_facet(...)`, merge it like the other fields (facet override wins, else global):

```python
        bottom_up_grouping=(
            override.bottom_up_grouping
            if override is not None and override.bottom_up_grouping is not None
            else self.bottom_up_grouping
        ),
```

(Match the exact construction style already used in `resolve_facet` for the other fields.)

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py -k bottom_up -v`
Expected: PASS.

- [ ] **Step 5: Confirm no config/layer_a regressions**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a.py tests/unit/test_config.py -q`
Expected: all PASS. (If `test_config.py` doesn't exist, just run `test_layer_a.py`.)

- [ ] **Step 6: Commit**

```bash
git add src/foodscholar/config.py tests/unit/test_layer_a_grouping.py
git commit -m "feat(layer_a): add BottomUpGroupingConfig (opt-in, per-facet)"
```

---

### Task 3: Leaf evidence collection

**Files:**
- Create: `src/foodscholar/layer_a/grouping.py`
- Test: `tests/unit/test_layer_a_grouping.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_layer_a_grouping.py`:

```python
from foodscholar.layer_a.grouping import collect_leaf_chunks
from foodscholar.io.chunk import Chunk  # adjust import to the real Chunk location if different


def _chunk(cid, foodon_ids):
    return Chunk(chunk_id=cid, text="x", source_doc_id="d", source_type="s",
                 foodon_ids=foodon_ids)


def test_collect_leaf_chunks_counts_distinct_chunks(make_food_ontology):
    api = make_food_ontology  # fixture: ontology with apple, banana under food product
    chunks = [_chunk("c1", ["FOODON:apple"]), _chunk("c2", ["FOODON:apple", "FOODON:banana"])]
    leaf_chunks = collect_leaf_chunks(iter(chunks), api, facet="foods", min_link_confidence=0.0)
    assert leaf_chunks["FOODON:apple"] == {"c1", "c2"}
    assert leaf_chunks["FOODON:banana"] == {"c2"}
```

Add a fixture at the top of the test file (or in `conftest.py`) building a tiny `FoodOnAPI` with `food product → {apple, banana, fish}`. Use the same `OntologyTerm` construction the existing `tests/unit/test_layer_a.py` uses (read it for the exact pattern and `prefix_filter=None`).

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py::test_collect_leaf_chunks_counts_distinct_chunks -v`
Expected: FAIL with `ModuleNotFoundError: foodscholar.layer_a.grouping`.

- [ ] **Step 3: Implement `collect_leaf_chunks`**

Create `src/foodscholar/layer_a/grouping.py`:

```python
"""Bottom-up + LLM semantic grouping construction for a Layer-A facet.

Opt-in alternative to the top-down `prune` path (see methods_layer_a_rework_brief).
Every corpus-mentioned leaf is kept (coverage by construction); the LLM proposes
~N human food groups anchored to real FoodOn ids and assigns each leaf to a group
by label. Each group becomes one flat Shelf; a leaf's foodon_id is recorded on its
group shelf's `see_also` so the existing attach resolver routes its chunks there.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

from foodscholar.layer_a.facet import route_link_to_facet
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from foodscholar.io.chunk import Chunk
    from foodscholar.io.graph import Facet
    from foodscholar.ontology import FoodOnAPI

_log = get_logger("foodscholar.layer_a.grouping")


def collect_leaf_chunks(
    chunks: Iterable[Chunk],
    ontology: FoodOnAPI,
    *,
    facet: Facet,
    min_link_confidence: float,
) -> dict[str, set[str]]:
    """Map each mentioned FoodOn leaf id -> set of chunk ids.

    A term contributes when it is in the ontology and either appears in the
    chunk's `foodon_ids` denorm or in an `entity_link` routing to `facet` with
    confidence >= floor. Distinct chunk-id sets (not counts) so group sizes can
    be deduped as a union later.
    """
    leaf_chunks: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        seen: set[str] = set()
        for fid in (getattr(chunk, "foodon_ids", None) or []):
            if fid in ontology:
                seen.add(fid)
        for link in (getattr(chunk, "entity_links", None) or []):
            if link.confidence < min_link_confidence:
                continue
            if link.ontology_id in ontology and route_link_to_facet(link) == facet:
                seen.add(link.ontology_id)
        for fid in seen:
            leaf_chunks[fid].add(chunk.chunk_id)
    return dict(leaf_chunks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py::test_collect_leaf_chunks_counts_distinct_chunks -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_a/grouping.py tests/unit/test_layer_a_grouping.py
git commit -m "feat(layer_a): grouping.collect_leaf_chunks (bottom-up leaf evidence)"
```

---

### Task 4: Label cleaning via FoodOn synonyms

**Files:**
- Modify: `src/foodscholar/layer_a/grouping.py`
- Test: `tests/unit/test_layer_a_grouping.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from foodscholar.layer_a.grouping import clean_label


def test_clean_label_prefers_clean_synonym(make_food_ontology_with_synonyms):
    api = make_food_ontology_with_synonyms  # 'legume food product' has exact synonym 'legume'
    assert clean_label("FOODON:legume_fp", api) == "legume"


def test_clean_label_strips_food_product_suffix_when_no_synonym(make_food_ontology):
    api = make_food_ontology  # 'fish food product', no synonyms
    assert clean_label("FOODON:fish_fp", api) == "fish"
```

Extend the fixtures so one ontology has a term `FOODON:legume_fp` labelled `legume food product` with exact synonym `legume`, and `FOODON:fish_fp` labelled `fish food product` with no synonyms.

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py -k clean_label -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `clean_label`**

Add to `grouping.py`:

```python
_SYN_BAD = re.compile(r"\d|\(|,|;|:")  # codes / parenthetical / list-y synonyms


def _clean_synonym(fid: str, ontology: FoodOnAPI) -> str | None:
    base = re.sub(r"\s+food product$", "", (ontology.id_to_label(fid) or "")).lower()
    cands = [
        s for s in ontology.id_to_synonyms(fid, include_related=False)
        if s and not _SYN_BAD.search(s) and 2 <= len(s) <= 30
    ]
    cands.sort(key=len)
    for s in cands:
        if s.lower() != base:
            return s
    return cands[0] if cands else None


def clean_label(fid: str, ontology: FoodOnAPI) -> str:
    """Display label: clean FoodOn synonym -> strip ' food product' suffix -> raw label."""
    syn = _clean_synonym(fid, ontology)
    if syn:
        return syn
    lbl = ontology.id_to_label(fid) or fid
    return re.sub(r"\s+food product$", "", lbl)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py -k clean_label -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_a/grouping.py tests/unit/test_layer_a_grouping.py
git commit -m "feat(layer_a): grouping.clean_label (FoodOn synonym labels)"
```

---

### Task 5: Group proposal + FoodOn-id anchoring

**Files:**
- Modify: `src/foodscholar/layer_a/grouping.py`
- Test: `tests/unit/test_layer_a_grouping.py`

- [ ] **Step 1: Write the failing test (with a fake LLM)**

Append:

```python
from foodscholar.layer_a.grouping import propose_groups, Group


class FakeLLM:
    model_id = "fake"
    def __init__(self, responses): self._responses = list(responses)
    def generate(self, prompt, max_tokens=1024): return ""
    def generate_json(self, prompt, schema, max_tokens=1024):
        return self._responses.pop(0)


def test_propose_groups_resolves_names_to_real_foodon_ids(make_food_ontology):
    api = make_food_ontology  # has 'fruit'(FOODON:fruit), 'fish food product'(FOODON:fish_fp)
    llm = FakeLLM([{"groups": ["Fruits", "Fish and Seafood", "Nonexistent Xyz"]}])
    groups = propose_groups(api, llm, leaf_freq={}, n_groups=14)
    names = {g.display_name for g in groups}
    assert "Fruits" in names and "Fish and Seafood" in names
    # unresolvable name dropped (no real FoodOn anchor)
    assert "Nonexistent Xyz" not in names
    fruits = next(g for g in groups if g.display_name == "Fruits")
    assert all(fid in api for fid in fruits.anchor_foodon_ids)


def test_propose_groups_uses_frozen_when_provided(make_food_ontology):
    api = make_food_ontology
    from foodscholar.config import FrozenGroup
    frozen = [FrozenGroup(display_name="Fruits", anchor_foodon_ids=["FOODON:fruit"])]
    groups = propose_groups(api, FakeLLM([]), leaf_freq={}, n_groups=14, frozen=frozen)
    assert [g.display_name for g in groups] == ["Fruits"]
    assert groups[0].anchor_foodon_ids == ["FOODON:fruit"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py -k propose_groups -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `Group` + `propose_groups`**

Add to `grouping.py`:

```python
from dataclasses import dataclass, field


@dataclass
class Group:
    display_name: str
    anchor_foodon_ids: list[str] = field(default_factory=list)


def _split_concepts(group_name: str) -> list[str]:
    parts = re.split(r"\s+and\s+|\s*,\s*|\s*/\s*", group_name.strip())
    return [p.strip().lower() for p in parts if p.strip()]


def _anchor_for_concept(concept: str, ontology: FoodOnAPI) -> str | None:
    singular = concept.rstrip("s")
    for cand in (concept, singular, concept + " food product", singular + " food product"):
        fid = ontology.name_to_id(cand)
        if fid:
            return fid
    for hit in ontology.search(concept, limit=12):
        lbl = (ontology.id_to_label(hit) or "").lower()
        clean = re.sub(r"\s+food product$", "", lbl)
        if clean in {concept, singular} and "(" not in lbl:
            return hit
    return None


def propose_groups(
    ontology: FoodOnAPI,
    llm,
    *,
    leaf_freq: dict[str, int],
    n_groups: int,
    frozen=None,
) -> list[Group]:
    """Return groups (display name + real FoodOn anchor ids).

    If `frozen` (list of config.FrozenGroup) is given, use it verbatim (no LLM
    call). Otherwise ask the LLM for ~n_groups human food-group names and resolve
    each to real FoodOn anchor ids; names with no anchor are dropped.
    """
    if frozen is not None:
        return [Group(g.display_name, list(g.anchor_foodon_ids)) for g in frozen]

    schema = {"type": "object",
              "properties": {"groups": {"type": "array", "items": {"type": "string"}}},
              "required": ["groups"]}
    sample = ", ".join(
        ontology.id_to_label(fid) or fid
        for fid, _ in sorted(leaf_freq.items(), key=lambda kv: -kv[1])[:50]
    )
    prompt = (
        f"Propose {n_groups} intuitive, MUTUALLY-EXCLUSIVE top-level food groups for "
        f"browsing a nutrition knowledge base (human category names like 'Vegetables', "
        f"'Dairy and Eggs', 'Fish and Seafood', 'Grains and Bread'). Use food TYPES, "
        f"not cross-cutting attributes like 'processed foods'. Frequent corpus foods "
        f"for context:\n{sample}\n\nReturn JSON {{\"groups\": [\"...\"]}}."
    )
    try:
        names = (llm.generate_json(prompt, schema, max_tokens=400) or {}).get("groups", [])
    except Exception as exc:
        _log.warning("grouping.propose_failed", error=str(exc))
        names = []

    groups: list[Group] = []
    for nm in names:
        anchors = []
        for concept in _split_concepts(nm):
            fid = _anchor_for_concept(concept, ontology)
            if fid is not None and fid not in anchors:
                anchors.append(fid)
        if anchors:
            groups.append(Group(nm, anchors))
    return groups
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py -k propose_groups -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_a/grouping.py tests/unit/test_layer_a_grouping.py
git commit -m "feat(layer_a): grouping.propose_groups (LLM groups -> real FoodOn anchors)"
```

---

### Task 6: Leaf → group assignment (batched LLM, defensive)

**Files:**
- Modify: `src/foodscholar/layer_a/grouping.py`
- Test: `tests/unit/test_layer_a_grouping.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from foodscholar.layer_a.grouping import assign_leaves


def test_assign_leaves_maps_by_label(make_food_ontology):
    api = make_food_ontology  # apple, banana, fish_fp
    groups = [Group("Fruits", ["FOODON:fruit"]), Group("Fish and Seafood", ["FOODON:fish_fp"])]
    # LLM returns assignments keyed by the label we show it
    llm = FakeLLM([{"assignments": [
        {"food": "apple", "group": "Fruits"},
        {"food": "banana", "group": "Fruits"},
        {"food": "fish", "group": "Fish and Seafood"},
    ]}])
    leaf_ids = ["FOODON:apple", "FOODON:banana", "FOODON:fish_fp"]
    assignment = assign_leaves(leaf_ids, groups, api, llm, batch_size=60)
    assert assignment["FOODON:apple"] == "Fruits"
    assert assignment["FOODON:fish_fp"] == "Fish and Seafood"


def test_assign_leaves_handles_unknown_group_as_unassigned(make_food_ontology):
    api = make_food_ontology
    groups = [Group("Fruits", ["FOODON:fruit"])]
    llm = FakeLLM([{"assignments": [{"food": "apple", "group": "Bogus"}]}])
    assignment = assign_leaves(["FOODON:apple"], groups, api, llm, batch_size=60)
    assert assignment.get("FOODON:apple") is None  # invalid group dropped → unassigned
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py -k assign_leaves -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `assign_leaves`**

Add to `grouping.py`:

```python
def assign_leaves(
    leaf_ids: list[str],
    groups: list[Group],
    ontology: FoodOnAPI,
    llm,
    *,
    batch_size: int,
) -> dict[str, str | None]:
    """Assign each leaf id to a group display_name (or None) by LABEL via the LLM.

    Batched; defensive — invalid/missing group names map to None (unassigned →
    the leaf keeps its own shelf, preserving coverage). Assignment is by the
    leaf's clean label, NOT is-a ancestry.
    """
    group_names = [g.display_name for g in groups]
    valid = set(group_names)
    label_to_ids: dict[str, list[str]] = defaultdict(list)
    for fid in leaf_ids:
        label_to_ids[clean_label(fid, ontology)].append(fid)
    labels = sorted(label_to_ids)

    schema = {"type": "object", "properties": {"assignments": {"type": "array",
              "items": {"type": "object",
                        "properties": {"food": {"type": "string"}, "group": {"type": "string"}},
                        "required": ["food", "group"]}}}, "required": ["assignments"]}

    label_group: dict[str, str] = {}
    for i in range(0, len(labels), batch_size):
        batch = labels[i:i + batch_size]
        prompt = (
            f"Assign each food to ONE of these groups: {', '.join(group_names)}, "
            f"or '(other)' if none fits.\nFoods:\n"
            + "\n".join(f"  - {l}" for l in batch)
            + '\n\nReturn JSON {"assignments": [{"food": "<food>", "group": "<group>"}, ...]} for every food.'
        )
        try:
            obj = llm.generate_json(prompt, schema, max_tokens=4096)
        except Exception as exc:
            _log.warning("grouping.assign_failed", batch_start=i, error=str(exc))
            obj = {}
        for a in (obj or {}).get("assignments", []):
            food, group = a.get("food"), a.get("group")
            if food in label_to_ids and group in valid:
                label_group[food] = group

    return {fid: label_group.get(clean_label(fid, ontology)) for fid in leaf_ids}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py -k assign_leaves -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_a/grouping.py tests/unit/test_layer_a_grouping.py
git commit -m "feat(layer_a): grouping.assign_leaves (batched LLM label->group)"
```

---

### Task 7: Build shelves from groups (the orchestrator)

**Files:**
- Modify: `src/foodscholar/layer_a/grouping.py`
- Test: `tests/unit/test_layer_a_grouping.py`

This is the function `_build_facet` will call. It ties Tasks 3-6 together and emits `Shelf` records: one synthetic facet root, one shelf per non-empty group (with `display_label` + member leaf ids in `see_also`), and one kept-leaf shelf per unassigned leaf. Coverage invariant: every input leaf is reachable.

- [ ] **Step 1: Write the failing test**

Append:

```python
from foodscholar.layer_a.grouping import build_grouped_shelves
from foodscholar.config import BottomUpGroupingConfig


def test_build_grouped_shelves_emits_group_and_kept_leaf_shelves(make_food_ontology):
    api = make_food_ontology  # apple, banana, fish_fp, + an oddball leaf 'gizmo' with no group
    chunks = [
        _chunk("c1", ["FOODON:apple"]), _chunk("c2", ["FOODON:banana"]),
        _chunk("c3", ["FOODON:fish_fp"]), _chunk("c4", ["FOODON:gizmo"]),
    ]
    llm = FakeLLM([
        {"groups": ["Fruits", "Fish and Seafood"]},                  # propose
        {"assignments": [                                            # assign
            {"food": "apple", "group": "Fruits"},
            {"food": "banana", "group": "Fruits"},
            {"food": "fish", "group": "Fish and Seafood"},
        ]},
    ])
    cfg = BottomUpGroupingConfig(enabled=True)
    shelves = build_grouped_shelves(iter(chunks), api, cfg, facet="foods", min_link_confidence=0.0)

    by_disp = {s.display_label or s.label: s for s in shelves}
    # group shelves exist, displayed by group name
    assert "Fruits" in by_disp and by_disp["Fruits"].chunk_count == 2  # c1, c2 distinct
    assert "Fish and Seafood" in by_disp
    # the Fruits group's see_also lists its member leaf foodon ids (for attach)
    assert set(by_disp["Fruits"].see_also) >= {"FOODON:apple", "FOODON:banana"}
    # unassigned leaf kept as its own shelf (coverage)
    assert any(s.foodon_id == "FOODON:gizmo" for s in shelves)
    # exactly one facet root at depth 0
    roots = [s for s in shelves if s.parent_shelf_id is None]
    assert len(roots) == 1 and roots[0].shelf_id == "facet:foods"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py -k build_grouped_shelves -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `build_grouped_shelves`**

Add to `grouping.py` (imports `Shelf`, `shelf_id_for_foodon`):

```python
from foodscholar.io.graph import Shelf
from foodscholar.layer_a.prune import shelf_id_for_foodon

_FACET_ROOT_LABELS = {
    "foods": "Foods", "health": "Health", "sustainability": "Sustainability",
    "dietary_patterns": "Dietary patterns", "allergies": "Allergies", "nutrients": "Nutrients",
}


def build_grouped_shelves(
    chunks: Iterable[Chunk],
    ontology: FoodOnAPI,
    cfg,  # BottomUpGroupingConfig
    *,
    facet: Facet,
    min_link_confidence: float,
    llm=None,
) -> list[Shelf]:
    """Bottom-up + LLM-grouping shelves for one facet.

    Emits: 1 synthetic facet root (depth 0); one group shelf per non-empty group
    (depth 1, display_label set, member leaf foodon_ids in see_also); one kept-leaf
    shelf (depth 1) per leaf not assigned to any group. Every mentioned leaf is
    represented — coverage by construction.
    """
    leaf_chunks = collect_leaf_chunks(
        chunks, ontology, facet=facet, min_link_confidence=min_link_confidence
    )
    leaf_chunks = {fid: cs for fid, cs in leaf_chunks.items() if len(cs) >= cfg.min_leaf_support}
    if not leaf_chunks:
        from foodscholar.layer_a.facet import stub_root
        return [stub_root(facet)]

    leaf_freq = {fid: len(cs) for fid, cs in leaf_chunks.items()}
    groups = propose_groups(
        ontology, llm, leaf_freq=leaf_freq, n_groups=cfg.n_groups, frozen=cfg.frozen_groups
    )
    assignment = assign_leaves(
        list(leaf_chunks), groups, ontology, llm, batch_size=cfg.assign_batch_size
    ) if groups else {fid: None for fid in leaf_chunks}

    root_id = f"facet:{facet}"
    shelves: list[Shelf] = []
    all_chunks: set[str] = set()

    # group shelves
    group_members: dict[str, list[str]] = defaultdict(list)
    for fid, gname in assignment.items():
        if gname is not None:
            group_members[gname].append(fid)
    for g in groups:
        members = group_members.get(g.display_name, [])
        if not members:
            continue
        chunk_ids: set[str] = set()
        for fid in members:
            chunk_ids |= leaf_chunks.get(fid, set())
        all_chunks |= chunk_ids
        anchor = g.anchor_foodon_ids[0]
        shelves.append(Shelf(
            shelf_id=shelf_id_for_foodon(anchor),
            label=ontology.id_to_label(anchor) or anchor,
            display_label=g.display_name,
            facet=facet,
            depth=1,
            foodon_id=anchor,
            parent_shelf_id=root_id,
            chunk_count=len(chunk_ids),
            support_direct=0,
            support_lifted=len(chunk_ids),
            see_also=sorted(set(members)),  # member leaves -> attach routes here
        ))

    # kept-leaf shelves (unassigned)
    for fid, gname in assignment.items():
        if gname is not None:
            continue
        cs = leaf_chunks.get(fid, set())
        all_chunks |= cs
        shelves.append(Shelf(
            shelf_id=shelf_id_for_foodon(fid),
            label=ontology.id_to_label(fid) or fid,
            display_label=clean_label(fid, ontology),
            facet=facet,
            depth=1,
            foodon_id=fid,
            parent_shelf_id=root_id,
            chunk_count=len(cs),
            support_direct=len(cs),
            support_lifted=0,
            see_also=[],
        ))

    root = Shelf(
        shelf_id=root_id, label=_FACET_ROOT_LABELS[facet], facet=facet, depth=0,
        foodon_id=None, parent_shelf_id=None, chunk_count=len(all_chunks),
        support_direct=0, support_lifted=len(all_chunks), see_also=[],
    )
    return [root, *sorted(shelves, key=lambda s: (s.display_label or s.label).lower())]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py -k build_grouped_shelves -v`
Expected: PASS.

- [ ] **Step 5: Add a coverage-invariant test**

Append:

```python
def test_build_grouped_shelves_covers_every_leaf(make_food_ontology):
    api = make_food_ontology
    chunks = [_chunk("c1", ["FOODON:apple"]), _chunk("c2", ["FOODON:gizmo"])]
    llm = FakeLLM([{"groups": ["Fruits"]}, {"assignments": [{"food": "apple", "group": "Fruits"}]}])
    shelves = build_grouped_shelves(iter(chunks), api, BottomUpGroupingConfig(enabled=True),
                                    facet="foods", min_link_confidence=0.0, llm=llm)
    represented = {s.foodon_id for s in shelves if s.foodon_id} | {
        fid for s in shelves for fid in s.see_also
    }
    assert "FOODON:apple" in represented  # via group see_also
    assert "FOODON:gizmo" in represented  # via kept-leaf shelf
```

- [ ] **Step 6: Run + commit**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py -v`
Expected: all PASS.

```bash
git add src/foodscholar/layer_a/grouping.py tests/unit/test_layer_a_grouping.py
git commit -m "feat(layer_a): grouping.build_grouped_shelves (groups + kept-leaf shelves, coverage)"
```

---

### Task 8: Wire grouping into the builder (opt-in branch)

**Files:**
- Modify: `src/foodscholar/layer_a/builder.py:35-125`
- Test: `tests/unit/test_layer_a_grouping.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from foodscholar.layer_a.builder import build_shelves
from foodscholar.config import LayerAConfig


def test_build_shelves_uses_grouping_when_enabled(make_food_ontology, fake_chunk_store_factory):
    api = make_food_ontology
    store = fake_chunk_store_factory([_chunk("c1", ["FOODON:apple"])])  # in-memory ChunkStore
    cfg = LayerAConfig(
        facets=["foods"],
        facet_overrides={"foods": {"bottom_up_grouping": {"enabled": True}}},
    )
    llm = FakeLLM([{"groups": ["Fruits"]}, {"assignments": [{"food": "apple", "group": "Fruits"}]}])
    shelves = build_shelves(store, api, cfg, llm=llm)
    assert any((s.display_label or "") == "Fruits" for s in shelves)


def test_build_shelves_uses_prune_when_grouping_disabled(make_food_ontology, fake_chunk_store_factory):
    api = make_food_ontology
    store = fake_chunk_store_factory([_chunk("c1", ["FOODON:apple"])])
    cfg = LayerAConfig(facets=["foods"])  # grouping disabled by default
    shelves = build_shelves(store, api, cfg, llm=None)
    # old path: no display_label set on any shelf
    assert all(s.display_label is None for s in shelves)
```

Add a `fake_chunk_store_factory` fixture: a minimal object with `iter_chunks()` yielding one batch (a list of the chunks). Read `tests/unit/test_layer_a.py` for the existing in-memory store helper and reuse it if present.

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py -k "build_shelves_uses" -v`
Expected: FAIL — `build_shelves` doesn't accept `llm`, and no grouping branch.

- [ ] **Step 3: Modify `builder.py`**

Thread `llm` and branch in `_build_facet`. Update signatures:

```python
def build_shelves(
    chunk_store: ChunkStore,
    ontology: FoodOnAPI,
    config: LayerAConfig,
    *,
    llm: "LLMClient | None" = None,
) -> list[Shelf]:
    all_shelves: list[Shelf] = []
    for facet in config.facets:
        all_shelves.extend(_build_facet(chunk_store, ontology, config, facet, llm=llm))
    return sorted(all_shelves, key=lambda s: (s.facet, s.depth, s.label.lower(), s.shelf_id))
```

```python
def build_layer_a(
    chunk_store: ChunkStore,
    graph_store: GraphStore,
    ontology: FoodOnAPI,
    *,
    config: LayerAConfig,
    full_config: FoodScholarConfig,
    llm: "LLMClient | None" = None,
) -> ArtifactMeta:
    shelves = build_shelves(chunk_store, ontology, config, llm=llm)
    graph_store.clear_layer_a()
    graph_store.upsert_shelves(shelves)
    # ... rest unchanged ...
```

In `_build_facet`, branch before `collect_support`:

```python
def _build_facet(
    chunk_store: ChunkStore,
    ontology: FoodOnAPI,
    config: LayerAConfig,
    facet: Facet,
    *,
    llm: "LLMClient | None" = None,
) -> list[Shelf]:
    facet_config = config.resolve_facet(facet)

    def chunk_iter():
        for batch in chunk_store.iter_chunks():
            yield from batch

    if facet_config.bottom_up_grouping.enabled:
        from foodscholar.layer_a.grouping import build_grouped_shelves
        return build_grouped_shelves(
            chunk_iter(), ontology, facet_config.bottom_up_grouping,
            facet=facet, min_link_confidence=facet_config.min_link_confidence, llm=llm,
        )

    support = collect_support(
        chunk_iter(), ontology,
        min_link_confidence=facet_config.min_link_confidence,
        facet=facet, link_blocklist=facet_config.link_blocklist,
    )
    if not support:
        return [stub_root(facet)]
    shelves = prune(support, ontology, facet_config, facet)
    if not shelves:
        return [stub_root(facet)]
    return _ensure_single_root(shelves, facet, support)
```

Add `LLMClient` to the `TYPE_CHECKING` imports from `foodscholar.storage.protocols`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a_grouping.py -k "build_shelves_uses" -v`
Expected: PASS.

- [ ] **Step 5: Full layer_a regression**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a.py -q`
Expected: all PASS (old path untouched; `build_shelves`/`build_layer_a` gained an optional kwarg).

- [ ] **Step 6: Commit**

```bash
git add src/foodscholar/layer_a/builder.py tests/unit/test_layer_a_grouping.py
git commit -m "feat(layer_a): branch build to grouping path when enabled (opt-in)"
```

---

### Task 9: Pass the LLM from the facade

**Files:**
- Modify: `src/foodscholar/facade.py` (the `build_layer_a` wrapper method, ~line 895)
- Test: covered by existing facade/integration tests + manual

- [ ] **Step 1: Find the facade call site**

Read `src/foodscholar/facade.py` around line 895 (`def build_layer_a`). It currently calls the module `build_layer_a(...)` without `llm`.

- [ ] **Step 2: Pass `self.llm`**

Update the facade method's internal call to include `llm=self.llm`:

```python
        return build_layer_a(
            self.chunk_store, self.graph_store, api,
            config=self.config.layer_a, full_config=self.config,
            llm=self.llm,
        )
```

(Match the exact existing arg names/order in the file.)

- [ ] **Step 3: Run the facade-level layer_a tests**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_layer_a.py -q`
Expected: all PASS (grouping disabled by default, so `self.llm` is passed but unused).

- [ ] **Step 4: Commit**

```bash
git add src/foodscholar/facade.py
git commit -m "feat(layer_a): thread fs.llm into build_layer_a"
```

---

### Task 10: Full-suite gate + brief cross-check

**Files:** none (verification task)

- [ ] **Step 1: Run the whole unit suite**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit -q`
Expected: all PASS. If any pre-existing failures are unrelated to this change, note them but do not fix in this plan.

- [ ] **Step 2: Lint/type check (if configured)**

Run: `/mnt/miniconda3/envs/foodscholar/bin/python -m ruff check src/foodscholar/layer_a/grouping.py && /mnt/miniconda3/envs/foodscholar/bin/python -m mypy src/foodscholar/layer_a/grouping.py`
Expected: clean (or only pre-existing project-wide mypy noise). Fix issues in the new file.

- [ ] **Step 3: Confirm the attach path is untouched and compatible**

Re-read `src/foodscholar/layer_a/attach.py:167-223`. Confirm group shelves resolve: a leaf's `foodon_id` is in its group shelf's `see_also`, so `facet_idx.by_seealso[fid]` returns the group shelf. No code change needed — just verify the invariant holds by reading. Document the confirmation in the commit message.

- [ ] **Step 4: Commit (docs/state only if anything changed)**

```bash
git commit --allow-empty -m "test(layer_a): full-suite gate green for bottom-up grouping; attach compatible via see_also"
```

---

## Self-Review Notes

- **Spec coverage:** bottom-up coverage (Task 3 + Task 7 coverage test), LLM group proposal w/ FoodOn anchoring (Task 5), label-by-LLM assignment (Task 6), synonym labels (Task 4), distinct-chunk counts (Task 7), display labels (Task 1), opt-in per-facet (Task 2, 8), frozen-vs-live group set (Task 2 `frozen_groups` + Task 5/7 support both), attach compatibility via `see_also` (Task 7 design + Task 10 verify). All covered.
- **Not in scope (deliberate):** prompt-tuning iterations, the nameability-guard polish for leaked organizational kept-leaves, porting other facets, and the demo HTML — these are follow-ups noted in the brief, not blockers.
- **Type consistency:** `Group(display_name, anchor_foodon_ids)` used identically in Tasks 5-7; `BottomUpGroupingConfig`/`FrozenGroup` fields match across Tasks 2/5/7; `build_grouped_shelves(...)` signature matches its caller in Task 8.
- **Risk:** the real `Chunk` import path and the in-memory `ChunkStore` test helper must be confirmed against `tests/unit/test_layer_a.py` before Task 3/8 — the plan says to read it for the exact pattern.
