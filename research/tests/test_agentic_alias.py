"""Tests for the aliasing-only pass (build_aliased_result).

The agentic method adds aliases and changes NOTHING else: structure and chunk
homing are copied verbatim from the backbone (e.g. 1a+). It never reparents a
node or a chunk.
"""

from pathlib import Path

from bakeoff.agentic.alias import build_aliased_result
from bakeoff.agentic.support import rollup_support
from bakeoff.agentic.tools import GraphTools
from bakeoff.result import MethodResult, from_children_map
from foodscholar.ontology import FoodOnAPI, load_ontology

ROOT = "TEST:0000001"        # food product
PLANT = "TEST:0000002"       # plant food
FRUIT = "TEST:0000004"       # fruit
APPLE = "TEST:0000006"       # apple
OLIVE = "TEST:0000007"       # olive
OLIVE_OIL = "TEST:0000008"   # olive oil
ANIMAL = "TEST:0000003"      # animal food
DAIRY = "TEST:0000011"       # dairy product

CHILDREN = {
    ROOT: [PLANT, ANIMAL],
    PLANT: [FRUIT],
    FRUIT: [APPLE, OLIVE],
    OLIVE: [OLIVE_OIL],
    ANIMAL: [DAIRY],
}


def _api() -> FoodOnAPI:
    return FoodOnAPI(
        load_ontology(Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"),
        prefix_filter=None,
    )


class ScriptedLLM:
    """Returns {alias} per node keyed by the NODE label in the prompt."""

    model_id = "scripted"

    def __init__(self, by_label: dict[str, dict]):
        self._by_label = by_label
        self.calls = 0

    def generate(self, prompt, max_tokens=1024):  # pragma: no cover - unused
        return ""

    def generate_json(self, prompt, schema, max_tokens=1024):
        self.calls += 1
        line = next(ln for ln in prompt.splitlines() if ln.startswith("NODE: "))
        label = line[len("NODE: "):].strip()
        return self._by_label.get(label, {"alias": None})


def _base(api: FoodOnAPI, leaf_chunks: dict[str, set[str]]) -> MethodResult:
    node_chunks = rollup_support(leaf_chunks, api, root=ROOT)
    all_nodes = {ROOT, *CHILDREN, *(c for kids in CHILDREN.values() for c in kids)}
    labels = {nid: api.id_to_label(nid) or nid for nid in all_nodes}
    counts = {nid: len(node_chunks.get(nid, set())) for nid in labels}
    return from_children_map(
        "1a+", root=ROOT, children_map=CHILDREN, counts=counts, labels=labels,
        ontology=api, mentioned_leaves=set(leaf_chunks),
    )


def _tools(api: FoodOnAPI, leaf_chunks: dict[str, set[str]], relation_index=None) -> GraphTools:
    support = {n: len(cs) for n, cs in rollup_support(leaf_chunks, api, root=ROOT).items()}
    return GraphTools(api, relation_index or {}, node_support=support, min_support=1)


def test_structure_and_homing_are_copied_verbatim():
    api = _api()
    leaf_chunks = {APPLE: {"c1"}, OLIVE_OIL: {"c2", "c3"}, DAIRY: {"c4"}}
    base = _base(api, leaf_chunks)
    res = build_aliased_result(base, tools=_tools(api, leaf_chunks), llm=ScriptedLLM({}))
    assert res.name == "agentic"
    # Nothing about the backbone moved — structure AND chunk homing are identical.
    assert res.edges == base.edges
    assert res.labels == base.labels
    assert res.leaf_home == base.leaf_home
    assert res.home_edge_type == base.home_edge_type
    assert res.home_distance == base.home_distance
    assert res.counts == base.counts
    assert res.aliases == {}  # no aliases proposed


def test_alias_is_additive_and_drives_display_without_moving_anything():
    api = _api()
    leaf_chunks = {APPLE: {"c1"}, DAIRY: {"c4"}}
    base = _base(api, leaf_chunks)
    llm = ScriptedLLM({
        "fruit": {"alias": "fruits"},
        "dairy product": {"alias": "milk & cheese"},
    })
    res = build_aliased_result(base, tools=_tools(api, leaf_chunks), llm=llm)
    assert res.aliases[FRUIT] == "fruits"
    assert res.labels[FRUIT] == "fruit"        # original label untouched
    assert res.display(FRUIT) == "fruits"      # browse label = alias
    assert res.display(APPLE) == "apple"       # un-aliased node falls back to label
    # homing still identical to the backbone — aliasing never reparents.
    assert res.leaf_home == base.leaf_home


def test_alias_equal_to_label_is_ignored():
    api = _api()
    leaf_chunks = {APPLE: {"c1"}}
    base = _base(api, leaf_chunks)
    llm = ScriptedLLM({"apple": {"alias": "apple"}})  # same as label -> no-op
    res = build_aliased_result(base, tools=_tools(api, leaf_chunks), llm=llm)
    assert APPLE not in res.aliases
    assert res.display(APPLE) == "apple"


def test_relations_appear_in_lens_for_naming_context():
    api = _api()
    from bakeoff.agentic.relations import Relation
    leaf_chunks = {OLIVE_OIL: {"c2"}}
    base = _base(api, leaf_chunks)
    rel_index = {OLIVE_OIL: [Relation("RO:0001000", "derives from", OLIVE)]}
    seen_prompts = []

    class CapturingLLM(ScriptedLLM):
        def generate_json(self, prompt, schema, max_tokens=1024):
            seen_prompts.append(prompt)
            return super().generate_json(prompt, schema, max_tokens)

    build_aliased_result(base, tools=_tools(api, leaf_chunks, rel_index), llm=CapturingLLM({}))
    oil_prompt = next(p for p in seen_prompts if p.startswith("NODE: olive oil"))
    assert "derives from -> olive" in oil_prompt  # relation surfaced to inform naming
