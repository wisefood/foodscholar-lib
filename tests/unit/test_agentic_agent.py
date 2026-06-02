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
        line = next(ln for ln in prompt.splitlines() if ln.startswith("NODE: "))
        label = line[len("NODE: "):].strip()
        return {"action": self._by_label.get(label, "KEEP"), "reason": "test"}


def test_agent_keeps_supported_tiers_and_homes_leaves_is_a():
    api = _mini()
    leaf_chunks = {"TEST:0000006": {"c1"}, "TEST:0000008": {"c2", "c3"}}
    llm = ScriptedLLM({})  # default KEEP everywhere
    result = build_agentic_result(
        leaf_chunks, api, relation_index={}, llm=llm,
        root="TEST:0000001", min_support=1, max_depth=6, max_children=12,
    )
    assert result.name == "agentic"
    assert set(result.leaf_home) == {"TEST:0000006", "TEST:0000008"}
    assert set(result.home_edge_type.values()) == {"is-a"}
    assert result.llm_calls == llm.calls > 0
    assert result.audit


def test_agent_collapse_lifts_children_to_parent():
    api = _mini()
    leaf_chunks = {"TEST:0000006": {"c1"}}  # apple under fruit under plant food
    llm = ScriptedLLM({"fruit": "COLLAPSE"})
    result = build_agentic_result(
        leaf_chunks, api, relation_index={}, llm=llm,
        root="TEST:0000001", min_support=1, max_depth=6, max_children=12,
    )
    assert "TEST:0000004" not in result.edges  # fruit collapsed: not a parent with kept children
    assert "TEST:0000006" in result.leaf_home
