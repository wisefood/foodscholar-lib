from pathlib import Path

from bakeoff.agentic.relations import Relation
from bakeoff.agentic.tools import GraphTools
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
    kids = tools.supported_children("TEST:0000002")  # plant food: fruit(30),veg(5),peanut(0)
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
