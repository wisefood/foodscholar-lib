from pathlib import Path

from bakeoff.agentic.support import rollup_support
from foodscholar.ontology import FoodOnAPI, load_ontology


def _mini() -> FoodOnAPI:
    return FoodOnAPI(load_ontology(Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"),
                     prefix_filter=None)


def test_rollup_support_aggregates_descendant_chunks():
    api = _mini()
    leaf_chunks = {"TEST:0000006": {"c1"}, "TEST:0000008": {"c2", "c3"}}  # apple, olive oil
    node_chunks = rollup_support(leaf_chunks, api, root="TEST:0000001")
    assert node_chunks["TEST:0000004"] == {"c1", "c2", "c3"}  # fruit (ancestor of both)
    assert node_chunks["TEST:0000001"] == {"c1", "c2", "c3"}  # food product
    assert node_chunks["TEST:0000006"] == {"c1"}              # apple keeps only its own
