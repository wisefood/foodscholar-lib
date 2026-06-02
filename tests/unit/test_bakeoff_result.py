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


from pathlib import Path

from foodscholar.layer_a.bakeoff.result import from_children_map
from foodscholar.ontology import FoodOnAPI, load_ontology


def _mini_foodon() -> FoodOnAPI:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"
    return FoodOnAPI(load_ontology(path), prefix_filter=None)


def test_from_children_map_homes_leaves_to_deepest_tree_ancestor():
    api = _mini_foodon()
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
    assert r.leaf_home["TEST:0000006"] == "TEST:0000002"  # apple under plant food
    assert r.leaf_home["TEST:0000011"] == "TEST:0000003"  # dairy under animal food
    assert r.home_edge_type["TEST:0000006"] == "is-a"


from foodscholar.io.graph import Shelf
from foodscholar.layer_a.bakeoff.result import from_shelves


def test_from_shelves_groups_mark_nonancestor_membership_fabricated():
    api = _mini_foodon()
    root = Shelf(shelf_id="facet:foods", label="Foods", facet="foods", depth=0)
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
    assert r.labels["TEST:0000004"] == "Fruits"              # display_label preferred
