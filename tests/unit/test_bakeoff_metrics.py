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
    assert coverage(r, {"A1", "A2", "B", "X"}) == 0.75


def test_fan_out_max_and_median_over_internal_nodes():
    mx, med = fan_out(_toy())
    assert mx == 2
    assert med == 2.0


def test_tree_depth_max_and_median():
    mx, med = tree_depth(_toy())
    assert mx == 2          # A1/A2 at depth 2
    assert med == 2.0       # home-node depths {A1:2, A2:2, B:1} -> max 2, median 2.0
