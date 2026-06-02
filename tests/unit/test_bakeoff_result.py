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
