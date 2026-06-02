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


from foodscholar.layer_a.bakeoff.metrics import findability, sample_query_leaves


def test_findability_clicks_from_root():
    r = _toy()  # depths: A1=2, A2=2, B=1
    out = findability(r, ["A1", "A2", "B"], k=2)
    assert out["median_clicks"] == 2.0
    assert out["p90_clicks"] == 2
    assert out["pct_within_k"] == 1.0
    assert out["pct_reachable"] == 1.0


def test_findability_unreachable_leaf_counts_against_reachable():
    r = _toy()
    out = findability(r, ["A1", "X"], k=2)
    assert out["pct_reachable"] == 0.5
    assert out["pct_within_k"] == 0.5


def test_sample_query_leaves_is_deterministic_and_stratified():
    freq = {"a": 100, "b": 50, "c": 1, "d": 1, "e": 1}
    s1 = sample_query_leaves(freq, n=4)
    s2 = sample_query_leaves(freq, n=4)
    assert s1 == s2
    assert "a" in s1 and "c" in s1
    assert len(s1) == 4


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
    assert reproducibility(a, b) == 1.0
    b.edges = {"root": ["A"], "A": ["A1"]}
    assert reproducibility(a, b) < 1.0


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
    assert score == 0.75


def test_nameability_zero_when_llm_raises():
    class Boom(_FakeLLM):
        def generate_json(self, prompt, schema, max_tokens=1024):
            raise RuntimeError("no llm")
    assert nameability(_toy(), Boom({}), sample=10) == 0.0
