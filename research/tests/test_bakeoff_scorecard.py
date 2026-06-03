from bakeoff.result import MethodResult
from bakeoff.scorecard import build_scorecard, render_scorecard_markdown


def _r(name) -> MethodResult:
    return MethodResult(
        name=name, root="root",
        edges={"root": ["A", "B"], "A": ["A1"]},
        labels={"root": "Foods", "A": "Fruit", "A1": "Apple", "B": "Dairy"},
        counts={"root": 3, "A": 2, "A1": 2, "B": 1},
        leaf_home={"A1": "A1", "B": "B"},
        home_edge_type={"A1": "is-a", "B": "is-a"},
    )


def test_build_scorecard_one_row_per_method():
    rows = build_scorecard(
        [_r("1a"), _r("1a+")],
        mentioned_leaves={"A1", "B"},
        query_leaves=["A1", "B"],
        k=2,
        llm=None,
    )
    assert [row["method"] for row in rows] == ["1a", "1a+"]
    assert rows[0]["coverage"] == 1.0
    assert rows[0]["faithfulness_is_a"] == 1.0
    assert "nameability" in rows[0]
    assert rows[0]["nameability"] is None


def test_render_scorecard_markdown_has_header_and_rows():
    rows = build_scorecard([_r("1a")], mentioned_leaves={"A1", "B"},
                           query_leaves=["A1"], k=2, llm=None)
    md = render_scorecard_markdown(rows)
    assert "| method |" in md
    assert "1a" in md
