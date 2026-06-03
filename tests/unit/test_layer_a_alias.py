"""Tests for the Layer A shelf aliasing pass."""

from pathlib import Path

from foodscholar.io.graph import Shelf
from foodscholar.layer_a.alias import alias_shelves
from foodscholar.ontology import FoodOnAPI, load_ontology


def _api() -> FoodOnAPI:
    return FoodOnAPI(
        load_ontology(Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"),
        prefix_filter=None,
    )


class ScriptedLLM:
    model_id = "scripted"

    def __init__(self, by_label):
        self._by_label = by_label
        self.calls = 0

    def generate(self, prompt, max_tokens=1024):  # pragma: no cover
        return ""

    def generate_json(self, prompt, schema, max_tokens=1024):
        self.calls += 1
        line = next(ln for ln in prompt.splitlines() if ln.startswith("NODE: "))
        label = line[len("NODE: "):].strip()
        return self._by_label.get(label, {"alias": None})


def _shelf(sid, label, *, facet="foods", depth=1, foodon_id=None, parent=None,
           display_label=None, chunk_count=0):
    return Shelf(shelf_id=sid, label=label, facet=facet, depth=depth,
                 foodon_id=foodon_id, parent_shelf_id=parent,
                 display_label=display_label, chunk_count=chunk_count)


def _shelves():
    return [
        _shelf("facet:foods", "foods", depth=0),                       # synthetic root
        _shelf("s_fruit", "fruit", foodon_id="TEST:0000004",
               parent="facet:foods", chunk_count=40),
        _shelf("s_jargon", "vegetable corn food product", foodon_id="TEST:0000005",
               parent="facet:foods", chunk_count=34),
        _shelf("s_grouped", "raw label", foodon_id="TEST:0000006",
               parent="facet:foods", display_label="Apples", chunk_count=10),
    ]


def test_alias_sets_display_label_on_jargon_shelf_only():
    api = _api()
    shelves = _shelves()
    llm = ScriptedLLM({"vegetable corn food product": {"alias": "corn"}})
    alias_shelves(shelves, api, llm=llm)
    by = {s.shelf_id: s for s in shelves}
    assert by["s_jargon"].display_label == "corn"   # aliased
    assert by["s_jargon"].label == "vegetable corn food product"  # label untouched
    assert by["s_jargon"].foodon_id == "TEST:0000005"             # id untouched


def test_alias_skips_synthetic_root_and_already_named():
    api = _api()
    shelves = _shelves()
    # LLM would alias everything it's asked about — but root + grouped must be skipped.
    llm = ScriptedLLM({})
    alias_shelves(shelves, api, llm=llm)
    by = {s.shelf_id: s for s in shelves}
    assert by["facet:foods"].display_label is None       # synthetic root never aliased
    assert by["s_grouped"].display_label == "Apples"     # pre-existing name preserved
    # root (no foodon_id) and grouped (already named) were not even asked.
    assert llm.calls == 2  # only fruit + jargon shelf


def test_alias_equal_to_label_ignored():
    api = _api()
    shelves = _shelves()
    llm = ScriptedLLM({"fruit": {"alias": "fruit"}})  # same as label
    alias_shelves(shelves, api, llm=llm)
    by = {s.shelf_id: s for s in shelves}
    assert by["s_fruit"].display_label is None
