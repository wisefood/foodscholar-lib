from pathlib import Path

from foodscholar.config import BottomUpGroupingConfig, LayerAConfig
from foodscholar.io.chunk import Chunk
from foodscholar.io.graph import Shelf
from foodscholar.layer_a.grouping import collect_leaf_chunks
from foodscholar.ontology import FoodOnAPI, load_ontology


def _mini_foodon() -> FoodOnAPI:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"
    return FoodOnAPI(load_ontology(path), prefix_filter=None)


def _chunk(chunk_id: str, foodon_ids: list[str]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="x",
        source_doc_id="d",
        source_type="abstract",
        section_type="abstract",
        foodon_ids=foodon_ids,
    )


def test_shelf_has_optional_display_label():
    s = Shelf(shelf_id="foodon:X", label="plant fruit food product", facet="foods", depth=1)
    assert s.display_label is None
    s2 = Shelf(
        shelf_id="foodon:X", label="plant fruit food product", facet="foods",
        depth=1, display_label="Fruits",
    )
    assert s2.display_label == "Fruits"


def test_bottom_up_grouping_defaults_disabled():
    cfg = LayerAConfig()
    resolved = cfg.resolve_facet("foods")
    assert resolved.bottom_up_grouping.enabled is False


def test_bottom_up_grouping_per_facet_override_enables():
    cfg = LayerAConfig(facet_overrides={"foods": {"bottom_up_grouping": {"enabled": True}}})
    assert cfg.resolve_facet("foods").bottom_up_grouping.enabled is True
    assert cfg.resolve_facet("health").bottom_up_grouping.enabled is False


def test_bottom_up_grouping_config_fields():
    c = BottomUpGroupingConfig(enabled=True)
    assert c.model == "llama-3.1-8b-instant"
    assert c.assign_batch_size == 60
    assert c.min_leaf_support == 1
    assert c.frozen_groups is None


def test_collect_leaf_chunks_counts_distinct_chunks():
    api = _mini_foodon()
    chunks = [_chunk("c1", ["TEST:0000006"]),
              _chunk("c2", ["TEST:0000006", "TEST:0000008"])]
    leaf_chunks = collect_leaf_chunks(iter(chunks), api, facet="foods", min_link_confidence=0.0)
    assert leaf_chunks["TEST:0000006"] == {"c1", "c2"}
    assert leaf_chunks["TEST:0000008"] == {"c2"}


from foodscholar.layer_a.grouping import clean_label


def test_clean_label_uses_short_clean_synonym():
    api = _mini_foodon()
    # olive oil (TEST:0000008) has exact synonyms "extra-virgin olive oil","EVOO";
    # _clean_synonym prefers the shortest clean one that differs from the base label.
    # "EVOO" (4 chars, no digits/parens/commas) qualifies.
    assert clean_label("TEST:0000008", api) == "EVOO"


def test_clean_label_falls_back_to_label_when_no_clean_synonym():
    api = _mini_foodon()
    # vegetable (TEST:0000005) has no synonyms -> raw label
    assert clean_label("TEST:0000005", api) == "vegetable"


def test_clean_label_strips_food_product_suffix_when_no_synonym():
    api = _mini_foodon()
    # food product (TEST:0000001), no synonyms; suffix-strip leaves "food product"
    # unchanged only if it equals the suffix — here label IS "food product" so it
    # strips to "" then we keep raw. Assert it returns a non-empty string (the raw label).
    assert clean_label("TEST:0000001", api) == "food product"


# ---------------------------------------------------------------------------
# Shared test double (reused by Tasks 6-7)
# ---------------------------------------------------------------------------

class FakeLLM:
    model_id = "fake"

    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, prompt, max_tokens=1024):
        return ""

    def generate_json(self, prompt, schema, max_tokens=1024):
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Task 5 — propose_groups
# ---------------------------------------------------------------------------

from foodscholar.layer_a.grouping import propose_groups, Group
from foodscholar.config import FrozenGroup


def test_propose_groups_resolves_names_to_real_foodon_ids():
    api = _mini_foodon()
    llm = FakeLLM([{"groups": ["Fruit", "Vegetable", "Nonexistent Xyz"]}])
    groups = propose_groups(api, llm, leaf_freq={}, n_groups=14)
    names = {g.display_name for g in groups}
    assert "Fruit" in names and "Vegetable" in names
    assert "Nonexistent Xyz" not in names      # unresolvable -> dropped
    fruit = next(g for g in groups if g.display_name == "Fruit")
    assert all(fid in api for fid in fruit.anchor_foodon_ids)
    assert "TEST:0000004" in fruit.anchor_foodon_ids


def test_propose_groups_uses_frozen_when_provided():
    api = _mini_foodon()
    frozen = [FrozenGroup(display_name="Fruits", anchor_foodon_ids=["TEST:0000004"])]
    groups = propose_groups(api, FakeLLM([]), leaf_freq={}, n_groups=14, frozen=frozen)
    assert [g.display_name for g in groups] == ["Fruits"]
    assert groups[0].anchor_foodon_ids == ["TEST:0000004"]
