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


# ---------------------------------------------------------------------------
# Task 6 — assign_leaves
# ---------------------------------------------------------------------------

from foodscholar.layer_a.grouping import assign_leaves, clean_label as _cl


def test_assign_leaves_maps_by_label():
    api = _mini_foodon()
    groups = [Group("Fruits", ["TEST:0000004"]), Group("Vegetables", ["TEST:0000005"])]
    leaf_ids = ["TEST:0000006", "TEST:0000005"]  # apple, vegetable
    # Build the LLM response keyed by each leaf's clean_label (robust to synonyms)
    resp = {"assignments": [
        {"food": _cl("TEST:0000006", api), "group": "Fruits"},
        {"food": _cl("TEST:0000005", api), "group": "Vegetables"},
    ]}
    llm = FakeLLM([resp])
    assignment = assign_leaves(leaf_ids, groups, api, llm, batch_size=60)
    assert assignment["TEST:0000006"] == "Fruits"
    assert assignment["TEST:0000005"] == "Vegetables"


def test_assign_leaves_handles_unknown_group_as_unassigned():
    api = _mini_foodon()
    groups = [Group("Fruits", ["TEST:0000004"])]
    resp = {"assignments": [{"food": _cl("TEST:0000006", api), "group": "Bogus"}]}
    llm = FakeLLM([resp])
    assignment = assign_leaves(["TEST:0000006"], groups, api, llm, batch_size=60)
    assert assignment.get("TEST:0000006") is None  # invalid group -> unassigned


def test_assign_leaves_unmentioned_leaf_is_unassigned():
    api = _mini_foodon()
    groups = [Group("Fruits", ["TEST:0000004"])]
    llm = FakeLLM([{"assignments": []}])  # LLM returns nothing
    assignment = assign_leaves(["TEST:0000006"], groups, api, llm, batch_size=60)
    assert assignment.get("TEST:0000006") is None


# ---------------------------------------------------------------------------
# Task 7 — build_grouped_shelves
# ---------------------------------------------------------------------------

from foodscholar.layer_a.grouping import build_grouped_shelves


def test_build_grouped_shelves_emits_group_and_kept_leaf_shelves():
    api = _mini_foodon()
    # apple(0006), olive oil(0008) -> Fruits group; peanut(0009) -> unassigned (kept leaf)
    chunks = [
        _chunk("c1", ["TEST:0000006"]),  # apple
        _chunk("c2", ["TEST:0000008"]),  # olive oil
        _chunk("c3", ["TEST:0000009"]),  # peanut (no group)
    ]
    propose_resp = {"groups": ["Fruit"]}  # resolves to TEST:0000004 (fruit)
    assign_resp = {"assignments": [
        {"food": _cl("TEST:0000006", api), "group": "Fruit"},
        {"food": _cl("TEST:0000008", api), "group": "Fruit"},
        # peanut intentionally omitted -> unassigned
    ]}
    llm = FakeLLM([propose_resp, assign_resp])
    cfg = BottomUpGroupingConfig(enabled=True)
    shelves = build_grouped_shelves(iter(chunks), api, cfg, facet="foods",
                                    min_link_confidence=0.0, llm=llm)

    by_disp = {(s.display_label or s.label): s for s in shelves}
    assert "Fruit" in by_disp
    assert by_disp["Fruit"].chunk_count == 2          # c1, c2 distinct
    assert by_disp["Fruit"].foodon_id == "TEST:0000004"
    assert set(by_disp["Fruit"].see_also) >= {"TEST:0000006", "TEST:0000008"}
    # unassigned peanut kept as its own shelf
    assert any(s.foodon_id == "TEST:0000009" for s in shelves)
    # exactly one facet root at depth 0
    roots = [s for s in shelves if s.parent_shelf_id is None]
    assert len(roots) == 1 and roots[0].shelf_id == "facet:foods" and roots[0].depth == 0


def test_build_grouped_shelves_covers_every_leaf():
    api = _mini_foodon()
    chunks = [_chunk("c1", ["TEST:0000006"]), _chunk("c2", ["TEST:0000009"])]  # apple, peanut
    llm = FakeLLM([
        {"groups": ["Fruit"]},
        {"assignments": [{"food": _cl("TEST:0000006", api), "group": "Fruit"}]},
    ])
    shelves = build_grouped_shelves(iter(chunks), api, BottomUpGroupingConfig(enabled=True),
                                    facet="foods", min_link_confidence=0.0, llm=llm)
    represented = {s.foodon_id for s in shelves if s.foodon_id} | {
        fid for s in shelves for fid in s.see_also
    }
    assert "TEST:0000006" in represented  # via group see_also
    assert "TEST:0000009" in represented  # via kept-leaf shelf


def test_build_grouped_shelves_empty_returns_stub_root():
    api = _mini_foodon()
    shelves = build_grouped_shelves(iter([]), api, BottomUpGroupingConfig(enabled=True),
                                    facet="foods", min_link_confidence=0.0, llm=FakeLLM([]))
    assert len(shelves) == 1  # stub root only


def test_build_grouped_shelves_no_shelf_id_collision_when_anchor_is_also_unassigned_leaf():
    api = _mini_foodon()
    # A chunk links 'fruit' (TEST:0000004) DIRECTLY — and 'fruit' is also the
    # anchor the "Fruit" group resolves to. apple is assigned to Fruit; the
    # direct 'fruit' leaf is left UNASSIGNED by the LLM. The group shelf must win
    # for the anchor id (no duplicate shelf_id), and the direct 'fruit' chunk is
    # folded into the group shelf so coverage holds.
    chunks = [_chunk("c1", ["TEST:0000006"]),   # apple -> assigned to Fruit
              _chunk("c2", ["TEST:0000004"])]   # fruit (direct) -> unassigned
    llm = FakeLLM([
        {"groups": ["Fruit"]},  # anchors to TEST:0000004
        {"assignments": [{"food": _cl("TEST:0000006", api), "group": "Fruit"}]},  # fruit-leaf omitted
    ])
    shelves = build_grouped_shelves(iter(chunks), api, BottomUpGroupingConfig(enabled=True),
                                    facet="foods", min_link_confidence=0.0, llm=llm)
    ids = [s.shelf_id for s in shelves]
    assert len(ids) == len(set(ids)), f"duplicate shelf_id(s): {ids}"
    fruit_shelves = [s for s in shelves if s.foodon_id == "TEST:0000004"]
    assert len(fruit_shelves) == 1
    assert fruit_shelves[0].display_label == "Fruit"
    # the directly-linked 'fruit' chunk (c2) is absorbed into the group shelf
    assert fruit_shelves[0].chunk_count == 2  # c1 (apple) + c2 (fruit direct)


# ---------------------------------------------------------------------------
# Task 8 — build_shelves grouping branch
# ---------------------------------------------------------------------------

from foodscholar.layer_a.builder import build_shelves
from foodscholar.config import LayerAConfig
from foodscholar.storage.memory import InMemoryChunkStore


def _store_with(chunks):
    store = InMemoryChunkStore()
    store.upsert(chunks)
    return store


def test_build_shelves_uses_grouping_when_enabled():
    api = _mini_foodon()
    store = _store_with([_chunk("c1", ["TEST:0000006"])])  # apple
    cfg = LayerAConfig(
        facets=["foods"],
        facet_overrides={"foods": {"bottom_up_grouping": {"enabled": True}}},
    )
    llm = FakeLLM([
        {"groups": ["Fruit"]},
        {"assignments": [{"food": _cl("TEST:0000006", api), "group": "Fruit"}]},
    ])
    shelves = build_shelves(store, api, cfg, llm=llm)
    assert any((s.display_label or "") == "Fruit" for s in shelves)


def test_build_shelves_uses_prune_when_grouping_disabled():
    api = _mini_foodon()
    store = _store_with([_chunk("c1", ["TEST:0000006"])])
    cfg = LayerAConfig(facets=["foods"])  # grouping disabled by default
    shelves = build_shelves(store, api, cfg, llm=None)
    # old prune path never sets display_label
    assert all(s.display_label is None for s in shelves)


def test_build_grouped_shelves_merges_two_groups_sharing_an_anchor():
    # Two LLM-proposed names ("Fruit" and "Fruits") both resolve to the same
    # FoodOn anchor TEST:0000004. They must merge into ONE shelf (no duplicate
    # shelf_id, no silently-dropped members).
    api = _mini_foodon()
    chunks = [_chunk("c1", ["TEST:0000006"]),   # apple -> "Fruit"
              _chunk("c2", ["TEST:0000007"])]   # olive -> "Fruits"
    llm = FakeLLM([
        {"groups": ["Fruit", "Fruits"]},        # both anchor to TEST:0000004
        {"assignments": [
            {"food": _cl("TEST:0000006", api), "group": "Fruit"},
            {"food": _cl("TEST:0000007", api), "group": "Fruits"},
        ]},
    ])
    shelves = build_grouped_shelves(iter(chunks), api, BottomUpGroupingConfig(enabled=True),
                                    facet="foods", min_link_confidence=0.0, llm=llm)
    ids = [s.shelf_id for s in shelves]
    assert len(ids) == len(set(ids)), f"duplicate shelf_id(s): {ids}"
    fruit_shelves = [s for s in shelves if s.foodon_id == "TEST:0000004"]
    assert len(fruit_shelves) == 1                       # merged into one
    # both members survive the merge (coverage held)
    assert set(fruit_shelves[0].see_also) >= {"TEST:0000006", "TEST:0000007"}
    assert fruit_shelves[0].chunk_count == 2             # c1 + c2


def test_propose_groups_drops_non_string_names():
    api = _mini_foodon()
    llm = FakeLLM([{"groups": ["Fruit", 123, None, "", "Vegetable"]}])
    groups = propose_groups(api, llm, leaf_freq={}, n_groups=14)
    names = {g.display_name for g in groups}
    assert names == {"Fruit", "Vegetable"}  # junk entries filtered, no crash
