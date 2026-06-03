"""Tests for the 1a+ backbone-first projection (build_backbone_shelves)."""

from pathlib import Path

from foodscholar.config import LayerAConfig
from foodscholar.layer_a.backbone import build_backbone_shelves
from foodscholar.layer_a.propagate import SupportTable
from foodscholar.layer_a.prune import shelf_id_for_foodon
from foodscholar.ontology import FoodOnAPI, load_ontology

FOOD = "TEST:0000001"   # food product (backbone root)
PLANT = "TEST:0000002"  # plant food
FRUIT = "TEST:0000004"  # fruit
APPLE = "TEST:0000006"  # apple
OLIVE = "TEST:0000007"  # olive   (0 direct, single child -> filing tier)
OIL = "TEST:0000008"    # olive oil


def _api() -> FoodOnAPI:
    return FoodOnAPI(
        load_ontology(Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"),
        prefix_filter=None,
    )


def _support() -> SupportTable:
    # apple: 2 direct; olive oil: 3 direct. Rolled up onto every ancestor.
    return SupportTable(
        direct_chunk_ids={APPLE: {"c1", "c2"}, OIL: {"c3", "c4", "c5"}},
        with_descendants_chunk_ids={
            APPLE: {"c1", "c2"},
            OIL: {"c3", "c4", "c5"},
            OLIVE: {"c3", "c4", "c5"},                 # 0 direct, all lifted from oil
            FRUIT: {"c1", "c2", "c3", "c4", "c5"},
            PLANT: {"c1", "c2", "c3", "c4", "c5"},
            FOOD: {"c1", "c2", "c3", "c4", "c5"},
        },
    )


def _facet_cfg():
    return LayerAConfig(projection="backbone", min_support=1, max_depth=6,
                        facets=["foods"]).resolve_facet("foods")


def _build():
    return build_backbone_shelves(_support(), _api(), _facet_cfg(), "foods",
                                  root_id=FOOD, max_children=12)


def test_backbone_collapses_filing_tier_and_keeps_real_nodes():
    shelves = {s.foodon_id: s for s in _build()}
    # olive (0 direct, single child) is a filing tier -> collapsed away.
    assert OLIVE not in shelves
    # the meaningful nodes survive; sub-threshold siblings (vegetable/peanut) don't.
    assert set(shelves) == {PLANT, FRUIT, APPLE, OIL}


def test_backbone_reparents_collapsed_child_to_real_ancestor():
    shelves = {s.foodon_id: s for s in _build()}
    # olive oil shows directly under fruit (olive collapsed) — still is-a faithful.
    assert shelves[OIL].parent_shelf_id == shelf_id_for_foodon(FRUIT)


def test_backbone_dl_counts():
    shelves = {s.foodon_id: s for s in _build()}
    assert (shelves[OIL].support_direct, shelves[OIL].support_lifted) == (3, 0)
    assert (shelves[FRUIT].support_direct, shelves[FRUIT].support_lifted) == (0, 5)
    assert shelves[FRUIT].chunk_count == 5


def test_backbone_is_single_parent_and_faithful():
    shelves = _build()
    fids = [s.foodon_id for s in shelves]
    assert len(fids) == len(set(fids))  # each FoodOn node placed once
    api = _api()
    by_sid = {s.shelf_id: s for s in shelves}
    for s in shelves:
        if s.parent_shelf_id:
            parent = by_sid[s.parent_shelf_id]
            assert api.is_subclass_of(s.foodon_id, parent.foodon_id)  # membership is is-a


def test_backbone_prunes_empty_leaf():
    # A supported node with 0 direct chunks and no supported children is a dead-end.
    support = SupportTable(
        direct_chunk_ids={APPLE: {"c1"}},
        with_descendants_chunk_ids={
            APPLE: {"c1"}, FRUIT: {"c1"}, PLANT: {"c1"}, FOOD: {"c1"},
            OLIVE: {"c9"},  # 0 direct, no supported child below it -> pruned
            OIL: set(),
        },
    )
    shelves = {s.foodon_id for s in build_backbone_shelves(
        support, _api(), _facet_cfg(), "foods", root_id=FOOD)}
    assert OLIVE not in shelves
    assert APPLE in shelves
