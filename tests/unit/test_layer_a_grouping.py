from foodscholar.io.graph import Shelf


def test_shelf_has_optional_display_label():
    s = Shelf(shelf_id="foodon:X", label="plant fruit food product", facet="foods", depth=1)
    assert s.display_label is None
    s2 = Shelf(
        shelf_id="foodon:X", label="plant fruit food product", facet="foods",
        depth=1, display_label="Fruits",
    )
    assert s2.display_label == "Fruits"


from foodscholar.config import BottomUpGroupingConfig, LayerAConfig


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
