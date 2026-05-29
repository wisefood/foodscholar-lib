from foodscholar.io.graph import Shelf


def test_shelf_has_optional_display_label():
    s = Shelf(shelf_id="foodon:X", label="plant fruit food product", facet="foods", depth=1)
    assert s.display_label is None
    s2 = Shelf(
        shelf_id="foodon:X", label="plant fruit food product", facet="foods",
        depth=1, display_label="Fruits",
    )
    assert s2.display_label == "Fruits"
