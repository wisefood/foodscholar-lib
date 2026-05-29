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
