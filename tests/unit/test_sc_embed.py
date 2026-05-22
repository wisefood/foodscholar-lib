"""Embedding step: excludes synthetic roots, builds label+synonym text."""

from __future__ import annotations

from pathlib import Path

from foodscholar.config import SemanticConsolidationConfig
from foodscholar.io.graph import Shelf
from foodscholar.layer_a.semantic_consolidation.embed import (
    embed_shelves,
    is_scaffolding,
    shelf_embed_text,
)
from foodscholar.ontology import FoodOnAPI, load_ontology
from tests.conftest import MockEmbedder


def _mini_foodon() -> FoodOnAPI:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"
    return FoodOnAPI(load_ontology(path), prefix_filter=None)


def _shelf(shelf_id: str, label: str, foodon_id: str | None) -> Shelf:
    return Shelf(shelf_id=shelf_id, label=label, facet="foods", depth=1,
                 foodon_id=foodon_id)


def test_text_includes_label_and_synonyms() -> None:
    ont = _mini_foodon()
    cfg = SemanticConsolidationConfig()
    # TEST:0000008 = olive oil, EXACT synonyms: extra-virgin olive oil, EVOO
    shelf = _shelf("foodon:8", "olive oil", "TEST:0000008")
    text = shelf_embed_text(shelf, ont, cfg)
    assert text.startswith("olive oil")
    assert "EVOO" in text
    assert "extra-virgin olive oil" in text


def test_excludes_synthetic_roots() -> None:
    ont = _mini_foodon()
    cfg = SemanticConsolidationConfig()
    embedder = MockEmbedder(dim=8)
    shelves = [
        _shelf("facet:foods", "Foods", None),  # synthetic root — no foodon_id
        _shelf("foodon:8", "olive oil", "TEST:0000008"),
        _shelf("foodon:6", "apple", "TEST:0000006"),
    ]
    embs = embed_shelves(shelves, ont, embedder, cfg)
    ids = {e.shelf_id for e in embs}
    assert ids == {"foodon:8", "foodon:6"}  # synthetic root excluded
    assert all(len(e.embedding) == embedder.dim for e in embs)
    assert all(e.embedder_id == embedder.model_id for e in embs)


def test_max_synonyms_cap() -> None:
    ont = _mini_foodon()
    cfg = SemanticConsolidationConfig(max_synonyms=1)
    shelf = _shelf("foodon:8", "olive oil", "TEST:0000008")
    text = shelf_embed_text(shelf, ont, cfg)
    assert text.count(" | ") == 1  # label + exactly one synonym


def test_empty_when_no_eligible_shelves() -> None:
    ont = _mini_foodon()
    cfg = SemanticConsolidationConfig()
    embs = embed_shelves(
        [_shelf("facet:foods", "Foods", None)], ont, MockEmbedder(), cfg
    )
    assert embs == []


def test_is_scaffolding_no_synonym_classifier_suffix() -> None:
    ont = _mini_foodon()
    cfg = SemanticConsolidationConfig()
    # 'food product' / 'dairy product': no synonyms, end in "product" → scaffolding.
    assert is_scaffolding(_shelf("s1", "food product", "TEST:0000001"), ont, cfg)
    assert is_scaffolding(_shelf("s11", "dairy product", "TEST:0000011"), ont, cfg)
    # 'olive oil' has synonyms → never scaffolding even if suffix matched.
    assert not is_scaffolding(_shelf("s8", "olive oil", "TEST:0000008"), ont, cfg)
    # 'apple' has a synonym and no classifier suffix → not scaffolding.
    assert not is_scaffolding(_shelf("s6", "apple", "TEST:0000006"), ont, cfg)


def test_scaffolding_excluded_from_embedding() -> None:
    ont = _mini_foodon()
    cfg = SemanticConsolidationConfig()  # exclude_scaffolding default True
    shelves = [
        _shelf("s1", "food product", "TEST:0000001"),  # scaffolding
        _shelf("s11", "dairy product", "TEST:0000011"),  # scaffolding
        _shelf("s8", "olive oil", "TEST:0000008"),  # real food
        _shelf("s6", "apple", "TEST:0000006"),  # real food
    ]
    ids = {e.shelf_id for e in embed_shelves(shelves, ont, MockEmbedder(), cfg)}
    assert ids == {"s8", "s6"}


def test_scaffolding_kept_when_filter_disabled() -> None:
    ont = _mini_foodon()
    cfg = SemanticConsolidationConfig(exclude_scaffolding=False)
    shelves = [
        _shelf("s1", "food product", "TEST:0000001"),
        _shelf("s8", "olive oil", "TEST:0000008"),
    ]
    ids = {e.shelf_id for e in embed_shelves(shelves, ont, MockEmbedder(), cfg)}
    assert ids == {"s1", "s8"}
