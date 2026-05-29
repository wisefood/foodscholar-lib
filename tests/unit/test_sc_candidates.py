"""Candidate generation: cosine threshold, dedup, and pre-LLM filters."""

from __future__ import annotations

from foodscholar.config import SemanticConsolidationConfig
from foodscholar.io.graph import Shelf
from foodscholar.layer_a.semantic_consolidation.candidates import (
    _already_merged,
    _compound_food,
    _subtype_collision,
    find_candidates,
)
from foodscholar.layer_a.semantic_consolidation.models import ShelfEmbedding


def _shelf(shelf_id: str, label: str, *, foodon_id: str | None = None,
           see_also: list[str] | None = None, depth: int = 1) -> Shelf:
    return Shelf(
        shelf_id=shelf_id,
        label=label,
        facet="foods",
        depth=depth,
        foodon_id=foodon_id or f"FOODON:{shelf_id}",
        see_also=see_also or [],
    )


def _emb(shelf_id: str, vec: list[float]) -> ShelfEmbedding:
    return ShelfEmbedding(
        shelf_id=shelf_id,
        foodon_id=f"FOODON:{shelf_id}",
        text=shelf_id,
        embedding=vec,
        embedder_id="test",
    )


def test_threshold_and_symmetric_dedup() -> None:
    cfg = SemanticConsolidationConfig(cosine_threshold=0.9)
    # a,b nearly identical; c orthogonal.
    embs = [
        _emb("a", [1.0, 0.0, 0.0]),
        _emb("b", [0.99, 0.01, 0.0]),
        _emb("c", [0.0, 1.0, 0.0]),
    ]
    by_id = {e.shelf_id: _shelf(e.shelf_id, e.shelf_id) for e in embs}
    candidates, filtered = find_candidates(embs, by_id, cfg)
    pairs = {(p.shelf_a, p.shelf_b) for p in candidates}
    assert pairs == {("a", "b")}  # one pair, deduped (i<j), c excluded
    assert filtered == []


def test_subtype_collision_filter() -> None:
    cfg = SemanticConsolidationConfig(cosine_threshold=0.5)
    embs = [_emb("x", [1.0, 0.0]), _emb("y", [1.0, 0.0])]
    by_id = {
        "x": _shelf("x", "turkey bacon"),
        "y": _shelf("y", "bacon"),
    }
    candidates, filtered = find_candidates(embs, by_id, cfg)
    assert candidates == []
    assert len(filtered) == 1
    assert filtered[0].filtered_reason.startswith("subtype_collision")


def test_compound_food_filter() -> None:
    cfg = SemanticConsolidationConfig(cosine_threshold=0.5)
    embs = [_emb("x", [1.0, 0.0]), _emb("y", [1.0, 0.0])]
    by_id = {
        "x": _shelf("x", "cream cheese"),
        "y": _shelf("y", "cream"),
    }
    candidates, filtered = find_candidates(embs, by_id, cfg)
    assert candidates == []
    assert filtered[0].filtered_reason.startswith("compound_food")


def test_already_merged_filter() -> None:
    cfg = SemanticConsolidationConfig(cosine_threshold=0.5)
    embs = [_emb("x", [1.0, 0.0]), _emb("y", [1.0, 0.0])]
    by_id = {
        "x": _shelf("x", "yogurt", foodon_id="FOODON:x", see_also=["FOODON:y"]),
        "y": _shelf("y", "yoghurt", foodon_id="FOODON:y"),
    }
    candidates, filtered = find_candidates(embs, by_id, cfg)
    assert candidates == []
    assert filtered[0].filtered_reason == "already_merged"


def test_qualifier_only_difference_is_candidate() -> None:
    cfg = SemanticConsolidationConfig(cosine_threshold=0.5)
    embs = [_emb("x", [1.0, 0.0]), _emb("y", [1.0, 0.0])]
    by_id = {
        "x": _shelf("x", "olive oil product"),
        "y": _shelf("y", "olive oil"),
    }
    candidates, filtered = find_candidates(embs, by_id, cfg)
    assert len(candidates) == 1  # only differs by "product" qualifier
    assert filtered == []


def test_cap_per_shelf_keeps_strongest() -> None:
    cfg = SemanticConsolidationConfig(cosine_threshold=0.5, max_candidates_per_shelf=1)
    # hub "h" is close to a,b,c; cap=1 should keep only its strongest pair.
    embs = [
        _emb("h", [1.0, 0.0, 0.0]),
        _emb("a", [0.99, 0.10, 0.0]),
        _emb("b", [0.95, 0.20, 0.0]),
        _emb("c", [0.90, 0.30, 0.0]),
    ]
    by_id = {e.shelf_id: _shelf(e.shelf_id, e.shelf_id) for e in embs}
    candidates, _ = find_candidates(embs, by_id, cfg)
    touching_h = [p for p in candidates if "h" in (p.shelf_a, p.shelf_b)]
    assert len(touching_h) == 1


def test_subtype_collision_unit() -> None:
    cfg = SemanticConsolidationConfig()
    assert _subtype_collision("turkey bacon", "bacon", cfg)
    assert _subtype_collision("bacon", "turkey bacon", cfg)  # order-independent
    assert _subtype_collision("bacon", "ham", cfg) is None  # neither is subtype
    # both subtypes => rule says nothing
    assert _subtype_collision("turkey bacon", "beef bacon", cfg) is None


def test_compound_food_unit() -> None:
    assert _compound_food("cream cheese", "cream")
    assert _compound_food("tuna salad", "tuna")
    assert _compound_food("olive oil product", "olive oil") is None  # qualifier only
    assert _compound_food("apple", "apple") is None  # identical


def test_already_merged_unit() -> None:
    a = _shelf("a", "yogurt", foodon_id="FOODON:a", see_also=["FOODON:b"])
    b = _shelf("b", "yoghurt", foodon_id="FOODON:b")
    assert _already_merged(a, b) == "already_merged"
    assert _already_merged(b, a) == "already_merged"  # symmetric
    c = _shelf("c", "cheese", foodon_id="FOODON:c")
    assert _already_merged(a, c) is None
