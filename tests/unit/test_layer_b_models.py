"""Tests for Layer B Pydantic models (Theme extensions + Layer B-specific models)."""

from __future__ import annotations

from foodscholar.io.graph import Theme


def test_theme_has_brief_extensions() -> None:
    """Theme gains per-theme metadata from layer_b_construction_brief.md §3:
    facet, discovery_pass, keyword_terms, foodon_id_signature, config_hash,
    version. shelf_ids stays a list (multi-shelf themes are still allowed
    long-term per BRIEF §4; the v1 builder emits length-1 lists)."""
    t = Theme(
        theme_id="t1",
        label="Calcium and bone health",
        shelf_ids=["s-cow-milk"],
        chunk_count=42,
        discovered_by="leiden",
        discovery_version="v0.1",
        facet="foods",
        discovery_pass="merged",
        keyword_terms=["calcium", "bone", "density"],
        foodon_id_signature=["FOODON:00001234", "FOODON:00005678"],
        config_hash="abc123",
        version="v0.1",
    )
    assert t.facet == "foods"
    assert t.discovery_pass == "merged"
    assert t.keyword_terms == ["calcium", "bone", "density"]
    assert t.foodon_id_signature == ["FOODON:00001234", "FOODON:00005678"]
    assert t.config_hash == "abc123"
    assert t.version == "v0.1"
    # Multi-shelf list preserved (length-1 in v1, but the model stays open)
    assert isinstance(t.shelf_ids, list)
    assert t.shelf_ids == ["s-cow-milk"]


def test_theme_defaults_for_new_fields() -> None:
    """All new fields default to safe empties so existing call sites keep working
    (only `facet` and `discovery_pass` are required additions)."""
    t = Theme(
        theme_id="t1",
        label="x",
        shelf_ids=["s1"],
        discovered_by="leiden",
        discovery_version="v0.1",
        facet="foods",
        discovery_pass="similarity",
    )
    assert t.keyword_terms == []
    assert t.foodon_id_signature == []
    assert t.config_hash == ""
    assert t.version == ""
    assert t.chunk_count == 0
