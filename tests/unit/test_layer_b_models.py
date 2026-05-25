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


# ----------------------------------------------------------------------------
# Layer B internal models (not persisted — intermediate to the pipeline)
# ----------------------------------------------------------------------------


def test_theme_candidate_holds_chunks_entities_centroid() -> None:
    from foodscholar.layer_b.models import ThemeCandidate

    c = ThemeCandidate(
        pass_name="similarity",
        chunk_ids={"c1", "c2", "c3"},
        foodon_ids={"FOODON:1", "FOODON:2"},
        centroid_embedding=[0.1] * 8,
        discovered_by="leiden",
    )
    assert len(c.chunk_ids) == 3
    assert c.pass_name == "similarity"
    assert c.discovered_by == "leiden"
    assert c.foodon_ids == {"FOODON:1", "FOODON:2"}


def test_theme_candidate_defaults_empty_entities_and_no_centroid() -> None:
    from foodscholar.layer_b.models import ThemeCandidate

    c = ThemeCandidate(pass_name="relatedness", chunk_ids={"c1"})
    assert c.foodon_ids == set()
    assert c.centroid_embedding is None
    assert c.discovered_by == "leiden"


def test_merge_decision_records_jaccards() -> None:
    from foodscholar.layer_b.models import MergeDecision

    d = MergeDecision(
        similarity_candidate_idx=0,
        relatedness_candidate_idx=1,
        chunk_jaccard=0.5,
        entity_jaccard=0.8,
        combined_similarity=0.62,
        merged=False,
    )
    assert d.combined_similarity == 0.62
    assert not d.merged


def test_layer_b_artifact_default_counters_zero() -> None:
    from foodscholar.layer_b.models import LayerBArtifact

    a = LayerBArtifact(
        artifact_id="lba-001",
        facet="foods",
        config_hash="abc",
        leiden_seed=42,
        started_at="2026-05-25T12:00:00Z",
        finished_at="2026-05-25T12:05:00Z",
    )
    assert a.n_shelves_themed == 0
    assert a.n_shelves_skipped == 0
    assert a.n_themes_total == 0
    assert a.n_themes_by_pass == {}


def test_layer_b_audit_report_passed_requires_perfect_invariants() -> None:
    from foodscholar.layer_b.models import LayerBAuditReport

    # All-green
    r_pass = LayerBAuditReport(parity=1.0, dangling_edges=0, empty_themes=0)
    assert r_pass.passed is True

    # Any non-zero failure flips passed to False
    assert LayerBAuditReport(parity=0.99, dangling_edges=0, empty_themes=0).passed is False
    assert LayerBAuditReport(parity=1.0, dangling_edges=1, empty_themes=0).passed is False
    assert LayerBAuditReport(parity=1.0, dangling_edges=0, empty_themes=1).passed is False
