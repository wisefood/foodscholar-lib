"""Tests for the nested LayerBConfig (per layer_b_construction_brief.md §5)."""

from __future__ import annotations

from foodscholar.config import LayerBConfig


def test_layer_b_config_defaults_match_brief() -> None:
    c = LayerBConfig()
    # Top-level (carry-over from the flat config)
    assert c.min_chunks_per_shelf == 50
    # Similarity pass
    assert c.similarity.knn_k == 15
    assert c.similarity.edge_threshold == 0.55
    assert c.similarity.require_mutual is True
    assert c.similarity.algorithm == "leiden"
    # Relatedness pass
    assert c.relatedness.tau_strict == 0.80
    assert c.relatedness.min_shared_ids == 2
    assert c.relatedness.max_doc_frequency == 0.40
    assert c.relatedness.algorithm == "leiden"
    # The umbrella class that survives Layer A is excluded by default.
    assert "FOODON:00001002" in c.relatedness.always_exclude_iris
    # Leiden
    assert c.leiden.resolution == 1.0
    assert c.leiden.n_iterations == 10
    assert c.leiden.min_community_size == 15
    assert c.leiden.random_state == 42
    # Merge
    assert c.merge.chunk_weight == 0.6
    assert c.merge.entity_weight == 0.4
    assert c.merge.dedupe_threshold == 0.70
    # Labeling — LLM is the v1 default
    assert c.labeling.strategy == "llm"
    assert c.labeling.top_keywords == 5
    # Audit gates
    assert c.audit.target_themes_per_shelf_min == 3
    assert c.audit.target_themes_per_shelf_max == 12
    assert c.audit.merged_rate_min == 0.20
    assert c.audit.merged_rate_max == 0.80
    # Embedded-fraction gate (added per Plan-agent review)
    assert c.min_embedded_fraction == 0.80


def test_layer_b_config_yaml_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The nested block loads cleanly from YAML via load_config and per-block
    overrides apply while sibling defaults stay intact."""
    from foodscholar.config import load_config

    yaml = """
corpus:
  chunks_path: tests/fixtures/sample_chunks.jsonl
layer_b:
  min_chunks_per_shelf: 100
  similarity:
    knn_k: 20
  merge:
    dedupe_threshold: 0.5
  labeling:
    strategy: keyword
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml)
    cfg = load_config(p)
    assert cfg.layer_b.min_chunks_per_shelf == 100
    assert cfg.layer_b.similarity.knn_k == 20
    assert cfg.layer_b.merge.dedupe_threshold == 0.5
    assert cfg.layer_b.labeling.strategy == "keyword"
    # Untouched siblings keep defaults.
    assert cfg.layer_b.relatedness.tau_strict == 0.80
    assert cfg.layer_b.leiden.random_state == 42


def test_layer_b_hdbscan_algorithm_rejected_for_v1() -> None:
    """V1 ships Leiden only on both passes; HDBSCAN as an algorithm choice
    must raise at construction time so misconfigured runs fail loudly."""
    import pytest
    from pydantic import ValidationError

    from foodscholar.config import RelatednessConfig, SimilarityConfig

    with pytest.raises(ValidationError):
        SimilarityConfig(algorithm="hdbscan")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RelatednessConfig(algorithm="hdbscan")  # type: ignore[arg-type]
