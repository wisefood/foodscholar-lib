"""Layer B BERTopic config: algorithm selector + bertopic knobs. Leiden stays default."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from foodscholar.config import BertopicConfig, LayerBConfig


def test_algorithm_defaults_to_leiden() -> None:
    c = LayerBConfig()
    assert c.algorithm == "leiden"  # Leiden remains the production default


def test_bertopic_config_defaults() -> None:
    b = BertopicConfig()
    assert b.scope == "direct"
    assert b.clusterer == "hdbscan"
    assert b.min_topic_size == 15
    assert b.n_clusters is None  # auto when kmeans + None


def test_layer_b_has_bertopic_block() -> None:
    c = LayerBConfig()
    assert isinstance(c.bertopic, BertopicConfig)


def test_algorithm_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        LayerBConfig(algorithm="kmeans")  # not a valid algorithm selector


def test_bertopic_scope_and_clusterer_validated() -> None:
    assert LayerBConfig(algorithm="bertopic").bertopic.scope == "direct"
    with pytest.raises(ValidationError):
        BertopicConfig(scope="bogus")
    with pytest.raises(ValidationError):
        BertopicConfig(clusterer="bogus")
