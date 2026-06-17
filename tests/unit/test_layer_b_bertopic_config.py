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


def test_scope_is_a_shared_top_level_knob() -> None:
    # scope now lives on LayerBConfig and applies to BOTH methods.
    assert LayerBConfig().scope == "direct"
    assert LayerBConfig(scope="subtree").scope == "subtree"


def test_resolved_scope_alias_precedence() -> None:
    # Leiden: only the shared knob is consulted (bertopic.scope ignored).
    assert LayerBConfig(algorithm="leiden", scope="subtree").resolved_scope() == "subtree"
    assert (
        LayerBConfig(
            algorithm="leiden", scope="direct", bertopic={"scope": "subtree"}
        ).resolved_scope()
        == "direct"
    )
    # bertopic: an explicit (non-default) bertopic.scope overrides the shared knob…
    assert (
        LayerBConfig(
            algorithm="bertopic", scope="direct", bertopic={"scope": "subtree"}
        ).resolved_scope()
        == "subtree"
    )
    # …but when bertopic.scope is left at its default, the shared knob governs.
    assert (
        LayerBConfig(algorithm="bertopic", scope="subtree").resolved_scope() == "subtree"
    )
