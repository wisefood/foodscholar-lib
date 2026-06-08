"""Layer C config: new Stage-1 / map-reduce / benchmark fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from foodscholar.config import LayerCConfig


def test_layer_c_defaults() -> None:
    c = LayerCConfig()
    # existing fields preserved
    assert c.llm_model == "llama-3.1-8b-instant"
    assert c.prompt_version == "v1"
    assert c.grounding_check == "strict"
    # new fields
    assert c.stage1_method == "lexrank"
    assert c.stage1_sentences == 8
    assert c.map_reduce_threshold == 400
    assert c.group_char_budget == 20_000
    assert c.max_summary_chars == 4000
    assert c.benchmark_out_dir == "data/layer_c_bench"


def test_layer_c_rejects_unknown_method() -> None:
    with pytest.raises(ValidationError):
        LayerCConfig(stage1_method="bogus")


def test_layer_c_forbids_extra() -> None:
    with pytest.raises(ValidationError):
        LayerCConfig(nonexistent_field=1)
