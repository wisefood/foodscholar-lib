"""Cluster judge: prompt construction, index→id mapping, defensive parsing."""

from __future__ import annotations

from pathlib import Path

from foodscholar.config import SemanticConsolidationConfig
from foodscholar.io.graph import Shelf
from foodscholar.layer_a.semantic_consolidation.judge import (
    _parse_cluster,
    judge_clusters,
)
from foodscholar.ontology import FoodOnAPI, load_ontology
from foodscholar.storage.memory import InMemoryChunkStore


def _mini_foodon() -> FoodOnAPI:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"
    return FoodOnAPI(load_ontology(path), prefix_filter=None)


class ScriptedJudge:
    model_id = "scripted-judge-v0"

    def __init__(self, response: dict[str, object]) -> None:
        self._response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:  # pragma: no cover
        return "{}"

    def generate_json(self, prompt, schema, max_tokens=1024):
        self.prompts.append(prompt)
        return dict(self._response)


def _shelves() -> dict[str, Shelf]:
    return {
        "foodon:8": Shelf(shelf_id="foodon:8", label="olive oil", facet="foods",
                          depth=2, foodon_id="TEST:0000008"),
        "foodon:7": Shelf(shelf_id="foodon:7", label="olive", facet="foods",
                          depth=1, foodon_id="TEST:0000007"),
        "foodon:6": Shelf(shelf_id="foodon:6", label="apple", facet="foods",
                          depth=2, foodon_id="TEST:0000006"),
    }


def test_prompt_carries_labels_synonyms_and_indices() -> None:
    judge = ScriptedJudge(
        {"merge_groups": [{"members": [1, 2], "canonical_name": "olive oil",
                           "confidence": 0.9, "rationale": "same"}],
         "keep_alone": []}
    )
    cluster = ["foodon:8", "foodon:7"]
    decisions = judge_clusters(
        [cluster], _shelves(), _mini_foodon(), InMemoryChunkStore(), judge,
        SemanticConsolidationConfig(),
    )
    assert len(decisions) == 1
    prompt = judge.prompts[0]
    assert "Shelf 1:" in prompt and "Shelf 2:" in prompt
    assert "olive oil" in prompt and "olive" in prompt
    assert "EVOO" in prompt  # synonym of olive oil
    assert "sample chunks" in prompt
    # few-shot on by default
    assert "EXAMPLES" in prompt


def test_few_shot_toggle() -> None:
    judge = ScriptedJudge({"merge_groups": [], "keep_alone": [1, 2]})
    cfg = SemanticConsolidationConfig(use_few_shot=False)
    judge_clusters([["foodon:8", "foodon:7"]], _shelves(), _mini_foodon(),
                   InMemoryChunkStore(), judge, cfg)
    assert "EXAMPLES" not in judge.prompts[0]


def test_index_mapping_to_shelf_ids() -> None:
    members = ["foodon:8", "foodon:7", "foodon:6"]
    obj = {
        "merge_groups": [{"members": [1, 2], "canonical_name": "olive oil",
                          "confidence": 0.92, "rationale": "same food"}],
        "keep_alone": [3],
    }
    d = _parse_cluster(members, obj, "m")
    assert len(d.merge_groups) == 1
    assert d.merge_groups[0].members == ["foodon:8", "foodon:7"]
    assert d.merge_groups[0].confidence == 0.92
    assert d.keep_alone == ["foodon:6"]


def test_parse_forgotten_member_defaults_to_keep_alone() -> None:
    members = ["a", "b", "c"]
    # model only mentioned a,b — c must not vanish or silently merge.
    obj = {"merge_groups": [{"members": [1, 2], "canonical_name": "x",
                             "confidence": 0.9, "rationale": "r"}],
           "keep_alone": []}
    d = _parse_cluster(members, obj, "m")
    assert d.keep_alone == ["c"]


def test_parse_drops_out_of_range_and_singletons() -> None:
    members = ["a", "b"]
    obj = {"merge_groups": [{"members": [1, 99], "canonical_name": "x",
                             "confidence": 0.9, "rationale": "r"}],  # 99 invalid
           "keep_alone": [2]}
    d = _parse_cluster(members, obj, "m")
    # group collapses to a single valid member → not a merge; a falls to keep.
    assert d.merge_groups == []
    assert set(d.keep_alone) == {"a", "b"}


def test_parse_clamps_confidence_and_handles_garbage() -> None:
    members = ["a", "b"]
    obj = {"merge_groups": [{"members": [1, 2], "canonical_name": "x",
                             "confidence": 5.0, "rationale": "r"}],
           "keep_alone": []}
    d = _parse_cluster(members, obj, "m")
    assert d.merge_groups[0].confidence == 1.0

    empty = _parse_cluster(members, {}, "m")  # malformed → everyone kept alone
    assert empty.merge_groups == []
    assert set(empty.keep_alone) == {"a", "b"}


def test_judge_error_keeps_all_alone() -> None:
    class Boom:
        model_id = "boom"

        def generate(self, *a, **k):  # pragma: no cover
            return "{}"

        def generate_json(self, *a, **k):
            raise RuntimeError("provider down")

    decisions = judge_clusters(
        [["foodon:8", "foodon:7"]], _shelves(), _mini_foodon(),
        InMemoryChunkStore(), Boom(), SemanticConsolidationConfig(),
    )
    assert decisions[0].merge_groups == []
    assert set(decisions[0].keep_alone) == {"foodon:8", "foodon:7"}
