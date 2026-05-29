"""Bucketing (confidence + block-list) and N-way merge application."""

from __future__ import annotations

from foodscholar.config import SemanticConsolidationConfig
from foodscholar.io.graph import Shelf
from foodscholar.layer_a.attach import ShelfIndex
from foodscholar.layer_a.semantic_consolidation.apply import (
    apply_groups,
    bucket_groups,
)
from foodscholar.layer_a.semantic_consolidation.models import (
    ClusterDecision,
    MergeGroup,
)


def _shelf(shelf_id, label, *, depth, foodon_id, parent=None,
           direct=0, lifted=0, see_also=None):
    return Shelf(shelf_id=shelf_id, label=label, facet="foods", depth=depth,
                 foodon_id=foodon_id, parent_shelf_id=parent,
                 support_direct=direct, support_lifted=lifted,
                 see_also=see_also or [])


def _group(members, *, conf=0.9, name="canon"):
    return MergeGroup(members=members, canonical_name=name, confidence=conf,
                      rationale="dup")


def _decision(groups, keep=None):
    return ClusterDecision(
        cluster_members=[m for g in groups for m in g.members] + (keep or []),
        merge_groups=groups, keep_alone=keep or [], llm_id="m",
        prompt_version="v2.0-cluster", decided_at="2026-01-01T00:00:00Z")


def _by_id(shelves):
    return {s.shelf_id: s for s in shelves}


# ----------------------------------------------------------------- bucketing


def test_bucket_confidence_gate() -> None:
    cfg = SemanticConsolidationConfig(auto_merge_confidence=0.80)
    shelves = [
        _shelf("a", "yogurt", depth=2, foodon_id="F:a"),
        _shelf("b", "yoghurt", depth=2, foodon_id="F:b"),
        _shelf("c", "curd", depth=2, foodon_id="F:c"),
        _shelf("d", "dahi", depth=2, foodon_id="F:d"),
    ]
    decisions = [_decision([_group(["a", "b"], conf=0.90),
                            _group(["c", "d"], conf=0.50)])]
    applied, uncertain, blocked = bucket_groups(decisions, _by_id(shelves), cfg)
    assert [g.members for g in applied] == [["a", "b"]]
    assert [g.members for g in uncertain] == [["c", "d"]]
    assert blocked == []


def test_bucket_blocklist_blocks_whole_group() -> None:
    cfg = SemanticConsolidationConfig(
        auto_merge_confidence=0.80,
        permanent_block_list=[("F:oil", "F:fat")],  # never merge oil+fat
    )
    shelves = [
        _shelf("oil", "oil", depth=2, foodon_id="F:oil"),
        _shelf("fat", "fat", depth=2, foodon_id="F:fat"),
        _shelf("lard", "lard", depth=2, foodon_id="F:lard"),
    ]
    # judge wanted all three merged; oil+fat is vetoed. Salvaging sub-groups
    # would re-link oil+fat transitively via lard, so we conservatively block
    # the whole group — the veto must hold.
    decisions = [_decision([_group(["oil", "fat", "lard"], conf=0.9)])]
    applied, _uncertain, blocked = bucket_groups(decisions, _by_id(shelves), cfg)

    assert applied == []
    assert len(blocked) == 1
    assert ("oil", "fat") in blocked[0].blocked_pairs


def test_bucket_blocklist_full_pair_blocks_entirely() -> None:
    cfg = SemanticConsolidationConfig(
        auto_merge_confidence=0.80,
        permanent_block_list=[("F:oil", "F:fat")],
    )
    shelves = [
        _shelf("oil", "oil", depth=2, foodon_id="F:oil"),
        _shelf("fat", "fat", depth=2, foodon_id="F:fat"),
    ]
    decisions = [_decision([_group(["oil", "fat"], conf=0.95)])]
    applied, _uncertain, blocked = bucket_groups(decisions, _by_id(shelves), cfg)
    assert applied == []  # nothing left to merge after the veto
    assert len(blocked) == 1


# ----------------------------------------------------------------- application


def test_apply_nway_folds_and_routes() -> None:
    cfg = SemanticConsolidationConfig()
    shelves = [
        _shelf("keep", "yogurt", depth=2, foodon_id="F:keep", direct=10, lifted=12),
        _shelf("d1", "yoghurt", depth=3, foodon_id="F:d1", direct=4, lifted=4),
        _shelf("d2", "curd", depth=3, foodon_id="F:d2", direct=2, lifted=3),
    ]
    out = apply_groups(shelves, [_group(["keep", "d1", "d2"])], cfg)
    assert {s.shelf_id for s in out} == {"keep"}  # both losers folded
    canonical = out[0]
    assert canonical.shelf_id == "keep"  # shallowest depth wins
    assert "F:d1" in canonical.see_also and "F:d2" in canonical.see_also
    assert canonical.support_direct == 16  # 10+4+2
    assert canonical.support_lifted == 19  # 12+4+3

    idx = ShelfIndex.from_shelves(out)
    assert idx.per_facet["foods"].by_seealso["F:d1"] is canonical
    assert idx.per_facet["foods"].by_seealso["F:d2"] is canonical


def test_apply_reparents_children_of_folded() -> None:
    cfg = SemanticConsolidationConfig()
    shelves = [
        _shelf("keep", "olive", depth=1, foodon_id="F:keep"),
        _shelf("drop", "olives", depth=1, foodon_id="F:drop"),
        _shelf("child", "olive oil", depth=2, foodon_id="F:child", parent="drop"),
    ]
    out = apply_groups(shelves, [_group(["keep", "drop"])], cfg)
    by_id = _by_id(out)
    assert "drop" not in by_id
    assert by_id["child"].parent_shelf_id == "keep"


def test_apply_idempotent() -> None:
    cfg = SemanticConsolidationConfig()
    shelves = [
        _shelf("keep", "yogurt", depth=2, foodon_id="F:keep"),
        _shelf("drop", "yoghurt", depth=3, foodon_id="F:drop"),
    ]
    once = apply_groups(shelves, [_group(["keep", "drop"])], cfg)
    twice = apply_groups(once, [_group(["keep", "drop"])], cfg)
    assert {s.shelf_id for s in twice} == {"keep"}
    assert twice[0].see_also.count("F:drop") == 1  # not double-added
