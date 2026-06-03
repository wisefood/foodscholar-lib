"""Greedy pair-assignment merge of similarity + relatedness candidates."""

from __future__ import annotations

from foodscholar.config import MergeConfig
from foodscholar.layer_b.merge import merge_candidates
from foodscholar.layer_b.models import ThemeCandidate


def _sim(chunks: set[str], ents: set[str] | None = None) -> ThemeCandidate:
    return ThemeCandidate(
        pass_name="global_similarity", chunk_ids=chunks, foodon_ids=ents or set()
    )


def _rel(chunks: set[str], ents: set[str]) -> ThemeCandidate:
    return ThemeCandidate(
        pass_name="relatedness", chunk_ids=chunks, foodon_ids=ents
    )


def test_merge_pairs_above_threshold() -> None:
    """s0 ↔ r0 with 100% chunk overlap merges; s1 (unrelated) stays alone."""
    sim = [_sim({"a", "b", "c"}), _sim({"x", "y", "z"})]
    rel = [_rel({"a", "b", "c"}, {"FOODON:1"})]
    cfg = MergeConfig(chunk_weight=1.0, entity_weight=0.0, dedupe_threshold=0.5)
    themes, _ = merge_candidates(sim, rel, cfg)
    kinds = sorted(t["discovery_pass"] for t in themes)
    assert kinds == ["global_similarity", "merged"]


def test_merge_greedy_picks_highest_combined_first() -> None:
    """Two relatedness candidates both pair with s0; greedy picks the higher
    combined similarity, leaves the other as a singleton."""
    sim = [_sim({"a", "b", "c", "d"})]
    rel = [
        _rel({"a", "b"}, set()),       # 50% chunk overlap with s0
        _rel({"a", "b", "c"}, set()),  # 75% chunk overlap with s0 (winner)
    ]
    cfg = MergeConfig(chunk_weight=1.0, entity_weight=0.0, dedupe_threshold=0.4)
    themes, _ = merge_candidates(sim, rel, cfg)
    merged = [t for t in themes if t["discovery_pass"] == "merged"]
    assert len(merged) == 1
    # Merged theme = union of s0 + rel[1] = {a,b,c,d}
    assert merged[0]["chunk_ids"] == {"a", "b", "c", "d"}
    rel_only = [t for t in themes if t["discovery_pass"] == "relatedness"]
    assert len(rel_only) == 1


def test_merge_below_threshold_leaves_both_alone() -> None:
    sim = [_sim({"a", "b"})]
    rel = [_rel({"c", "d"}, set())]
    cfg = MergeConfig(chunk_weight=0.6, entity_weight=0.4, dedupe_threshold=0.5)
    themes, _ = merge_candidates(sim, rel, cfg)
    kinds = sorted(t["discovery_pass"] for t in themes)
    assert kinds == ["global_similarity", "relatedness"]


def test_merge_empty_inputs_returns_empty() -> None:
    cfg = MergeConfig()
    themes, decisions = merge_candidates([], [], cfg)
    assert themes == []
    assert decisions == []


def test_merge_records_decision_for_every_pair() -> None:
    """Every (sim_i, rel_j) pair gets a MergeDecision regardless of outcome —
    so audit can answer 'why didn't these merge?' without re-running."""
    sim = [_sim({"a"}), _sim({"b"})]
    rel = [_rel({"a"}, set()), _rel({"c"}, set())]
    cfg = MergeConfig(dedupe_threshold=0.99)
    _, decisions = merge_candidates(sim, rel, cfg)
    assert len(decisions) == 4


def test_merge_tie_break_is_deterministic() -> None:
    """Two pairs with identical combined_similarity must resolve the same way
    across runs. Deterministic tie-breaker keys: (-combined, sim_idx, rel_idx)."""
    # Construct two ties at combined = 0.5 — (s0, r0) vs (s0, r1)
    sim = [_sim({"a", "b"})]
    rel = [_rel({"a"}, set()), _rel({"b"}, set())]
    cfg = MergeConfig(chunk_weight=1.0, entity_weight=0.0, dedupe_threshold=0.4)
    out1 = merge_candidates(sim, rel, cfg)
    out2 = merge_candidates(sim, rel, cfg)
    # Same merged-pair structure both runs.
    merged1 = sorted(
        tuple(sorted(t["chunk_ids"])) for t in out1[0] if t["discovery_pass"] == "merged"
    )
    merged2 = sorted(
        tuple(sorted(t["chunk_ids"])) for t in out2[0] if t["discovery_pass"] == "merged"
    )
    assert merged1 == merged2


def test_merge_all_pass_through_when_no_overlap() -> None:
    """Disjoint candidates: every input candidate becomes its own theme.
    The 'all-merged canary' is the inverse case (Pass 2 isn't earning compute)."""
    sim = [_sim({"a"}), _sim({"b"})]
    rel = [_rel({"x"}, set()), _rel({"y"}, set())]
    cfg = MergeConfig(dedupe_threshold=0.5)
    themes, _ = merge_candidates(sim, rel, cfg)
    assert len(themes) == 4
    kinds = sorted(t["discovery_pass"] for t in themes)
    assert kinds == ["global_similarity", "global_similarity", "relatedness", "relatedness"]


def test_merge_all_merged_canary() -> None:
    """If every sim pairs with every rel above threshold, greedy still
    assigns each side once — extras pass through as singletons.
    This documents the upper bound: a perfectly-aligned dual pass produces
    min(len(sim), len(rel)) merged themes."""
    sim = [_sim({"a"}), _sim({"b"}), _sim({"c"})]
    rel = [_rel({"a"}, set()), _rel({"b"}, set())]  # only 2 rel candidates
    cfg = MergeConfig(chunk_weight=1.0, entity_weight=0.0, dedupe_threshold=0.99)
    themes, _ = merge_candidates(sim, rel, cfg)
    merged = [t for t in themes if t["discovery_pass"] == "merged"]
    sim_only = [t for t in themes if t["discovery_pass"] == "global_similarity"]
    rel_only = [t for t in themes if t["discovery_pass"] == "relatedness"]
    assert len(merged) == 2
    assert len(sim_only) == 1  # the unused global_similarity
    assert len(rel_only) == 0


def test_merge_uses_both_chunk_and_entity_weights() -> None:
    """combined = chunk_weight * J(chunks) + entity_weight * J(entities).
    A high entity overlap alone should be enough to merge if entity_weight
    is dialed up."""
    sim = [_sim({"a", "b"}, {"FOODON:1", "FOODON:2"})]
    rel = [_rel({"c", "d"}, {"FOODON:1", "FOODON:2"})]
    # chunk_jaccard = 0, entity_jaccard = 1.0, combined = 0.7 with these weights
    cfg = MergeConfig(chunk_weight=0.3, entity_weight=0.7, dedupe_threshold=0.5)
    themes, decisions = merge_candidates(sim, rel, cfg)
    merged = [t for t in themes if t["discovery_pass"] == "merged"]
    assert len(merged) == 1
    # And the audit decision records the right components
    d = decisions[0]
    assert d.chunk_jaccard == 0.0
    assert d.entity_jaccard == 1.0
    assert d.merged is True


def test_merge_global_and_local_returns_themes_with_union_shelf_ids() -> None:
    """A global similarity candidate that overlaps a per-shelf relatedness
    candidate produces a merged theme whose source shelves = union of both."""
    from foodscholar.layer_b.merge import merge_global_and_local_candidates

    global_cands = [
        ThemeCandidate(
            pass_name="global_similarity",
            chunk_ids={"c1", "c2", "c3", "c4"},
            foodon_ids=set(),
            centroid_embedding=[0.1] * 3,
        ),
    ]
    rel_cands_by_shelf = {
        "shelf:fat": [
            ThemeCandidate(
                pass_name="relatedness",
                chunk_ids={"c1", "c2"},
                foodon_ids={"FOODON:1"},
            ),
        ],
        "shelf:meat": [
            ThemeCandidate(
                pass_name="relatedness",
                chunk_ids={"c3", "c4"},
                foodon_ids={"FOODON:2"},
            ),
        ],
    }
    # Lower threshold so both per-shelf rel-cands merge with the global cand.
    # global {c1,c2,c3,c4} vs shelf:fat {c1,c2}: J=0.5, combined=0.5*1.0=0.5
    # global {c1,c2,c3,c4} vs shelf:meat {c3,c4}: J=0.5, combined=0.5
    cfg = MergeConfig(chunk_weight=1.0, entity_weight=0.0, dedupe_threshold=0.4)
    themes, _decisions = merge_global_and_local_candidates(
        global_cands, rel_cands_by_shelf, cfg,
    )
    merged = [t for t in themes if t["discovery_pass"] == "merged"]
    assert any(set(t["shelf_ids"]) == {"shelf:fat", "shelf:meat"} for t in merged)


def test_merge_global_and_local_unmerged_global_keeps_empty_shelf_ids() -> None:
    """A global similarity theme that didn't merge with any relatedness
    candidate returns shelf_ids=[] so the orchestrator can backfill."""
    from foodscholar.layer_b.merge import merge_global_and_local_candidates

    global_cands = [
        ThemeCandidate(
            pass_name="global_similarity",
            chunk_ids={"c100", "c101"},
            foodon_ids=set(),
            centroid_embedding=[0.1] * 3,
        ),
    ]
    cfg = MergeConfig()
    themes, _ = merge_global_and_local_candidates(global_cands, {}, cfg)
    glob = [t for t in themes if t["discovery_pass"] == "global_similarity"]
    assert len(glob) == 1
    assert glob[0]["shelf_ids"] == []


def test_merge_global_and_local_unmerged_per_shelf_uses_origin_shelf() -> None:
    """In per-shelf Pass 1 the similarity candidate carries its origin_shelf_id;
    an unmerged similarity theme must attach to that ONE shelf — not [] and not a
    chunk-derived union (which over-attaches via lifted chunks). The transient
    origin_shelf_id must not leak into the theme dict."""
    from foodscholar.layer_b.merge import merge_global_and_local_candidates

    global_cands = [
        ThemeCandidate(
            pass_name="global_similarity",
            chunk_ids={"c100", "c101"},
            foodon_ids=set(),
            centroid_embedding=[0.1] * 3,
            origin_shelf_id="shelf:spice",
        ),
    ]
    cfg = MergeConfig()
    themes, _ = merge_global_and_local_candidates(global_cands, {}, cfg)
    glob = [t for t in themes if t["discovery_pass"] == "global_similarity"]
    assert len(glob) == 1
    assert glob[0]["shelf_ids"] == ["shelf:spice"]
    assert "origin_shelf_id" not in glob[0]
