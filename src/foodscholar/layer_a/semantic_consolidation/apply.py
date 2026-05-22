"""Classify cluster decisions into buckets and apply confirmed merges N-way.

A `MergeGroup` is applied iff its confidence clears `auto_merge_confidence`
*and* it survives the permanent block-list. The block-list (pairs of FoodOn
ids that must never co-merge) is enforced by splitting a group into sub-groups
that respect every veto, via connected components over the
"allowed to merge" graph.

Applying a group folds all its members into one canonical shelf:

  - the canonical is the shallowest member (tie-break: shorter, then
    alphabetically-earlier label);
  - every other member's `foodon_id` (and prior `see_also`) is appended to the
    canonical's `see_also` — the routing primitive `attach.ShelfIndex` indexes,
    so re-running attach re-homes their chunks onto the canonical;
  - support counts are summed (provisional — attach recomputes the honest
    deduped `chunk_count`);
  - members parented on a folded shelf are re-parented onto the canonical;
  - the folded members are dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.config import SemanticConsolidationConfig
    from foodscholar.io.graph import Shelf
    from foodscholar.layer_a.semantic_consolidation.models import (
        ClusterDecision,
        MergeGroup,
    )

_log = get_logger("foodscholar.semantic_consolidation")


def bucket_groups(
    decisions: list[ClusterDecision],
    shelves_by_id: dict[str, Shelf],
    cfg: SemanticConsolidationConfig,
) -> tuple[list[MergeGroup], list[MergeGroup], list[MergeGroup]]:
    """Split every proposed group into ``(applied, uncertain, blocked)``.

    Block-list enforcement runs first: a group containing a vetoed pair is
    split into the largest sub-groups that contain no veto. The original
    (un-split) group, annotated with its `blocked_pairs`, is recorded as
    blocked for the audit; the surviving sub-groups are then gated on
    confidence like any other group.
    """
    blocked_pairs = _blocklist_set(cfg)
    applied: list[MergeGroup] = []
    uncertain: list[MergeGroup] = []
    blocked: list[MergeGroup] = []

    for decision in decisions:
        for group in decision.merge_groups:
            vetoed = _vetoed_pairs(group, shelves_by_id, blocked_pairs)
            if vetoed:
                # Conservative: a vetoed pair must never co-merge, not even
                # transitively (oil~lard~fat would re-link oil+fat). Salvaging
                # veto-free sub-groups risks exactly that, so block the whole
                # group. Re-author the candidates/threshold if a legitimate
                # merge is collateral — that's rarer than a bad merge.
                blocked.append(group.model_copy(update={"blocked_pairs": vetoed}))
                continue
            if group.confidence >= cfg.auto_merge_confidence:
                applied.append(group)
            else:
                uncertain.append(group)
    return applied, uncertain, blocked


def apply_groups(
    shelves: list[Shelf],
    groups: list[MergeGroup],
    cfg: SemanticConsolidationConfig,
) -> list[Shelf]:
    """Fold each confirmed group into its canonical shelf."""
    by_id: dict[str, Shelf] = {s.shelf_id: s for s in shelves}
    # Strongest first, so the highest-confidence claim on a shelf wins if two
    # groups ever overlap (they shouldn't after bucketing, but be safe).
    for group in sorted(groups, key=lambda g: g.confidence, reverse=True):
        members = [by_id[m] for m in group.members if m in by_id]
        if len(members) < 2:
            continue  # some members already folded by an earlier group
        canonical = _pick_canonical(members)
        see = list(canonical.see_also)
        direct = canonical.support_direct
        lifted = canonical.support_lifted
        for loser in members:
            if loser.shelf_id == canonical.shelf_id:
                continue
            if loser.foodon_id and loser.foodon_id not in see:
                see.append(loser.foodon_id)
            see.extend(fid for fid in loser.see_also if fid not in see)
            direct += loser.support_direct
            lifted += loser.support_lifted
            del by_id[loser.shelf_id]
        by_id[canonical.shelf_id] = canonical.model_copy(
            update={
                "see_also": see,
                "support_direct": direct,
                "support_lifted": lifted,
            }
        )
        # Re-parent anything that hung off a folded member.
        folded = {m.shelf_id for m in members if m.shelf_id != canonical.shelf_id}
        for sid, s in list(by_id.items()):
            if s.parent_shelf_id in folded:
                by_id[sid] = s.model_copy(
                    update={"parent_shelf_id": canonical.shelf_id}
                )
        _log.info(
            "semantic_consolidation.merged",
            canonical=canonical.shelf_id,
            folded=sorted(folded),
            confidence=group.confidence,
        )
    return list(by_id.values())


# ----------------------------------------------------------------- internals


def _blocklist_set(cfg: SemanticConsolidationConfig) -> set[frozenset[str]]:
    return {frozenset(pair) for pair in cfg.permanent_block_list if len(pair) == 2}


def _vetoed_pairs(
    group: MergeGroup,
    shelves_by_id: dict[str, Shelf],
    blocked: set[frozenset[str]],
) -> list[tuple[str, str]]:
    """Member pairs (by shelf id) whose foodon_ids are block-listed together."""
    out: list[tuple[str, str]] = []
    members = group.members
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            fa = _foodon(members[i], shelves_by_id)
            fb = _foodon(members[j], shelves_by_id)
            if fa and fb and frozenset((fa, fb)) in blocked:
                out.append((members[i], members[j]))
    return out


def _foodon(shelf_id: str, shelves_by_id: dict[str, Shelf]) -> str | None:
    shelf = shelves_by_id.get(shelf_id)
    return shelf.foodon_id if shelf else None


def _pick_canonical(members: list[Shelf]) -> Shelf:
    """Shallowest member wins; tie-break shorter then earlier label."""
    return min(members, key=lambda s: (s.depth, len(s.label), s.label))
