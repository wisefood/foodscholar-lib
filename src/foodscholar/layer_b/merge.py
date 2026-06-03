"""Greedy pair-assignment merge of similarity + relatedness candidates.

Per `layer_b_construction_brief.md` §4.4. For every (sim_i, rel_j) compute:

    combined_similarity = chunk_weight * J(chunks) + entity_weight * J(entities)

where J is Jaccard. Sort all decisions by `(-combined, sim_idx, rel_idx)`
for a deterministic tie-break, then greedily assign each candidate at most
once (highest combined first). Unmerged candidates pass through as
single-pass themes.

Output `themes` are plain dicts (not Pydantic `Theme` objects) because the
builder still needs to add labels, theme_ids, and metadata. Final Theme
construction lives in `builder.build_shelf_themes`.

Tradeoff documented: this is greedy, not optimal. The optimal version is
the Hungarian assignment problem (max-weight bipartite matching). At v1
scale (≤ ~30 candidates per shelf), the greedy heuristic is sufficient and
much simpler to audit. Switch only if hand audit shows pathological
mis-matches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.layer_b.models import MergeDecision

if TYPE_CHECKING:
    from foodscholar.config import MergeConfig
    from foodscholar.layer_b.models import ThemeCandidate



def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def merge_candidates(
    sim_cands: list[ThemeCandidate],
    rel_cands: list[ThemeCandidate],
    cfg: MergeConfig,
) -> tuple[list[dict], list[MergeDecision]]:
    """Returns `(themes_as_dicts, all_decisions)`.

    `themes_as_dicts` carries `chunk_ids`, `foodon_ids`, `discovery_pass`,
    `discovered_by`. The orchestrator fills in `label`, `theme_id`,
    `config_hash`, `version` before constructing Pydantic `Theme` records.

    `all_decisions` is the full cartesian product — every (i, j) pair gets
    a MergeDecision row, merged or not, so the audit artifact can answer
    "why didn't these merge?" without re-running.
    """
    decisions: list[MergeDecision] = []
    for i, s in enumerate(sim_cands):
        for j, r in enumerate(rel_cands):
            cj = _jaccard(s.chunk_ids, r.chunk_ids)
            ej = _jaccard(s.foodon_ids, r.foodon_ids)
            combined = cfg.chunk_weight * cj + cfg.entity_weight * ej
            decisions.append(
                MergeDecision(
                    similarity_candidate_idx=i,
                    relatedness_candidate_idx=j,
                    chunk_jaccard=cj,
                    entity_jaccard=ej,
                    combined_similarity=combined,
                    merged=combined >= cfg.dedupe_threshold,
                )
            )

    # Greedy assignment: highest combined first, ties broken by index order
    # for determinism (audit needs reproducible theme membership across runs).
    ordered = sorted(
        decisions,
        key=lambda d: (
            -d.combined_similarity,
            d.similarity_candidate_idx,
            d.relatedness_candidate_idx,
        ),
    )
    used_sim: set[int] = set()
    used_rel: set[int] = set()
    merged_pairs: list[tuple[int, int]] = []
    for d in ordered:
        if not d.merged:
            continue
        if (
            d.similarity_candidate_idx in used_sim
            or d.relatedness_candidate_idx in used_rel
        ):
            continue
        merged_pairs.append(
            (d.similarity_candidate_idx, d.relatedness_candidate_idx)
        )
        used_sim.add(d.similarity_candidate_idx)
        used_rel.add(d.relatedness_candidate_idx)

    themes: list[dict] = []
    # 1. Merged themes — union chunks + entities from both candidates
    for i, j in merged_pairs:
        s, r = sim_cands[i], rel_cands[j]
        themes.append(
            {
                "chunk_ids": s.chunk_ids | r.chunk_ids,
                "foodon_ids": s.foodon_ids | r.foodon_ids,
                "discovery_pass": "merged",
                "discovered_by": s.discovered_by,  # tie-break: sim wins
            }
        )
    # 2. Sim-only themes — propagate the source candidate's pass_name so that
    #    global_similarity candidates keep their pass_name rather than being
    #    relabelled "similarity".
    for i, s in enumerate(sim_cands):
        if i in used_sim:
            continue
        themes.append(
            {
                "chunk_ids": s.chunk_ids,
                "foodon_ids": s.foodon_ids,
                "discovery_pass": s.pass_name,
                "discovered_by": s.discovered_by,
                # Carried through so the caller can attach a per-shelf Pass-1
                # theme to its origin shelf. Transient — stripped before persist.
                "origin_shelf_id": s.origin_shelf_id,
            }
        )
    # 3. Rel-only themes
    for j, r in enumerate(rel_cands):
        if j in used_rel:
            continue
        themes.append(
            {
                "chunk_ids": r.chunk_ids,
                "foodon_ids": r.foodon_ids,
                "discovery_pass": "relatedness",
                "discovered_by": r.discovered_by,
            }
        )
    return themes, decisions


def merge_global_and_local_candidates(
    global_sim_cands: list[ThemeCandidate],
    rel_cands_by_shelf: dict[str, list[ThemeCandidate]],
    cfg: MergeConfig,
) -> tuple[list[dict], list[MergeDecision]]:
    """Merge one global similarity-candidate set against per-shelf relatedness
    candidates, producing theme dicts with ``shelf_ids: list[str]``.

    Algorithm:
      1. Flatten rel_cands_by_shelf, remembering origin shelf per candidate.
      2. Reuse existing ``merge_candidates(global_sim_cands, flat_rel, cfg)``.
      3. For each emitted theme:
         - merged: shelf_ids = union of contributing rel-cands' origin shelves
         - global_similarity (unmerged): shelf_ids = [] (orchestrator backfills)
         - relatedness (unmerged): shelf_ids = [origin_shelf]
    """
    flat_rel: list[ThemeCandidate] = []
    rel_origin_shelf: list[str] = []  # parallel to flat_rel
    for shelf_id, cands in rel_cands_by_shelf.items():
        for c in cands:
            flat_rel.append(c)
            rel_origin_shelf.append(shelf_id)

    themes, decisions = merge_candidates(global_sim_cands, flat_rel, cfg)

    out: list[dict] = []
    for t in themes:
        pass_kind = t["discovery_pass"]
        if pass_kind == "merged":
            # Find contributing rel-cands: their chunk_ids must be a subset of
            # the merged theme's chunk_ids (merge unions them).
            theme_chunks = set(t["chunk_ids"])
            contributing_shelves = {
                rel_origin_shelf[i]
                for i, rc in enumerate(flat_rel)
                if rc.chunk_ids and rc.chunk_ids.issubset(theme_chunks)
            }
            t = {**t, "shelf_ids": sorted(contributing_shelves)}
        elif pass_kind == "global_similarity":
            # Per-shelf Pass 1 tags the candidate with its origin shelf -> attach
            # there directly. Global Pass 1 leaves it None -> shelf_ids=[] so the
            # orchestrator backfills from the (genuinely cross-shelf) chunk union.
            origin = t.get("origin_shelf_id")
            t = {k: v for k, v in t.items() if k != "origin_shelf_id"}
            t["shelf_ids"] = [origin] if origin else []
        elif pass_kind == "relatedness":
            # Match by exact chunk_ids to find origin shelf
            origin = None
            t_chunks = set(t["chunk_ids"])
            for i, rc in enumerate(flat_rel):
                if rc.chunk_ids == t_chunks:
                    origin = rel_origin_shelf[i]
                    break
            t = {**t, "shelf_ids": [origin] if origin else []}
        out.append(t)
    return out, decisions
