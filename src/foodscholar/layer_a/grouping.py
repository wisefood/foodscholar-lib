"""Bottom-up + LLM semantic grouping construction for a Layer-A facet.

Opt-in alternative to the top-down `prune` path (see methods_layer_a_rework_brief).
Every corpus-mentioned leaf is kept (coverage by construction); the LLM proposes
~N human food groups anchored to real FoodOn ids and assigns each leaf to a group
by label. Each group becomes one flat Shelf; a leaf's foodon_id is recorded on its
group shelf's `see_also` so the existing attach resolver routes its chunks there.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from foodscholar.io.graph import Shelf
from foodscholar.layer_a.facet import route_link_to_facet, stub_root
from foodscholar.layer_a.prune import shelf_id_for_foodon
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from foodscholar.io.chunk import Chunk
    from foodscholar.io.graph import Facet
    from foodscholar.ontology import FoodOnAPI

_log = get_logger("foodscholar.layer_a.grouping")


def collect_leaf_chunks(
    chunks: Iterable[Chunk],
    ontology: FoodOnAPI,
    *,
    facet: Facet,
    min_link_confidence: float,
) -> dict[str, set[str]]:
    """Map each mentioned FoodOn leaf id -> set of chunk ids.

    A term contributes when it is in the ontology and either appears in the
    chunk's `foodon_ids` denorm or in an `entity_link` routing to `facet` with
    confidence >= floor. Distinct chunk-id sets (not counts) so group sizes can
    be deduped as a union later.
    """
    leaf_chunks: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        seen: set[str] = set()
        for fid in (getattr(chunk, "foodon_ids", None) or []):
            if fid in ontology:
                seen.add(fid)
        for link in (getattr(chunk, "entity_links", None) or []):
            if link.confidence < min_link_confidence:
                continue
            if link.ontology_id in ontology and route_link_to_facet(link) == facet:
                seen.add(link.ontology_id)
        for fid in seen:
            leaf_chunks[fid].add(chunk.chunk_id)
    return dict(leaf_chunks)


_SYN_BAD = re.compile(r"\d|\(|,|;|:")  # codes / parenthetical / list-y synonyms


def _clean_synonym(fid: str, ontology: FoodOnAPI) -> str | None:
    base = re.sub(r"\s+food product$", "", (ontology.id_to_label(fid) or "")).lower()
    cands = [
        s for s in ontology.id_to_synonyms(fid, include_related=False)
        if s and not _SYN_BAD.search(s) and 2 <= len(s) <= 30
    ]
    cands.sort(key=len)
    for s in cands:
        if s.lower() != base:
            return s
    return cands[0] if cands else None


def clean_label(fid: str, ontology: FoodOnAPI) -> str:
    """Display label: clean FoodOn synonym -> strip ' food product' suffix -> raw label."""
    syn = _clean_synonym(fid, ontology)
    if syn:
        return syn
    lbl = ontology.id_to_label(fid) or fid
    return re.sub(r"\s+food product$", "", lbl) or lbl


# ---------------------------------------------------------------------------
# Group dataclass + LLM-driven group proposal
# ---------------------------------------------------------------------------


@dataclass
class Group:
    display_name: str
    anchor_foodon_ids: list[str] = field(default_factory=list)


def _split_concepts(group_name: str) -> list[str]:
    parts = re.split(r"\s+and\s+|\s*,\s*|\s*/\s*", group_name.strip())
    return [p.strip().lower() for p in parts if p.strip()]


def _anchor_for_concept(concept: str, ontology: FoodOnAPI) -> str | None:
    singular = concept.rstrip("s")
    for cand in (concept, singular, concept + " food product", singular + " food product"):
        fid = ontology.name_to_id(cand)
        if fid:
            return fid
    for hit in ontology.search(concept, limit=12):
        lbl = (ontology.id_to_label(hit) or "").lower()
        clean = re.sub(r"\s+food product$", "", lbl)
        if clean in {concept, singular} and "(" not in lbl:
            return hit
    return None


def propose_groups(
    ontology: FoodOnAPI,
    llm,
    *,
    leaf_freq: dict[str, int],
    n_groups: int,
    frozen=None,
) -> list[Group]:
    """Return groups (display name + real FoodOn anchor ids).

    If `frozen` (list of config.FrozenGroup) is given, use it verbatim (no LLM
    call). Otherwise ask the LLM for ~n_groups human food-group names and resolve
    each to real FoodOn anchor ids; names with no anchor are dropped.
    """
    if frozen is not None:
        return [Group(g.display_name, list(g.anchor_foodon_ids)) for g in frozen]

    schema = {
        "type": "object",
        "properties": {"groups": {"type": "array", "items": {"type": "string"}}},
        "required": ["groups"],
    }
    sample = ", ".join(
        ontology.id_to_label(fid) or fid
        for fid, _ in sorted(leaf_freq.items(), key=lambda kv: -kv[1])[:50]
    )
    prompt = (
        f"Propose {n_groups} intuitive, MUTUALLY-EXCLUSIVE top-level food groups for "
        f"browsing a nutrition knowledge base (human category names like 'Vegetables', "
        f"'Dairy and Eggs', 'Fish and Seafood', 'Grains and Bread'). Use food TYPES, "
        f"not cross-cutting attributes like 'processed foods'. Frequent corpus foods "
        f"for context:\n{sample}\n\nReturn JSON {{\"groups\": [\"...\"]}}."
    )
    try:
        names = (llm.generate_json(prompt, schema, max_tokens=400) or {}).get("groups", [])
    except Exception as exc:
        _log.warning("grouping.propose_failed", error=str(exc))
        names = []

    groups: list[Group] = []
    for nm in names:
        anchors: list[str] = []
        for concept in _split_concepts(nm):
            fid = _anchor_for_concept(concept, ontology)
            if fid is not None and fid not in anchors:
                anchors.append(fid)
        if anchors:
            groups.append(Group(nm, anchors))
    return groups


def assign_leaves(
    leaf_ids: list[str],
    groups: list[Group],
    ontology: FoodOnAPI,
    llm,
    *,
    batch_size: int,
) -> dict[str, str | None]:
    """Assign each leaf id to a group display_name (or None) by LABEL via the LLM.

    Batched; defensive — invalid/missing group names map to None (unassigned →
    the leaf keeps its own shelf, preserving coverage). Assignment is by the
    leaf's clean label, NOT is-a ancestry.
    """
    group_names = [g.display_name for g in groups]
    valid = set(group_names)
    label_to_ids: dict[str, list[str]] = defaultdict(list)
    for fid in leaf_ids:
        label_to_ids[clean_label(fid, ontology)].append(fid)
    labels = sorted(label_to_ids)

    schema = {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "food": {"type": "string"},
                        "group": {"type": "string"},
                    },
                    "required": ["food", "group"],
                },
            }
        },
        "required": ["assignments"],
    }

    label_group: dict[str, str] = {}
    for i in range(0, len(labels), batch_size):
        batch = labels[i : i + batch_size]
        prompt = (
            f"Assign each food to ONE of these groups: {', '.join(group_names)}, "
            f"or '(other)' if none fits.\nFoods:\n"
            + "\n".join(f"  - {lbl}" for lbl in batch)
            + '\n\nReturn JSON {"assignments": [{"food": "<food>", "group": "<group>"}, ...]}'
            " for every food."
        )
        try:
            obj = llm.generate_json(prompt, schema, max_tokens=4096)
        except Exception as exc:
            _log.warning("grouping.assign_failed", batch_start=i, error=str(exc))
            obj = {}
        for a in (obj or {}).get("assignments", []):
            food, group = a.get("food"), a.get("group")
            if food in label_to_ids and group in valid:
                label_group[food] = group

    return {fid: label_group.get(clean_label(fid, ontology)) for fid in leaf_ids}


# ---------------------------------------------------------------------------
# Task 7 — integration orchestrator
# ---------------------------------------------------------------------------

_FACET_ROOT_LABELS = {
    "foods": "Foods", "health": "Health", "sustainability": "Sustainability",
    "dietary_patterns": "Dietary patterns", "allergies": "Allergies", "nutrients": "Nutrients",
}


def build_grouped_shelves(
    chunks: Iterable[Chunk],
    ontology: FoodOnAPI,
    cfg,  # BottomUpGroupingConfig
    *,
    facet: Facet,
    min_link_confidence: float,
    llm=None,
) -> list[Shelf]:
    """Bottom-up + LLM-grouping shelves for one facet.

    Emits: 1 synthetic facet root (depth 0); one group shelf per non-empty group
    (depth 1, display_label set, member leaf foodon_ids in see_also); one kept-leaf
    shelf (depth 1) per leaf not assigned to any group. Every mentioned leaf is
    represented — coverage by construction.
    """
    leaf_chunks = collect_leaf_chunks(
        chunks, ontology, facet=facet, min_link_confidence=min_link_confidence
    )
    leaf_chunks = {fid: cs for fid, cs in leaf_chunks.items() if len(cs) >= cfg.min_leaf_support}
    if not leaf_chunks:
        return [stub_root(facet)]

    leaf_freq = {fid: len(cs) for fid, cs in leaf_chunks.items()}
    groups = propose_groups(
        ontology, llm, leaf_freq=leaf_freq, n_groups=cfg.n_groups, frozen=cfg.frozen_groups
    )
    if groups:
        assignment = assign_leaves(
            list(leaf_chunks), groups, ontology, llm, batch_size=cfg.assign_batch_size
        )
    else:
        assignment = {fid: None for fid in leaf_chunks}

    root_id = f"facet:{facet}"
    shelves: list[Shelf] = []
    all_chunks: set[str] = set()

    group_members: dict[str, list[str]] = defaultdict(list)
    for fid, gname in assignment.items():
        if gname is not None:
            group_members[gname].append(fid)

    # Anchors that actually become group shelves. A kept-leaf shelf must not be
    # emitted for any of these fids — it would collide on shelf_id with the group
    # shelf (the graph store MERGEs on shelf_id, silently overwriting one). The
    # group shelf wins; an anchor's own direct chunks are folded into it below.
    group_anchor_ids: set[str] = {
        g.anchor_foodon_ids[0] for g in groups if group_members.get(g.display_name)
    }

    for g in groups:
        members = group_members.get(g.display_name, [])
        if not members:
            continue
        anchor = g.anchor_foodon_ids[0]
        chunk_ids: set[str] = set()
        for fid in members:
            chunk_ids |= leaf_chunks.get(fid, set())
        # Fold the anchor's own direct chunks in too — the anchor may itself be a
        # mentioned leaf that the LLM left unassigned; those chunks belong here.
        chunk_ids |= leaf_chunks.get(anchor, set())
        all_chunks |= chunk_ids
        see_also = set(members)
        if anchor in leaf_chunks:
            see_also.add(anchor)  # anchor-as-leaf is represented by this group shelf
        shelves.append(Shelf(
            shelf_id=shelf_id_for_foodon(anchor),
            label=ontology.id_to_label(anchor) or anchor,
            display_label=g.display_name,
            facet=facet,
            depth=1,
            foodon_id=anchor,
            parent_shelf_id=root_id,
            chunk_count=len(chunk_ids),
            support_direct=0,
            support_lifted=len(chunk_ids),
            see_also=sorted(see_also),
        ))

    for fid, gname in assignment.items():
        if gname is not None:
            continue
        if fid in group_anchor_ids:
            continue  # already represented by its group shelf (no shelf_id collision)
        cs = leaf_chunks.get(fid, set())
        all_chunks |= cs
        shelves.append(Shelf(
            shelf_id=shelf_id_for_foodon(fid),
            label=ontology.id_to_label(fid) or fid,
            display_label=clean_label(fid, ontology),
            facet=facet,
            depth=1,
            foodon_id=fid,
            parent_shelf_id=root_id,
            chunk_count=len(cs),
            support_direct=len(cs),
            support_lifted=0,
            see_also=[],
        ))

    root = Shelf(
        shelf_id=root_id, label=_FACET_ROOT_LABELS[facet], facet=facet, depth=0,
        foodon_id=None, parent_shelf_id=None, chunk_count=len(all_chunks),
        support_direct=0, support_lifted=len(all_chunks), see_also=[],
    )
    return [root, *sorted(shelves, key=lambda s: (s.display_label or s.label).lower())]
