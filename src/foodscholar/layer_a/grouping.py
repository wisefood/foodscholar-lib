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

from foodscholar.layer_a.facet import route_link_to_facet
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
