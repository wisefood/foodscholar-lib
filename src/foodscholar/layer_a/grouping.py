"""Bottom-up + LLM semantic grouping construction for a Layer-A facet.

Opt-in alternative to the top-down `prune` path (see methods_layer_a_rework_brief).
Every corpus-mentioned leaf is kept (coverage by construction); the LLM proposes
~N human food groups anchored to real FoodOn ids and assigns each leaf to a group
by label. Each group becomes one flat Shelf; a leaf's foodon_id is recorded on its
group shelf's `see_also` so the existing attach resolver routes its chunks there.
"""

from __future__ import annotations

import re  # noqa: F401  # used by Tasks 4-7 extending this module
from collections import defaultdict
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
