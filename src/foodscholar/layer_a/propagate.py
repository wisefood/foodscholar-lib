"""Support propagation: count chunks per ontology term, with ancestors.

Pure function — no I/O. Takes a chunk iterator and the ontology, returns a
`SupportTable` that the pruner consumes. Two metrics tracked side-by-side:

- `direct[term_id]`        — chunks mentioning this exact term.
- `with_descendants[id]`   — direct support summed across the term and every
                             descendant (the threshold metric per the spec).

Each (chunk, term) pair contributes at most one count to `direct`; the
ancestor walk dedupes per chunk so a chunk mentioning the same term twice
counts once, and a chunk mentioning a term + an ancestor of that term counts
once for each. The threshold-metric `with_descendants` is then a roll-up of
`direct` across the closed-descendants set.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from foodscholar.layer_a.facet import route_link_to_facet

if TYPE_CHECKING:
    from foodscholar.io.chunk import Chunk
    from foodscholar.io.graph import Facet
    from foodscholar.ontology import FoodOnAPI


@dataclass
class SupportTable:
    direct: dict[str, int] = field(default_factory=dict)
    with_descendants: dict[str, int] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.with_descendants)


def collect_support(
    chunks: Iterable[Chunk],
    ontology: FoodOnAPI,
    *,
    min_link_confidence: float,
    facet: Facet,
) -> SupportTable:
    """Build the per-facet support table from a chunk iterator.

    A link contributes when:
      - `confidence >= min_link_confidence`
      - `route_link_to_facet(link) == facet`
      - its `ontology_id` is in the loaded ontology

    A chunk's `foodon_ids` denormalization is honored for the foods facet only:
    if `chunk.foodon_ids` is populated and `facet == 'foods'`, those ids count
    too (this is the cheap path the prototype's nel_loader produces).
    """
    table = SupportTable()

    for chunk in chunks:
        # Per-chunk dedupe set: each term contributes at most one direct count.
        seen_direct: set[str] = set()

        for link in chunk.entity_links:
            if link.confidence < min_link_confidence:
                continue
            if route_link_to_facet(link) != facet:
                continue
            term_id = link.ontology_id
            if term_id not in ontology:
                continue
            seen_direct.add(term_id)

        # `foodon_ids` is the denormalized list set by ingest. The
        # pre-computed-NEL path drops it onto chunks without an EntityLink
        # (since the prototype CSV has no per-mention metadata), so we have to
        # read it directly to populate the foods facet from that corpus.
        if facet == "foods":
            for term_id in chunk.foodon_ids:
                if term_id in ontology:
                    seen_direct.add(term_id)

        if not seen_direct:
            continue

        # Per-chunk ancestor union — propagate this chunk to each direct term's
        # closed-ancestor set once.
        propagation_set: set[str] = set(seen_direct)
        for term_id in seen_direct:
            for ancestor in ontology.id_to_ancestors(term_id):
                if ancestor in ontology:
                    propagation_set.add(ancestor)

        for term_id in seen_direct:
            table.direct[term_id] = table.direct.get(term_id, 0) + 1
        for term_id in propagation_set:
            table.with_descendants[term_id] = table.with_descendants.get(term_id, 0) + 1

    return table
