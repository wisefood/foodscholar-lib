"""Support propagation: count chunks per ontology term, with ancestors.

Pure function — no I/O. Takes a chunk iterator and the ontology, returns a
`SupportTable` that the pruner consumes. Two metrics tracked side-by-side:

- `direct[term_id]`        — chunks mentioning this exact term.
- `with_descendants[id]`   — direct support summed across the term and every
                             descendant (the threshold metric per the spec).

Implemented as `dict[str, set[ChunkId]]` under the hood so any cross-shelf
union (e.g. the synthetic facet root's "chunks reachable through this facet")
can dedupe at chunk-level rather than summing already-deduped per-shelf
counts. The `.direct` / `.with_descendants` properties expose
`dict[str, int]` views (via `len(s)`) — pruner code that reads
`support.direct.get(term_id, 0)` keeps working unchanged.

Each (chunk, term) pair contributes at most one chunk-id to `direct`; the
ancestor walk dedupes per chunk so a chunk mentioning the same term twice
counts once, and a chunk mentioning a term + an ancestor of that term counts
once for each.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from foodscholar.layer_a.facet import route_link_to_facet

if TYPE_CHECKING:
    from foodscholar.config import LinkBlocklistEntry
    from foodscholar.io.chunk import Chunk, ChunkId
    from foodscholar.io.graph import Facet
    from foodscholar.ontology import FoodOnAPI


@dataclass
class SupportTable:
    """Per-term chunk-id sets, deduped per chunk.

    The integer-view properties (`direct`, `with_descendants`) are dict
    facades: `support.direct[term_id]` returns the chunk count (len of the
    underlying set) rather than the set itself. This preserves the existing
    pruner API (`support.direct.get(term_id, 0) -> int`) while letting
    higher-level code (e.g. `_ensure_single_root` in builder.py) reach the
    chunk-id sets for honest cross-shelf unioning.
    """

    direct_chunk_ids: dict[str, set[ChunkId]] = field(default_factory=dict)
    with_descendants_chunk_ids: dict[str, set[ChunkId]] = field(default_factory=dict)

    @property
    def direct(self) -> _IntCountView:
        return _IntCountView(self.direct_chunk_ids)

    @property
    def with_descendants(self) -> _IntCountView:
        return _IntCountView(self.with_descendants_chunk_ids)

    def __bool__(self) -> bool:
        return bool(self.with_descendants_chunk_ids)


class _IntCountView:
    """Read-only dict-like view: keys = term ids, values = chunk counts."""

    __slots__ = ("_d",)

    def __init__(self, d: dict[str, set]) -> None:
        self._d = d

    def get(self, term_id: str, default: int = 0) -> int:
        s = self._d.get(term_id)
        return len(s) if s is not None else default

    def __getitem__(self, term_id: str) -> int:
        return len(self._d[term_id])

    def __contains__(self, term_id: object) -> bool:
        return term_id in self._d

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def items(self):
        return ((tid, len(s)) for tid, s in self._d.items())

    def values(self):
        return (len(s) for s in self._d.values())

    def keys(self):
        return self._d.keys()


def collect_support(
    chunks: Iterable[Chunk],
    ontology: FoodOnAPI,
    *,
    min_link_confidence: float,
    facet: Facet,
    link_blocklist: list[LinkBlocklistEntry] | None = None,
) -> SupportTable:
    """Build the per-facet support table from a chunk iterator.

    A link contributes when:
      - `confidence >= min_link_confidence`
      - `route_link_to_facet(link) == facet`
      - its `ontology_id` is in the loaded ontology
      - `(mention.text.lower(), ontology_id)` is NOT in `link_blocklist`
        (NEL-drift filter for known polysemous surface forms)

    A chunk's `foodon_ids` denormalization is honored for the foods facet only:
    if `chunk.foodon_ids` is populated and `facet == 'foods'`, those ids count
    too — but ONLY for terms the chunk has no `entity_link` for. When a term is
    in both, the `entity_links` loop is authoritative (it applies the
    confidence floor and the blocklist); re-adding it from the bare-id
    `foodon_ids` list would silently undo a blocklist skip, since `foodon_ids`
    carries no surface text to match against. The `foodon_ids` path therefore
    only fills the gap it was designed for: prototype chunks whose terms have
    no per-mention `entity_link` at all.
    """
    table = SupportTable()
    # Indexed for O(1) lookup; case-insensitive on the surface side.
    blocklist_set = (
        {(e.surface.lower(), e.ontology_id) for e in link_blocklist}
        if link_blocklist
        else set()
    )

    for chunk in chunks:
        # Per-chunk dedupe set: each term contributes at most one direct count.
        seen_direct: set[str] = set()

        # Terms the chunk carries an entity_link for (any confidence/facet) —
        # the entity_links loop is the authority on these, so the foodon_ids
        # path below must not second-guess its filtering.
        linked_terms = {link.ontology_id for link in chunk.entity_links}

        for link in chunk.entity_links:
            if link.confidence < min_link_confidence:
                continue
            if route_link_to_facet(link) != facet:
                continue
            term_id = link.ontology_id
            if term_id not in ontology:
                continue
            if (link.mention.text.lower(), term_id) in blocklist_set:
                continue
            seen_direct.add(term_id)

        # `foodon_ids` is the denormalized list set by ingest. The
        # pre-computed-NEL path drops it onto chunks without an EntityLink
        # (since the prototype CSV has no per-mention metadata), so we have to
        # read it directly to populate the foods facet from that corpus. Skip
        # terms already covered by an entity_link: those were decided above,
        # blocklist included.
        if facet == "foods":
            for term_id in chunk.foodon_ids:
                if term_id in linked_terms:
                    continue
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
            table.direct_chunk_ids.setdefault(term_id, set()).add(chunk.chunk_id)
        for term_id in propagation_set:
            table.with_descendants_chunk_ids.setdefault(term_id, set()).add(chunk.chunk_id)

    return table
