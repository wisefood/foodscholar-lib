"""Grounding — map extracted entity surfaces onto ontology ids.

This is the conceptual core of the kggen integration. kggen's entities are raw
LLM strings; foodscholar's are FoodOn ids::

    kggen:       ("olive oil", "reduces", "LDL cholesterol")   # free text
    foodscholar: Entity(ontology_id="FOODON:03301710", ...)    # grounded

Left alone the repository would hold two disjoint entity universes. Running
every triple endpoint through the **existing** `fs.linker` — the same
`HNSWLinker` over the same FoodOn index that produced every `EntityLink` in the
corpus — makes a relation an edge between the same `Entity` records Layer A
projects and Layer B clusters.

Endpoints that do not clear the threshold are kept as ``NIL:<slug>`` rather
than dropped: discarding them would silently delete every relation touching a
concept FoodOn lacks, which for a nutrition corpus is most biomarkers,
hormones and physiological processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from foodscholar.io.chunk import Mention
from foodscholar.io.relation import nil_id
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.storage.protocols import Linker

_log = get_logger("foodscholar.relations.ground")

# The linker applies a semantic-type gate; "food" is the permissive type that
# lets a candidate through, matching `HNSWLinker.dry_run`.
_GROUNDING_ENTITY_TYPE = "food"


@dataclass
class GroundingResult:
    """`surface -> ontology_id` plus the diagnostics that make gaps visible."""

    ids: dict[str, str] = field(default_factory=dict)
    linked: set[str] = field(default_factory=set)
    confidence: dict[str, float] = field(default_factory=dict)

    def id_for(self, surface: str) -> str:
        return self.ids.get(surface, nil_id(surface))

    def is_linked(self, surface: str) -> bool:
        return surface in self.linked

    @property
    def n_surfaces(self) -> int:
        return len(self.ids)

    @property
    def link_rate(self) -> float:
        return len(self.linked) / len(self.ids) if self.ids else 0.0

    def nil_surfaces(self) -> list[str]:
        return sorted(s for s in self.ids if s not in self.linked)

    def report(self, *, top_n: int = 20) -> dict[str, object]:
        """Diagnostics worth logging: the top NIL surfaces are the highest-value
        output of this phase — they name what the ontology is missing."""
        nils = self.nil_surfaces()
        return {
            "n_surfaces": self.n_surfaces,
            "n_linked": len(self.linked),
            "link_rate": round(self.link_rate, 4),
            "n_nil": len(nils),
            "top_nil_surfaces": nils[:top_n],
        }


def ground_surfaces(
    surfaces: list[str],
    *,
    linker: Linker,
    min_sim: float = 0.70,
    keep_nil: bool = True,
) -> GroundingResult:
    """Link every distinct surface in one batched call.

    Batching matters: the same surface recurs across hundreds of triples, so
    linking per triple would be two orders of magnitude more encoder work.
    `HNSWLinker.link_many` does one encode plus one kNN for the whole batch.
    """
    result = GroundingResult()
    ordered = sorted(set(s for s in surfaces if s and s.strip()))
    if not ordered:
        return result

    mentions = [
        Mention(
            text=surface,
            start=0,
            end=len(surface),
            score=1.0,
            ner_model_version="relations-grounding",
            entity_type=_GROUNDING_ENTITY_TYPE,  # type: ignore[arg-type]
        )
        for surface in ordered
    ]

    link_many = getattr(linker, "link_many", None)
    links = (
        link_many(mentions)
        if callable(link_many)
        else [linker.link(m) for m in mentions]
    )

    for surface, link in zip(ordered, links, strict=True):
        if link is not None and link.confidence >= min_sim:
            result.ids[surface] = link.ontology_id
            result.linked.add(surface)
            result.confidence[surface] = link.confidence
        elif keep_nil:
            result.ids[surface] = nil_id(surface)
        # keep_nil=False simply omits the surface; callers drop those triples

    _log.info("relations.grounding.done", **result.report())
    return result
