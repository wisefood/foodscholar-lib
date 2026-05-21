"""Facet routing for Layer A.

Maps GLiNER entity-type labels to Layer A facets, and provides stub roots for
facets that have no corpus support. Single source of truth — `fs.build_entities`
imports the mapping from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.io.graph import Facet, Shelf

if TYPE_CHECKING:
    from foodscholar.io.chunk import EntityLink

# Maps GLiNER label vocabulary -> Layer A facet. Entries left out
# (Country, Measurement, Population, Time expression, "other") carry no facet
# hint and return None from `route_link_to_facet`.
ENTITY_TYPE_TO_FACET: dict[str, Facet] = {
    "food": "foods",
    "food component": "foods",
    "nutrient": "nutrients",
    "micronutrient": "nutrients",
    "macronutrient": "nutrients",
    "dietary supplement": "nutrients",
    "dietary pattern": "dietary_patterns",
    "medical condition": "health",
    "biomarker": "health",
    "allergen": "allergies",
}


def facet_for_entity_type(entity_type: str | None) -> Facet | None:
    if not entity_type:
        return None
    return ENTITY_TYPE_TO_FACET.get(entity_type)


def route_link_to_facet(link: EntityLink) -> Facet | None:
    """Route an EntityLink to a facet via its mention's entity_type.

    Returns None when no mapping exists. Foods is special: any FOODON ontology
    id also routes to foods regardless of entity_type, so the prototype's
    `entity_type='other'` mentions with FoodOn ids still populate foods.
    """
    facet = facet_for_entity_type(link.mention.entity_type)
    if facet is not None:
        return facet
    if link.ontology_id.startswith("FOODON:"):
        return "foods"
    return None


def stub_root(facet: Facet) -> Shelf:
    """Single-shelf stub for a facet with no corpus support.

    Sustainability has no entity_type that maps to it and no OBO ontology we
    project; it always becomes a stub root. Other facets become stub roots when
    their support table is empty (e.g. on a corpus with `entity_type='other'`
    for every mention).
    """
    label = {
        "foods": "Foods",
        "health": "Health",
        "sustainability": "Sustainability",
        "dietary_patterns": "Dietary patterns",
        "allergies": "Allergies",
        "nutrients": "Nutrients",
    }[facet]
    return Shelf(
        shelf_id=f"facet:{facet}",
        label=label,
        facet=facet,
        depth=0,
        foodon_id=None,
        parent_shelf_id=None,
        chunk_count=0,
        support_direct=0,
        support_lifted=0,
        see_also=[],
    )
