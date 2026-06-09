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


# Maps OBO ontology prefix -> Layer A facet, used as a FALLBACK when the NER
# left `entity_type='other'` (so `ENTITY_TYPE_TO_FACET` can't fire). Without
# this, every non-FOODON link routes nowhere and the non-food facets collapse to
# their stub root. The imported OBO ontologies live in `foodon.owl`, so these
# entities still have a hierarchy to project onto. Trade-off: prefix routing
# inherits some NEL mislink noise (documented in `route_link_to_facet`); it's
# the pragmatic way to populate facets from entity_type='other' annotations.
PREFIX_TO_FACET: dict[str, Facet] = {
    # nutrients — chemicals, nutrition descriptors, units, proteins
    "CHEBI": "nutrients", "CDNO": "nutrients", "ONS": "nutrients",
    "UO": "nutrients", "PR": "nutrients",
    # health — anatomy, phenotypes/qualities, disease, ancestry, biological process
    "UBERON": "health", "MONDO": "health", "HP": "health", "NCIT": "health",
    "GO": "health", "OBI": "health", "PATO": "health", "HANCESTRO": "health",
    # sustainability — environments, ecology, geography, agronomy, plants
    "ENVO": "sustainability", "ECOCORE": "sustainability", "GAZ": "sustainability",
    "AGRO": "sustainability", "FAO": "sustainability", "PO": "sustainability",
    "FLOPO": "sustainability",
}


def facet_for_entity_type(entity_type: str | None) -> Facet | None:
    if not entity_type:
        return None
    return ENTITY_TYPE_TO_FACET.get(entity_type)


def facet_for_prefix(ontology_id_or_prefix: str | None) -> Facet | None:
    """Route by OBO prefix (`CHEBI:16646` or just `CHEBI`). None if unmapped."""
    if not ontology_id_or_prefix:
        return None
    prefix = ontology_id_or_prefix.split(":", 1)[0]
    return PREFIX_TO_FACET.get(prefix)


def route_link_to_facet(link: EntityLink) -> Facet | None:
    """Route an EntityLink to a facet via its mention's entity_type.

    Returns None when no mapping exists. Foods is special: any FOODON ontology
    id also routes to foods regardless of entity_type, so the prototype's
    `entity_type='other'` mentions with FoodOn ids still populate foods.

    Precedence: explicit `entity_type` → FOODON-id-means-foods → OBO-prefix
    fallback (`PREFIX_TO_FACET`). The prefix fallback exists because the
    prototype NER leaves `entity_type='other'`, so without it the non-food
    facets never populate. The known cost: the upstream linker mis-assigns
    across OBOs (e.g. "heart disease" → UBERON, "breakfast" → NCIT), so prefix
    routing inherits some noise. The cleaner long-term fix is re-annotation with
    a NER that populates entity_type — see BRIEF §15 (MONDO/CHEBI v2) — but
    prefix routing is the pragmatic way to use the OBO links we already have.
    """
    facet = facet_for_entity_type(link.mention.entity_type)
    if facet is not None:
        return facet
    if link.ontology_id.startswith("FOODON:"):
        return "foods"
    # Fallback: route the non-FOODON OBO link by its ontology prefix so the
    # non-food facets populate even when `entity_type='other'` (the prototype
    # NER case). See PREFIX_TO_FACET for the trade-off note.
    return facet_for_prefix(link.ontology_id)


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
