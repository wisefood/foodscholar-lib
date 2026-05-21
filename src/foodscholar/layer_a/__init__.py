"""Layer A — backbone projection of the ontology onto the corpus."""

from foodscholar.layer_a.builder import (
    build_layer_a,
    build_shelves,
    shelf_id_for_foodon,
)
from foodscholar.layer_a.facet import (
    ENTITY_TYPE_TO_FACET,
    facet_for_entity_type,
    route_link_to_facet,
    stub_root,
)
from foodscholar.layer_a.propagate import SupportTable, collect_support
from foodscholar.layer_a.prune import prune

__all__ = [
    "ENTITY_TYPE_TO_FACET",
    "SupportTable",
    "build_layer_a",
    "build_shelves",
    "collect_support",
    "facet_for_entity_type",
    "prune",
    "route_link_to_facet",
    "shelf_id_for_foodon",
    "stub_root",
]
