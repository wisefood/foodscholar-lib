"""FoodOn ontology loader + lookup API.

Public surface:
  load_ontology(path, cache_path=...) -> list[OntologyTerm]
  FoodOnAPI(terms)                    -> read-only lookup surface

Most users access the ontology via `fs.ontology` rather than constructing
either of these directly.
"""

from foodscholar.ontology.api import FoodOnAPI
from foodscholar.ontology.foodon import load_ontology

__all__ = ["FoodOnAPI", "load_ontology"]
