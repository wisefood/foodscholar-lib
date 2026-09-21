"""KGGEN Extended — Passage-aware Knowledge Graph Generation.

Extends the original KGGen library with passage-level provenance tracking.
Each triplet in the knowledge graph is linked to the source passage(s) it was
extracted from. After deduplication, triplets can be connected to multiple
passages (when aliases from different passages collapse to the same canonical
entity/triplet).

Key classes:
  - ExtendedGraph: passage-aware data model
  - ExtendedKGGen: orchestrator with passage tracking through extraction,
    aggregation, and deduplication.

Key exports:
  - export.py: GraphML (with passage nodes + Source edges), triples CSV/JSONL,
    passages CSV/JSON.
"""

from kggen_extended.models import ExtendedGraph
from kggen_extended.kg_gen_extended import ExtendedKGGen
from kggen_extended.steps._3_deduplicate import DeduplicateMethod
from kggen_extended.export import (
    export_all,
    export_graphml,
    export_triples,
    export_passages,
    to_nx_extended,
    to_nx_extended_with_passage_ids,
)

__all__ = [
    "ExtendedGraph",
    "ExtendedKGGen",
    "DeduplicateMethod",
    "export_all",
    "export_graphml",
    "export_triples",
    "export_passages",
    "to_nx_extended",
    "to_nx_extended_with_passage_ids",
]
