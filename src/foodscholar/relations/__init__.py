"""Layer 0 — typed, corpus-grounded relations between entities.

Built by `fs.build_relations()` after `fs.build_entities()`; read through the
`fs.relations` namespace. See `foodscholar.io.relation.Relation`.
"""

from foodscholar.relations.aggregate import build_relations, dedupe_graph
from foodscholar.relations.builder import build, covered_chunk_ids, extract_corpus
from foodscholar.relations.dedupe import deduplicate_surfaces
from foodscholar.relations.extract import extract_chunk
from foodscholar.relations.extracted import ExtractedGraph
from foodscholar.relations.ground import ground_surfaces
from foodscholar.relations.persist import persist

__all__ = [
    "ExtractedGraph",
    "build",
    "build_relations",
    "covered_chunk_ids",
    "dedupe_graph",
    "deduplicate_surfaces",
    "extract_chunk",
    "extract_corpus",
    "ground_surfaces",
    "persist",
]
