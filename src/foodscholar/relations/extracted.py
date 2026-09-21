"""The extractor's working type — in memory, one batch at a time.

This is a deliberately slim rewrite of kggen's `ExtendedGraph`. It is **not** a
library data contract: it never persists, and it never appears in a public
signature. `io.relation.Relation` is the contract; this is scaffolding between
the LLM call and `relations.persist`.

The upstream `ExtendedGraph` carries ~150 lines of stale-provenance repair
(loose normalization, `SequenceMatcher` predicate matching, alias-cluster
rewriting) that exists because saved chunk-graph JSON files drifted out of sync
with their dedup clusters *on disk*. foodscholar has stores rather than
savepoint files, so importing that machinery would import a class of bug the
library does not have.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from foodscholar.io.chunk import ChunkId

Triple = tuple[str, str, str]


@dataclass
class ExtractedGraph:
    """Raw LLM output for one or more chunks, before grounding and dedup.

    Entities and predicates here are **free-text surfaces**, not ontology ids.
    `relations.ground` is what turns them into a `Relation`.
    """

    entities: set[str] = field(default_factory=set)
    triples: set[Triple] = field(default_factory=set)
    triple_chunks: dict[Triple, set[ChunkId]] = field(
        default_factory=lambda: defaultdict(set)
    )
    """Provenance — the whole reason this pipeline exists."""
    entity_chunks: dict[str, set[ChunkId]] = field(
        default_factory=lambda: defaultdict(set)
    )
    triple_mentions: dict[Triple, int] = field(default_factory=lambda: defaultdict(int))
    """Times each triple was emitted, before any collapse."""

    def add_triple(self, triple: Triple, chunk_id: ChunkId) -> None:
        subject, _predicate, obj = triple
        self.triples.add(triple)
        self.entities.add(subject)
        self.entities.add(obj)
        self.triple_chunks[triple].add(chunk_id)
        self.triple_mentions[triple] += 1
        self.entity_chunks[subject].add(chunk_id)
        self.entity_chunks[obj].add(chunk_id)

    def add_entity(self, surface: str, chunk_id: ChunkId) -> None:
        self.entities.add(surface)
        self.entity_chunks[surface].add(chunk_id)

    def merge(self, other: ExtractedGraph) -> None:
        self.entities |= other.entities
        self.triples |= other.triples
        for triple, chunks in other.triple_chunks.items():
            self.triple_chunks[triple] |= chunks
        for surface, chunks in other.entity_chunks.items():
            self.entity_chunks[surface] |= chunks
        for triple, count in other.triple_mentions.items():
            self.triple_mentions[triple] += count

    def surfaces(self) -> list[str]:
        """Distinct entity surfaces, sorted.

        Sorted because everything downstream of the LLM must be deterministic,
        and because the grounding step batches over this list — a stable order
        makes the linker's batch reproducible.
        """
        return sorted(self.entities)

    def predicates(self) -> list[str]:
        return sorted({p for _, p, _ in self.triples})

    @property
    def chunk_ids(self) -> set[ChunkId]:
        out: set[ChunkId] = set()
        for chunks in self.triple_chunks.values():
            out |= chunks
        return out

    def stats(self) -> dict[str, int]:
        return {
            "entities": len(self.entities),
            "triples": len(self.triples),
            "predicates": len(self.predicates()),
            "chunks": len(self.chunk_ids),
        }
