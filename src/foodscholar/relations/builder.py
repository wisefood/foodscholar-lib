"""Orchestration for `fs.build_relations()`.

Pipeline::

    chunk_store.iter_chunks(batch_size)
       |
       +-- extract.py    per chunk: entities -> relations (2 LLM calls)
       +-- dedupe.py     semhash over surfaces + predicates, provenance kept
       +-- ground.py     surfaces -> ontology_id via fs.linker (one batch)
       +-- aggregate.py  collapse to one Relation per (subj, pred, obj)
       +-- persist.py    RelationStore + GraphStore

Resume is the store, not a savepoint directory: the reference implementation
wrote one JSON file per chunk to resume, which is a second and driftable source
of truth. Here, chunks already covered by a relation in the store are skipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.logging import get_logger
from foodscholar.relations.aggregate import build_relations, dedupe_graph
from foodscholar.relations.extract import extract_chunk, extractor_version
from foodscholar.relations.extracted import ExtractedGraph
from foodscholar.relations.ground import ground_surfaces

if TYPE_CHECKING:
    from foodscholar.config import RelationsConfig
    from foodscholar.io.chunk import Chunk, ChunkId
    from foodscholar.io.relation import Relation
    from foodscholar.storage.protocols import (
        ChunkStore,
        Linker,
        LLMClient,
        RelationStore,
    )

_log = get_logger("foodscholar.relations.builder")


def covered_chunk_ids(relation_store: RelationStore) -> set[ChunkId]:
    """Chunks already represented in the store — the resume set."""
    out: set[ChunkId] = set()
    for batch in relation_store.iter_relations(batch_size=1000):
        for relation in batch:
            out.update(relation.chunk_ids)
    return out


def extract_corpus(
    chunks: list[Chunk],
    *,
    llm: LLMClient,
    cfg: RelationsConfig,
    progress_every: int = 50,
) -> ExtractedGraph:
    """Run the extractor over `chunks`, accumulating one graph."""
    combined = ExtractedGraph()
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        graph = extract_chunk(
            chunk.text, chunk.chunk_id, llm=llm, max_tokens=cfg.max_tokens
        )
        combined.merge(graph)
        if progress_every and (index % progress_every == 0 or index == total):
            _log.info(
                "relations.extract.progress",
                processed=index,
                total=total,
                **combined.stats(),
            )
    return combined


def build(
    chunk_store: ChunkStore,
    *,
    llm: LLMClient,
    linker: Linker,
    cfg: RelationsConfig,
    chunk_ids: list[ChunkId] | None = None,
    relation_store: RelationStore | None = None,
    force: bool = False,
) -> tuple[list[Relation], dict[str, object]]:
    """Extract, dedupe, ground and aggregate. Returns `(relations, report)`.

    Pure with respect to storage — persistence is the caller's job, which keeps
    `dry_run` a one-line difference at the facade.
    """
    if chunk_ids is not None:
        wanted = set(chunk_ids)
        chunks = [c for c in chunk_store.scan() if c.chunk_id in wanted]
    else:
        chunks = chunk_store.scan()

    skipped = 0
    if not force and relation_store is not None:
        covered = covered_chunk_ids(relation_store)
        if covered:
            before = len(chunks)
            chunks = [c for c in chunks if c.chunk_id not in covered]
            skipped = before - len(chunks)

    if not chunks:
        _log.info("relations.build.nothing_to_do", skipped=skipped)
        return [], {"n_chunks": 0, "n_skipped": skipped, "n_relations": 0}

    graph = extract_corpus(chunks, llm=llm, cfg=cfg)
    entity_dedupe, predicate_dedupe = dedupe_graph(graph, cfg)

    # Ground the DEDUPLICATED surface set — fewer distinct strings to encode.
    canonical_surfaces = sorted(set(entity_dedupe.canonical.values()))
    grounding = ground_surfaces(
        canonical_surfaces,
        linker=linker,
        min_sim=cfg.grounding.min_sim,
        keep_nil=cfg.grounding.keep_nil,
    )

    relations = build_relations(
        graph,
        grounding=grounding,
        entity_dedupe=entity_dedupe,
        predicate_dedupe=predicate_dedupe,
        extractor_version=extractor_version(llm),
        keep_nil=cfg.grounding.keep_nil,
    )

    report: dict[str, object] = {
        "n_chunks": len(chunks),
        "n_skipped": skipped,
        "n_triples": len(graph.triples),
        "n_relations": len(relations),
        "n_fully_grounded": sum(1 for r in relations if r.is_fully_grounded),
        "n_surfaces_merged": entity_dedupe.n_merged,
        "n_predicates_merged": predicate_dedupe.n_merged,
        "dedupe_version": entity_dedupe.version,
        **grounding.report(),
    }
    _log.info("relations.build.done", **report)
    return relations, report
