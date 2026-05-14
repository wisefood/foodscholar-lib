"""Pure orchestration of the annotate phase.

For every chunk: run NER → link mentions → embed text → write back to the
chunk store. Stamps an ArtifactMeta on completion. No I/O outside the stores
it's handed; all dependencies are injected so the runner is unit-testable
without any mocks beyond the protocols.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.annotate.embedder import SourceTypeRouter
from foodscholar.io.artifacts import ArtifactMeta
from foodscholar.io.chunk import EntityLink
from foodscholar.logging import get_logger
from foodscholar.versioning import make_artifact_meta

if TYPE_CHECKING:
    from foodscholar.config import FoodScholarConfig
    from foodscholar.storage.protocols import NER, ChunkStore, Embedder, Linker


_log = get_logger("foodscholar.annotate")

ENRICHMENT_VERSION = "annotate-v1"


def run(
    chunk_store: ChunkStore,
    *,
    ner: NER,
    linker: Linker,
    embedder: Embedder,
    config: FoodScholarConfig,
) -> ArtifactMeta:
    """Annotate every chunk in `chunk_store` and write the enriched copies back.

    Idempotent: re-running over an already-annotated chunk replaces its
    mentions/links/embedding with the freshest set. The Pydantic models are
    frozen, so writes go through `model_copy(update=...)`.
    """
    router = embedder if isinstance(embedder, SourceTypeRouter) else None

    chunks = chunk_store.scan()
    n_chunks = len(chunks)
    n_mentions = 0
    n_links = 0

    for chunk in chunks:
        mentions = ner.extract(chunk.text)
        n_mentions += len(mentions)

        links: list[EntityLink] = []
        foodon_ids: list[str] = []
        for m in mentions:
            link = linker.link(m)
            if link is None:
                continue
            links.append(link)
            if link.ontology_id not in foodon_ids:
                foodon_ids.append(link.ontology_id)
        n_links += len(links)

        if router is not None:
            vec, model_id = router.embed_chunk(chunk.text, chunk.source_type)
        else:
            [vec] = embedder.embed([chunk.text])
            model_id = embedder.model_id

        enriched = chunk.model_copy(
            update={
                "mentions": mentions,
                "entity_links": links,
                "foodon_ids": foodon_ids,
                "embedding": vec,
                "embedding_model": model_id,
                "enrichment_version": ENRICHMENT_VERSION,
            }
        )
        chunk_store.upsert([enriched])

    meta = make_artifact_meta(
        phase="annotate",
        config=config,
        record_count=n_chunks,
    )
    _log.info(
        "annotate.done",
        n_chunks=n_chunks,
        n_mentions=n_mentions,
        n_links=n_links,
        coverage=(n_links / n_mentions) if n_mentions else 0.0,
        artifact_id=meta.artifact_id,
        config_hash=meta.config_hash,
    )
    return meta


def dry_run(
    text: str,
    *,
    ner: NER,
    linker: Linker,
) -> tuple[list, list[EntityLink]]:
    """Run NER+linker on a single text and return (mentions, links) without writes.

    Useful in notebooks and tests: probe what the linker would do for an
    arbitrary string without going through the full pipeline.
    """
    mentions = ner.extract(text)
    links: list[EntityLink] = []
    for m in mentions:
        link = linker.link(m)
        if link is not None:
            links.append(link)
    return mentions, links
