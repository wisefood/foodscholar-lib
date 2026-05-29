"""Pure orchestration of the annotate phase.

Batched: for every batch of N chunks we run a single `NER.extract_batch`
call, then `Linker.link_many` against the NEL index (also batched
internally), then a single batched `Embedder.embed` call over the batch's
texts. The enriched chunks are upserted back to the chunk store in one bulk
write per batch. No I/O outside the stores it's handed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.io.artifacts import ArtifactMeta
from foodscholar.io.chunk import EntityLink, Mention
from foodscholar.logging import get_logger
from foodscholar.versioning import make_artifact_meta

if TYPE_CHECKING:
    from foodscholar.config import FoodScholarConfig
    from foodscholar.storage.protocols import NER, ChunkStore, Embedder, Linker


_log = get_logger("foodscholar.annotate")

ENRICHMENT_VERSION = "annotate-v2"


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
    batch_size = config.annotate.batch_size

    chunks = chunk_store.scan()
    n_chunks = len(chunks)
    n_mentions = 0
    n_links = 0

    for start in range(0, n_chunks, batch_size):
        batch = chunks[start : start + batch_size]
        mentions_per_chunk = _extract_batch(ner, [c.text for c in batch])
        flat_mentions: list[Mention] = []
        offsets: list[int] = []
        for ms in mentions_per_chunk:
            offsets.append(len(flat_mentions))
            flat_mentions.extend(ms)
        offsets.append(len(flat_mentions))

        links_flat = _link_batch(linker, flat_mentions)
        n_mentions += len(flat_mentions)
        n_links += sum(1 for ln in links_flat if ln is not None)

        vecs = embedder.embed([c.text for c in batch])
        model_id = embedder.model_id

        enriched_batch = []
        for i, chunk in enumerate(batch):
            chunk_mentions = mentions_per_chunk[i]
            chunk_links_raw = links_flat[offsets[i] : offsets[i + 1]]
            chunk_links: list[EntityLink] = [ln for ln in chunk_links_raw if ln is not None]
            foodon_ids: list[str] = []
            for ln in chunk_links:
                if ln.ontology_id not in foodon_ids:
                    foodon_ids.append(ln.ontology_id)

            enriched_batch.append(
                chunk.model_copy(
                    update={
                        "mentions": chunk_mentions,
                        "entity_links": chunk_links,
                        "foodon_ids": foodon_ids,
                        "embedding": vecs[i],
                        "embedding_model": model_id,
                        "enrichment_version": ENRICHMENT_VERSION,
                    }
                )
            )

        chunk_store.upsert(enriched_batch)

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


def _extract_batch(ner: NER, texts: list[str]) -> list[list[Mention]]:
    """Use `ner.extract_batch` when available; fall back to per-text otherwise."""
    extract_batch = getattr(ner, "extract_batch", None)
    if callable(extract_batch):
        return extract_batch(texts)  # type: ignore[no-any-return]
    return [ner.extract(t) for t in texts]


def _link_batch(linker: Linker, mentions: list[Mention]) -> list[EntityLink | None]:
    link_many = getattr(linker, "link_many", None)
    if callable(link_many):
        return link_many(mentions)  # type: ignore[no-any-return]
    return [linker.link(m) for m in mentions]


def dry_run(
    text: str,
    *,
    ner: NER,
    linker: Linker,
) -> tuple[list, list[EntityLink]]:
    """Run NER+linker on a single text and return (mentions, links) without writes."""
    mentions = _extract_batch(ner, [text])[0]
    links_with_none = _link_batch(linker, mentions)
    links: list[EntityLink] = [ln for ln in links_with_none if ln is not None]
    return mentions, links
