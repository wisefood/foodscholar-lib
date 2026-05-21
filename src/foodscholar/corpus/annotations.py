"""Merge external annotation/linking output into stored chunks.

Extraction and entity linking may be produced by another pipeline. This module
keeps the handoff simple: records are keyed by `chunk_id` and replace the
annotation fields on the matching chunk.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field

from foodscholar.io.chunk import ChunkId, EntityLink, Mention
from foodscholar.storage.protocols import ChunkStore


class ChunkAnnotation(BaseModel):
    chunk_id: ChunkId
    foodon_ids: list[str] = Field(default_factory=list)
    mentions: list[Mention] = Field(default_factory=list)
    entity_links: list[EntityLink] = Field(default_factory=list)
    enrichment_version: str = "external-annotations-v1"


def merge_annotations(
    chunk_store: ChunkStore,
    annotations: Iterable[ChunkAnnotation],
    *,
    strict: bool = True,
) -> int:
    """Replace annotation fields on chunks matched by `chunk_id`.

    Returns the number of chunks updated. In strict mode, an annotation for an
    unknown chunk raises `KeyError`; non-strict mode skips unknown chunk ids.
    """
    updated = 0
    for ann in annotations:
        chunk = chunk_store.get(ann.chunk_id)
        if chunk is None:
            if strict:
                raise KeyError(f"chunk not found for annotation: {ann.chunk_id}")
            continue
        chunk_store.update_annotations(
            ann.chunk_id,
            mentions=list(ann.mentions),
            entity_links=list(ann.entity_links),
            foodon_ids=sorted(set(ann.foodon_ids)),
            enrichment_version=ann.enrichment_version,
        )
        updated += 1
    return updated
