"""In-memory implementations of the ChunkStore and GraphStore protocols.

Used by unit tests and by `foodscholar init` when running without ES/Neo4j.
Behavior is intentionally simple: BM25 is approximated by token-overlap scoring
and vector similarity by cosine. Hybrid combines them via reciprocal rank fusion.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

from foodscholar.io.chunk import Chunk, ChunkId, EntityLink, Mention
from foodscholar.io.entity import Entity
from foodscholar.io.graph import Card, Shelf, ShelfId, Theme, ThemeId
from foodscholar.io.ontology import OntologyId

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _rrf(ranks: list[list[ChunkId]], k: int = 60) -> list[ChunkId]:
    scores: dict[ChunkId, float] = defaultdict(float)
    for ranking in ranks:
        for rank, cid in enumerate(ranking):
            scores[cid] += 1.0 / (k + rank + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


class InMemoryChunkStore:
    """Stores chunks in a dict. Search is a toy hybrid of token overlap + cosine."""

    def __init__(self) -> None:
        self._chunks: dict[ChunkId, Chunk] = {}

    def init(self) -> None:
        """No-op — there is nothing to provision for an in-memory store."""
        return

    def upsert(self, chunks: Iterable[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.chunk_id] = c

    def get(self, chunk_id: ChunkId) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def get_many(self, chunk_ids: list[ChunkId]) -> list[Chunk]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def search(
        self,
        query: str,
        theme_ids: list[ThemeId] | None = None,
        shelf_ids: list[ShelfId] | None = None,
        k: int = 10,
        use_vector: bool = True,
        use_bm25: bool = True,
    ) -> list[Chunk]:
        candidates = list(self._chunks.values())
        if shelf_ids:
            wanted = set(shelf_ids)
            candidates = [c for c in candidates if wanted.intersection(c.shelf_ids)]
        if theme_ids:
            wanted = set(theme_ids)
            candidates = [c for c in candidates if wanted.intersection(c.theme_ids)]

        if not candidates:
            return []

        q_tokens = set(_tokenize(query))
        bm25_ranking: list[ChunkId] = []
        if use_bm25 and q_tokens:
            scored = [
                (c.chunk_id, len(q_tokens.intersection(_tokenize(c.text))))
                for c in candidates
            ]
            scored = [s for s in scored if s[1] > 0]
            scored.sort(key=lambda x: x[1], reverse=True)
            bm25_ranking = [cid for cid, _ in scored]

        vec_ranking: list[ChunkId] = []
        if use_vector:
            with_embed = [c for c in candidates if c.embedding]
            if with_embed:
                pivot = with_embed[0].embedding or []
                scored_v = [(c.chunk_id, _cosine(c.embedding or [], pivot)) for c in with_embed]
                scored_v.sort(key=lambda x: x[1], reverse=True)
                vec_ranking = [cid for cid, _ in scored_v]

        rankings = [r for r in (bm25_ranking, vec_ranking) if r]
        if not rankings:
            return candidates[:k]
        fused = _rrf(rankings)
        return [self._chunks[cid] for cid in fused[:k]]

    def update_attachments(
        self,
        chunk_id: ChunkId,
        shelf_ids: list[ShelfId],
        theme_ids: list[ThemeId],
    ) -> None:
        c = self._chunks.get(chunk_id)
        if c is None:
            return
        self._chunks[chunk_id] = c.model_copy(
            update={"shelf_ids": list(shelf_ids), "theme_ids": list(theme_ids)}
        )

    def update_annotations(
        self,
        chunk_id: ChunkId,
        mentions: list[Mention],
        entity_links: list[EntityLink],
        foodon_ids: list[str],
        enrichment_version: str,
    ) -> None:
        c = self._chunks.get(chunk_id)
        if c is None:
            return
        self._chunks[chunk_id] = c.model_copy(
            update={
                "mentions": list(mentions),
                "entity_links": list(entity_links),
                "foodon_ids": list(foodon_ids),
                "enrichment_version": enrichment_version,
            }
        )

    def update_embedding(
        self,
        chunk_id: ChunkId,
        embedding: list[float],
        embedding_model: str,
    ) -> None:
        c = self._chunks.get(chunk_id)
        if c is None:
            return
        self._chunks[chunk_id] = c.model_copy(
            update={"embedding": list(embedding), "embedding_model": embedding_model}
        )

    def scan(self) -> list[Chunk]:
        return list(self._chunks.values())

    def iter_chunks(self, batch_size: int = 1000) -> Iterable[list[Chunk]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batch: list[Chunk] = []
        for chunk in self._chunks.values():
            batch.append(chunk)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


class InMemoryGraphStore:
    def __init__(self) -> None:
        self._shelves: dict[ShelfId, Shelf] = {}
        self._themes: dict[ThemeId, Theme] = {}
        self._cards: dict[tuple[str, str], Card] = {}
        self._shelf_chunks: dict[ShelfId, set[ChunkId]] = defaultdict(set)
        self._theme_chunks: dict[ThemeId, set[ChunkId]] = defaultdict(set)
        # First-class entities (mirrors the Neo4j Entity graph). Each entity
        # gets a Pydantic record plus a chunk_id → (confidence, method) map
        # so re-attaching the same chunk updates rather than duplicates.
        self._entities: dict[OntologyId, Entity] = {}
        self._entity_chunks: dict[OntologyId, dict[ChunkId, tuple[float, str]]] = (
            defaultdict(dict)
        )

    def init(self) -> None:
        """No-op — there is nothing to provision for an in-memory store."""
        return

    def upsert_shelves(self, shelves: list[Shelf]) -> None:
        for s in shelves:
            self._shelves[s.shelf_id] = s

    def upsert_themes(self, themes: list[Theme]) -> None:
        for t in themes:
            self._themes[t.theme_id] = t

    def upsert_cards(self, cards: list[Card]) -> None:
        for c in cards:
            self._cards[(c.target_id, c.target_type)] = c

    def attach_chunks_to_shelf(self, shelf_id: ShelfId, chunk_ids: list[ChunkId]) -> None:
        self._shelf_chunks[shelf_id].update(chunk_ids)

    def attach_chunks_to_theme(self, theme_id: ThemeId, chunk_ids: list[ChunkId]) -> None:
        self._theme_chunks[theme_id].update(chunk_ids)

    def get_shelf(self, shelf_id: ShelfId) -> Shelf | None:
        return self._shelves.get(shelf_id)

    def get_themes_for_shelf(self, shelf_id: ShelfId) -> list[Theme]:
        return [t for t in self._themes.values() if shelf_id in t.shelf_ids]

    def get_chunks_for_theme(self, theme_id: ThemeId) -> list[ChunkId]:
        return list(self._theme_chunks.get(theme_id, set()))

    def get_neighbors(self, shelf_id: ShelfId, hops: int = 1) -> list[ShelfId]:
        if shelf_id not in self._shelves:
            return []
        frontier: set[ShelfId] = {shelf_id}
        visited: set[ShelfId] = {shelf_id}
        for _ in range(hops):
            new_frontier: set[ShelfId] = set()
            for sid in frontier:
                shelf = self._shelves.get(sid)
                if shelf and shelf.parent_shelf_id and shelf.parent_shelf_id not in visited:
                    new_frontier.add(shelf.parent_shelf_id)
                for other in self._shelves.values():
                    if other.parent_shelf_id == sid and other.shelf_id not in visited:
                        new_frontier.add(other.shelf_id)
            visited.update(new_frontier)
            frontier = new_frontier
        visited.discard(shelf_id)
        return list(visited)

    def get_card(
        self, target_id: str, target_type: Literal["shelf", "theme"]
    ) -> Card | None:
        return self._cards.get((target_id, target_type))

    def list_shelves(self) -> list[Shelf]:
        return list(self._shelves.values())

    def list_themes(self) -> list[Theme]:
        return list(self._themes.values())

    # ------------------------------------------------------------- entities

    def upsert_entities(self, entities: list[Entity]) -> None:
        for e in entities:
            self._entities[e.ontology_id] = e

    def attach_chunks_to_entity(
        self,
        ontology_id: OntologyId,
        chunk_links: list[tuple[ChunkId, float, str]],
    ) -> None:
        bucket = self._entity_chunks[ontology_id]
        for chunk_id, confidence, method in chunk_links:
            bucket[chunk_id] = (float(confidence), str(method))


class InMemoryEntityStore:
    """Dict-backed `EntityStore`. Search is a token-overlap toy, same flavor
    as `InMemoryChunkStore`. Used by tests and the `in_memory()` facade.
    """

    def __init__(self) -> None:
        self._entities: dict[OntologyId, Entity] = {}

    def init(self) -> None:
        """No-op — there is nothing to provision for an in-memory store."""
        return

    def upsert(self, entities: Iterable[Entity]) -> None:
        for e in entities:
            self._entities[e.ontology_id] = e

    def get(self, ontology_id: OntologyId) -> Entity | None:
        return self._entities.get(ontology_id)

    def get_many(self, ontology_ids: list[OntologyId]) -> list[Entity]:
        return [self._entities[oid] for oid in ontology_ids if oid in self._entities]

    def list_by_prefix(self, prefix: str, *, k: int = 100) -> list[Entity]:
        out = [e for e in self._entities.values() if e.prefix == prefix]
        out.sort(key=lambda e: e.chunk_count, reverse=True)
        return out[:k]

    def search(
        self, query: str, *, prefix: str | None = None, k: int = 10
    ) -> list[Entity]:
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []
        scored: list[tuple[Entity, int]] = []
        for e in self._entities.values():
            if prefix is not None and e.prefix != prefix:
                continue
            hay = set(_tokenize(e.label))
            for syn in e.synonyms:
                hay.update(_tokenize(syn))
            score = len(q_tokens & hay)
            if score > 0:
                scored.append((e, score))
        scored.sort(key=lambda x: (x[1], x[0].chunk_count), reverse=True)
        return [e for e, _ in scored[:k]]

    def scan(self) -> list[Entity]:
        return list(self._entities.values())
