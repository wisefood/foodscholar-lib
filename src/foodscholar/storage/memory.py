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

from foodscholar.io.chunk import Chunk, ChunkId
from foodscholar.io.graph import Card, Shelf, ShelfId, Theme, ThemeId

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

    def scan(self) -> list[Chunk]:
        return list(self._chunks.values())


class InMemoryGraphStore:
    def __init__(self) -> None:
        self._shelves: dict[ShelfId, Shelf] = {}
        self._themes: dict[ThemeId, Theme] = {}
        self._cards: dict[tuple[str, str], Card] = {}
        self._shelf_chunks: dict[ShelfId, set[ChunkId]] = defaultdict(set)
        self._theme_chunks: dict[ThemeId, set[ChunkId]] = defaultdict(set)

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
