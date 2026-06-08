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

    def bulk_update_attachments(
        self,
        items: list[tuple[ChunkId, list[ShelfId], list[ThemeId]]],
        *,
        wait_for_refresh: bool = False,
    ) -> None:
        # `wait_for_refresh` is a no-op for in-memory — writes are immediately
        # visible. Kept in the signature so callers don't need a backend check.
        for chunk_id, shelf_ids, theme_ids in items:
            self.update_attachments(chunk_id, shelf_ids, theme_ids)

    def clear_attachments(self) -> None:
        for cid, c in list(self._chunks.items()):
            if c.shelf_ids or c.theme_ids:
                self._chunks[cid] = c.model_copy(
                    update={"shelf_ids": [], "theme_ids": []}
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

    def update_embeddings_bulk(
        self,
        items: list[tuple[ChunkId, list[float], str]],
    ) -> None:
        # Inlined rather than looping over `update_embedding` so tests can
        # assert the bulk path is taken without spying on the single-call
        # method (and so the in-memory contract stays a faithful mirror of
        # the Elastic adapter: one logical write per call).
        for chunk_id, embedding, embedding_model in items:
            c = self._chunks.get(chunk_id)
            if c is None:
                continue
            self._chunks[chunk_id] = c.model_copy(
                update={
                    "embedding": list(embedding),
                    "embedding_model": embedding_model,
                }
            )

    def bulk_set_theme_ids(
        self,
        items: list[tuple[ChunkId, list[ThemeId]]],
    ) -> None:
        # Touches `theme_ids` only — `shelf_ids` are preserved verbatim. This
        # is the safe path for Layer B persistence: a concurrent `fs.attach()`
        # writing `shelf_ids` won't collide.
        for chunk_id, theme_ids in items:
            c = self._chunks.get(chunk_id)
            if c is None:
                continue
            self._chunks[chunk_id] = c.model_copy(
                update={"theme_ids": list(theme_ids)}
            )

    def knn_search_chunks(
        self,
        query_vector: list[float],
        *,
        k: int,
        exclude_ids: list[ChunkId] | None = None,
        candidate_ids: list[ChunkId] | None = None,
    ) -> list[tuple[ChunkId, float]]:
        """Return the top-k cosine-nearest chunks to `query_vector`.

        Uses NumPy for vectorised dot-products. Chunks missing an embedding
        are silently skipped. `exclude_ids` are dropped before scoring;
        `candidate_ids` restricts the search universe when provided.
        """
        import numpy as np

        exclude = set(exclude_ids or [])
        if candidate_ids is not None:
            pool_ids = [cid for cid in candidate_ids if cid in self._chunks]
        else:
            pool_ids = list(self._chunks.keys())
        # Filter to chunks with embeddings, minus exclusions.
        pool = [
            (cid, self._chunks[cid].embedding)
            for cid in pool_ids
            if cid not in exclude and self._chunks[cid].embedding is not None
        ]
        if not pool:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return []
        q = q / qn
        ids = [cid for cid, _ in pool]
        corpus_matrix = np.stack([np.asarray(v, dtype=np.float32) for _, v in pool])
        norms = np.linalg.norm(corpus_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        corpus_matrix = corpus_matrix / norms
        sims = (corpus_matrix @ q).tolist()
        ranked = sorted(zip(ids, sims, strict=False), key=lambda x: x[1], reverse=True)
        return ranked[:k]

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
        # shelf_id -> {chunk_id -> lifted_from foodon_ids}. The inner dict
        # carries provenance written by `fs.attach()` per edge; an empty list
        # means "direct" (chunk linked to the shelf's own foodon_id).
        self._shelf_chunks: dict[ShelfId, dict[ChunkId, list[str]]] = defaultdict(dict)
        self._theme_chunks: dict[ThemeId, set[ChunkId]] = defaultdict(set)
        # Layer B per-edge metadata: (chunk_id, theme_id) -> (primary, weight).
        # Lives in a side dict so the legacy `attach_chunks_to_theme` (no
        # metadata) keeps working for tests/notebooks that pre-date the bulk
        # method. Queries always read this dict first; missing edges default
        # to (False, 0.0).
        self._theme_edge_meta: dict[tuple[ChunkId, ThemeId], tuple[bool, float]] = {}
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

    def clear_layer_a(self) -> None:
        """Drop every shelf + the shelf-side of layer-A attachments.

        Mirrors `MATCH (s:Shelf) DETACH DELETE s` on Neo4j: shelves go, the
        shelf→chunk attachment map goes, and any shelf-target cards go too.
        Themes and chunks themselves stay.
        """
        self._shelves.clear()
        self._shelf_chunks.clear()
        self._cards = {
            key: card
            for key, card in self._cards.items()
            if key[1] != "shelf"
        }

    def clear_attachments(self) -> None:
        """Drop every shelf->chunk attachment without touching the shelves."""
        self._shelf_chunks.clear()

    def upsert_themes(self, themes: list[Theme]) -> None:
        for t in themes:
            self._themes[t.theme_id] = t

    def upsert_cards(self, cards: list[Card]) -> None:
        for c in cards:
            self._cards[(c.target_id, c.target_type)] = c

    def attach_chunks_to_shelf(
        self,
        shelf_id: ShelfId,
        attachments: list[tuple[ChunkId, list[str]]],
    ) -> None:
        bucket = self._shelf_chunks[shelf_id]
        for chunk_id, lifted_from in attachments:
            bucket[chunk_id] = list(lifted_from)

    def attach_chunks_to_theme(self, theme_id: ThemeId, chunk_ids: list[ChunkId]) -> None:
        self._theme_chunks[theme_id].update(chunk_ids)

    def attach_chunks_to_themes_bulk(
        self,
        items: list[tuple[ChunkId, ThemeId, bool, float]],
    ) -> None:
        # Mirrors the Neo4j adapter's UNWIND-driven bulk write: one logical
        # call covers many (chunk, theme, primary, weight) tuples. The
        # in-memory implementation stores the metadata in a side dict.
        for chunk_id, theme_id, primary, weight in items:
            self._theme_chunks[theme_id].add(chunk_id)
            self._theme_edge_meta[(chunk_id, theme_id)] = (primary, float(weight))

    def clear_themes(self) -> None:
        """Drop every theme node + its chunk attachments + per-edge metadata.

        Shelves and the chunk store survive. Chunk-side `theme_ids` denorm
        is the caller's responsibility — `build_layer_b()` always pairs this
        with `chunk_store.bulk_set_theme_ids([(cid, []) ...])` for the same
        set of chunks.
        """
        self._themes.clear()
        self._theme_chunks.clear()
        self._theme_edge_meta.clear()
        # Theme-target cards become orphans; drop them too.
        self._cards = {
            key: card
            for key, card in self._cards.items()
            if key[1] != "theme"
        }

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

    def list_chunk_shelf_attachments(self) -> dict[ChunkId, set[ShelfId]]:
        out: dict[ChunkId, set[ShelfId]] = defaultdict(set)
        for shelf_id, bucket in self._shelf_chunks.items():
            for chunk_id in bucket:
                out[chunk_id].add(shelf_id)
        return dict(out)

    def list_chunk_foodon_mentions(self) -> dict[ChunkId, set[str]]:
        out: dict[ChunkId, set[str]] = defaultdict(set)
        for ontology_id, links in self._entity_chunks.items():
            if not ontology_id.startswith("FOODON:"):
                continue
            for chunk_id in links:
                out[chunk_id].add(ontology_id)
        return dict(out)

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


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class InMemoryCardStore:
    """Dict-backed `CardStore`. `knn_search_cards` is a brute-force cosine over
    stored embeddings — test-grade, mirrors `InMemoryChunkStore`'s flavor. Cards
    without an embedding are skipped by knn.
    """

    def __init__(self) -> None:
        self._cards: dict[str, Card] = {}

    def init(self) -> None:
        """No-op — nothing to provision for an in-memory store."""
        return

    def upsert(self, cards: list[Card]) -> None:
        for c in cards:
            self._cards[c.card_id] = c

    def get_many(self, card_ids: list[str]) -> list[Card]:
        return [self._cards[cid] for cid in card_ids if cid in self._cards]

    def knn_search_cards(
        self,
        query_vector: list[float],
        *,
        k: int,
        exclude_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        excluded = set(exclude_ids or ())
        scored: list[tuple[str, float]] = []
        for cid, card in self._cards.items():
            if cid in excluded or not card.embedding:
                continue
            scored.append((cid, _cosine(query_vector, card.embedding)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
