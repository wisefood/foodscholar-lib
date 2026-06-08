# Layer C Card Embeddings — Design

**Date:** 2026-06-08
**Status:** Implemented (595 unit tests green; ES card store integration-only)
**Builds on:** `2026-06-06-layer-c-theme-summarization-design.md`

## Goal
Every Layer C card carries a vector embedding (from `title + summary`), stored in a new Elastic cards
index, and searchable via `knn_search_cards`. Reuses the existing embedder infra; does **not** implement
`fs.query()` (still deferred) — it provides the searchable card store a future query path would use.

## Decisions (locked)
- **Embed input:** `f"{title}\n\n{summary}"`.
- **Storage:** new Elastic `foodscholar_cards` index (dense_vector 768, hnsw, cosine + the same
  `_source`-strip workaround chunks use). Cards persist to **both** Neo4j (as today) and this index.
- **Compute point:** in the Layer C builder, after Stage 2, before persist (one lazy embedder load).
- **Card store:** dedicated `ElasticCardStore` (sibling to `ElasticChunkStore`) + `InMemoryCardStore`.

## Changes

### 1. Card model (`io/graph.py`)
Add two optional fields mirroring `Chunk`:
```python
embedding: list[float] | None = None
embedding_model: str | None = None
```
Optional/defaulted → existing Neo4j cards and all current tests stay valid.

### 2. Card store protocol (`storage/protocols.py`)
New `CardStore` protocol:
```python
def init(self) -> None: ...
def upsert(self, cards: list[Card]) -> None: ...
def get_many(self, card_ids: list[str]) -> list[Card]: ...
def knn_search_cards(self, query_vector: list[float], *, k: int,
                     exclude_ids: list[str] | None = None) -> list[tuple[str, float]]: ...
```

### 3. ElasticCardStore (`storage/elastic.py`)
Sibling to `ElasticChunkStore`, index `foodscholar_cards`:
- `init()` / `_create_index()` / `recreate()` with `dense_vector(768, hnsw, cosine)`; non-bbq hnsw so
  the raw vector survives in a fetchable field. On read, if `embedding` absent from `_source` but
  `embedding_model` present, fetch via the fields API (mirror `_fetch_embedding`).
- `upsert(cards)` (bulk), `get_many(card_ids)`, `knn_search_cards(query_vector, *, k, exclude_ids)`.
- Card id = `card.card_id` (the ES `_id`).

### 4. InMemoryCardStore (`storage/memory.py`)
Dict keyed by `card_id`; `knn_search_cards` = brute-force cosine over stored embeddings (test-grade).

### 5. Compute in builder (`layer_c/builder.py`)
After collecting cards, before persist (skipped when `dry_run`):
```python
texts = [f"{c.title}\n\n{c.summary}" for c in cards]
vecs = fs.embedder.embed(texts)
cards = [c.model_copy(update={"embedding": v, "embedding_model": fs.embedder.model_id})
         for c, v in zip(cards, vecs)]
```

### 6. Persistence (`layer_c/persist.py`)
`persist_cards(cards, graph_store, card_store)` writes Neo4j (unchanged) AND `card_store.upsert(cards)`.
Builder passes `fs.card_store`.

### 7. Config + facade
- `StorageConfig.card_store: CardStoreConfig` (backend `elastic|memory`, url, index default
  `foodscholar_cards`, api_key/auth like chunk_store).
- `fs.card_store` property + build wiring mirroring `chunk_store`; `fs.init()` also calls
  `card_store.init()`.
- `fs.search_cards(text, k=...)` convenience: embed `text` via `fs.embedder`, call
  `card_store.knn_search_cards`, return cards. (Thin; the real `query()` stays deferred.)

## Testing
- Card model: embedding/embedding_model default None; round-trip with a vector.
- InMemoryCardStore: upsert/get_many/knn ordering (nearest first), exclude_ids.
- Builder: with a mock embedder, cards get embedding + embedding_model; dry_run leaves them None and
  persists nothing.
- persist_cards: writes to both graph store and card store; empty input no-op.
- Facade: `card_store` property resolves for memory backend; `search_cards` returns nearest cards.
- ES `ElasticCardStore`: marked `@pytest.mark.integration` (needs live ES), not in the unit gate.

## Out of scope
- `fs.query()` retrieval/answer synthesis (deferred).
- Re-embedding existing cards as a standalone phase (builder computes at generation time).
- Hybrid BM25+vector over cards (knn only for now).
