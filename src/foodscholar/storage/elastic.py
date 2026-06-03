"""`ElasticChunkStore` — `ChunkStore` backed by Elasticsearch 8.x.

Index layout:

  - `chunk_id`             keyword (also the ES `_id`)
  - `text`                 text (BM25-analyzed)
  - `source_doc_id`,
    `source_type`,
    `section_type`         keyword
  - `year`                 integer
  - `source_metadata`      flattened
  - `embedding`            dense_vector(dims=768, cosine, hnsw)
  - `embedding_model`      keyword
  - `mentions`,
    `entity_links`         nested
  - `foodon_ids`,
    `shelf_ids`,
    `theme_ids`            keyword[]
  - `enrichment_version`,
    `created_at`           keyword / date

`init()` creates the index with this mapping if it's missing. The vector field
is pinned to 768 dims (BGE-base, the sole production embedder) and uses plain
`hnsw` index_options — the ES 9.x default of `bbq_hnsw` would drop the raw
vector from `_source`, which Pydantic round-trips need for `Chunk.embedding`.

This is a hot path of foodscholar production runs, so the implementation
sticks to the boring choices: bulk-helpers for writes, point-in-time +
search_after for scans, no async, no fancy DSL builders.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from foodscholar.io.chunk import Chunk, ChunkId, EntityLink, Mention
from foodscholar.io.graph import ShelfId, ThemeId
from foodscholar.logging import get_logger

_log = get_logger("foodscholar.storage.elastic")

# Default page size for bulk upserts when the constructor is invoked without
# a `bulk_size=` override. Conservative default — ES `_bulk` slows down on
# very large payloads but 500 is well within the safe envelope.
_DEFAULT_BULK_SIZE = 500
_SCAN_PAGE = 500


class ElasticChunkStore:
    """ES-backed implementation of the `ChunkStore` protocol.

    Authentication: pass an `api_key` (config or `$ELASTICSEARCH_API_KEY`),
    or a `(username, password)` HTTP-basic pair. Anonymous access is used when
    none is configured (suitable for an unauthenticated local cluster).
    """

    def __init__(
        self,
        url: str,
        index: str,
        *,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
        bulk_size: int = _DEFAULT_BULK_SIZE,
    ) -> None:
        if not url or not index:
            raise ValueError("ElasticChunkStore needs both `url` and `index`")
        if bulk_size <= 0:
            raise ValueError(f"bulk_size must be positive, got {bulk_size}")
        self._bulk_size = bulk_size
        try:
            from elasticsearch import Elasticsearch  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "the 'elasticsearch>=8' package is required for ElasticChunkStore. "
                "Install with: pip install 'foodscholar[elastic]'"
            ) from e
        self._url = url
        self.index = index

        client_kwargs: dict[str, Any] = {"request_timeout": 60}
        # HTTP-basic wins when present; otherwise prefer an explicit API key,
        # falling back to the environment for both.
        if username and password is not None:
            client_kwargs["basic_auth"] = (username, password)
        else:
            import os as _os

            effective_key = api_key or _os.environ.get("ELASTICSEARCH_API_KEY")
            if effective_key:
                client_kwargs["api_key"] = effective_key
        self._es = Elasticsearch(url, **client_kwargs)
        # Tracks whether init() (or upsert's self-heal) has provisioned the
        # index with the explicit mapping in this process. Without this, the
        # very first upsert against a fresh cluster would let ES auto-create
        # the index with dynamic text inference and break sort / term aggs.
        self._ensured_init = False

    # ------------------------------------------------------------------ admin

    def init(self) -> None:
        """Create the index with the FoodScholar mapping if missing. Idempotent."""
        if self._es.indices.exists(index=self.index):
            self._ensured_init = True
            return
        self._es.indices.create(
            index=self.index,
            body={
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {"analyzer": {"default": {"type": "standard"}}},
                },
                "mappings": {
                    "dynamic": "false",
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "text": {"type": "text"},
                        "source_doc_id": {"type": "keyword"},
                        "source_type": {"type": "keyword"},
                        "section_type": {"type": "keyword"},
                        "year": {"type": "integer"},
                        "source_metadata": {"type": "flattened"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 768,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {
                                "type": "hnsw",
                                "m": 16,
                                "ef_construction": 100,
                            },
                        },
                        "embedding_model": {"type": "keyword"},
                        "mentions": {
                            "type": "nested",
                            "properties": {
                                "text": {"type": "keyword"},
                                "start": {"type": "integer"},
                                "end": {"type": "integer"},
                                "score": {"type": "float"},
                                "ner_model_version": {"type": "keyword"},
                                "entity_type": {"type": "keyword"},
                            },
                        },
                        "entity_links": {
                            "type": "nested",
                            "properties": {
                                "ontology_id": {"type": "keyword"},
                                "confidence": {"type": "float"},
                                "method": {"type": "keyword"},
                                "linker_version": {"type": "keyword"},
                                "mention": {
                                    "properties": {
                                        "text": {"type": "keyword"},
                                        "start": {"type": "integer"},
                                        "end": {"type": "integer"},
                                        "score": {"type": "float"},
                                        "ner_model_version": {"type": "keyword"},
                                        "entity_type": {"type": "keyword"},
                                    }
                                },
                            },
                        },
                        "foodon_ids": {"type": "keyword"},
                        "shelf_ids": {"type": "keyword"},
                        "theme_ids": {"type": "keyword"},
                        "enrichment_version": {"type": "keyword"},
                        "created_at": {"type": "date"},
                    },
                },
            },
        )
        self._ensured_init = True
        _log.info("elastic.index_created", index=self.index)

    def recreate(self) -> None:
        """Delete the index (if present) and recreate it with the CURRENT mapping.

        `init()` is idempotent and leaves an existing index untouched, so a stale
        on-disk mapping (e.g. embedding dims changed, or a field type that no longer
        matches) causes `BulkIndexError` on re-ingest. Call this for a clean slate.
        Destructive: drops all chunks in the index."""
        self._es.indices.delete(index=self.index, ignore_unavailable=True)
        self._ensured_init = False
        self.init()
        _log.info("elastic.index_recreated", index=self.index)

    # ------------------------------------------------------------------ writes

    def upsert(self, chunks: Iterable[Chunk]) -> None:
        from elasticsearch.helpers import bulk  # type: ignore[import-not-found]

        if not self._ensured_init:
            self.init()

        actions: list[dict[str, Any]] = []
        for chunk in chunks:
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.index,
                    "_id": chunk.chunk_id,
                    "_source": _chunk_to_doc(chunk),
                }
            )
            if len(actions) >= self._bulk_size:
                bulk(self._es, actions, refresh=False)
                actions = []
        if actions:
            bulk(self._es, actions, refresh="wait_for")

    def update_attachments(
        self,
        chunk_id: ChunkId,
        shelf_ids: list[ShelfId],
        theme_ids: list[ThemeId],
    ) -> None:
        self._es.update(
            index=self.index,
            id=chunk_id,
            body={"doc": {"shelf_ids": list(shelf_ids), "theme_ids": list(theme_ids)}},
            refresh="wait_for",
        )

    def bulk_update_attachments(
        self,
        items: list[tuple[ChunkId, list[ShelfId], list[ThemeId]]],
        *,
        wait_for_refresh: bool = False,
    ) -> None:
        """Patch shelf_ids + theme_ids on many chunks via one `_bulk` call.

        `wait_for_refresh=True` blocks until the new values are searchable
        — pass on the last flush so subsequent BM25/kNN queries see them.
        Intermediate flushes pass `wait_for_refresh=False` and amortize the
        index refresh.

        Errors handling: when `bulk(..., raise_on_error=False)` returns a
        non-empty errors list we surface a deterministic exception. The
        prior implementation discarded both `success_count` and `errors`,
        which silently lost up to ~57% of writes in production: a
        `clear_attachments` call (which uses `update_by_query` and bumps
        every doc's `_version`) was followed by these bulk updates while
        the cluster was still settling the version state, and many actions
        returned `version_conflict_engine_exception`. Without checking the
        result we never knew.

        `retry_on_conflict=5` tells ES to internally retry version
        conflicts a few times before giving up. Combined with the explicit
        error check this protects against transient concurrent-update
        races without masking real failures.
        """
        if not items:
            return
        from elasticsearch.helpers import bulk  # type: ignore[import-not-found]

        actions = [
            {
                "_op_type": "update",
                "_index": self.index,
                "_id": chunk_id,
                "retry_on_conflict": 5,
                "doc": {
                    "shelf_ids": list(shelf_ids),
                    "theme_ids": list(theme_ids),
                },
            }
            for chunk_id, shelf_ids, theme_ids in items
        ]
        success, errors = bulk(
            self._es,
            actions,
            refresh="wait_for" if wait_for_refresh else False,
            raise_on_error=False,
            raise_on_exception=False,
            stats_only=False,
        )
        if errors:
            # Truncate the surfaced error list to keep the exception
            # readable; full error details are loggable for debugging.
            sample = errors[:5]
            n_failed = len(errors)
            _log.error(
                "elastic.bulk_update_attachments.partial_failure",
                index=self.index,
                attempted=len(actions),
                succeeded=success,
                failed=n_failed,
                first_errors=sample,
            )
            raise RuntimeError(
                f"bulk_update_attachments lost {n_failed}/{len(actions)} writes "
                f"on index {self.index!r} (succeeded={success}). "
                f"First failures: {sample}"
            )

    def clear_attachments(self) -> None:
        """Wipe `shelf_ids` + `theme_ids` on every chunk in the index.

        Implemented as `_update_by_query` — server-side, single round-trip,
        no chunk download. The query is `match_all` because the per-chunk
        cost of "did this chunk ever have an attachment?" via a script
        wouldn't beat just rewriting the field set.
        """
        self._es.update_by_query(
            index=self.index,
            body={
                "script": {
                    "source": "ctx._source.shelf_ids = []; ctx._source.theme_ids = []",
                    "lang": "painless",
                },
                "query": {"match_all": {}},
            },
            refresh=True,
            conflicts="proceed",
            wait_for_completion=True,
        )

    def update_annotations(
        self,
        chunk_id: ChunkId,
        mentions: list[Mention],
        entity_links: list[EntityLink],
        foodon_ids: list[str],
        enrichment_version: str,
    ) -> None:
        self._es.update(
            index=self.index,
            id=chunk_id,
            body={
                "doc": {
                    "mentions": [m.model_dump(mode="json") for m in mentions],
                    "entity_links": [ln.model_dump(mode="json") for ln in entity_links],
                    "foodon_ids": list(foodon_ids),
                    "enrichment_version": enrichment_version,
                }
            },
            refresh="wait_for",
        )

    def update_embedding(
        self,
        chunk_id: ChunkId,
        embedding: list[float],
        embedding_model: str,
    ) -> None:
        self._es.update(
            index=self.index,
            id=chunk_id,
            body={
                "doc": {
                    "embedding": list(embedding),
                    "embedding_model": embedding_model,
                }
            },
            refresh="wait_for",
        )

    def update_embeddings_bulk(
        self,
        items: list[tuple[ChunkId, list[float], str]],
    ) -> None:
        """Bulk partial-doc update — one `_bulk` HTTP request per call
        instead of one `_update` per chunk. Hot path for tunneled embed runs
        where per-doc latency dominates GPU time.

        `refresh=False` (no `wait_for`) — we don't need each embedding
        searchable mid-run; the run as a whole finishes with a final
        no-op refresh, and `fs.embed()` is typically followed by other
        bulk phases that already trigger refreshes.

        Errors handling: `raise_on_error=False` so ES returns per-doc failures
        instead of `BulkIndexError("N documents failed to index.")` with no
        visible reasons. Failures are logged with a sample and re-raised as
        a `RuntimeError` whose message embeds the first few ES error bodies,
        so a notebook traceback actually shows *why* the writes were rejected
        (mapping mismatch, doc-not-found, vector dim drift, etc.).
        """
        if not items:
            return
        from elasticsearch.helpers import bulk  # type: ignore[import-not-found]

        actions = [
            {
                "_op_type": "update",
                "_index": self.index,
                "_id": chunk_id,
                "doc": {
                    "embedding": list(embedding),
                    "embedding_model": embedding_model,
                },
            }
            for chunk_id, embedding, embedding_model in items
        ]
        success, errors = bulk(
            self._es,
            actions,
            refresh=False,
            raise_on_error=False,
            raise_on_exception=False,
            stats_only=False,
        )
        if errors:
            sample = errors[:3]
            n_failed = len(errors)
            _log.error(
                "elastic.update_embeddings_bulk.partial_failure",
                index=self.index,
                attempted=len(actions),
                succeeded=success,
                failed=n_failed,
                first_errors=sample,
            )
            raise RuntimeError(
                f"update_embeddings_bulk lost {n_failed}/{len(actions)} writes "
                f"on index {self.index!r} (succeeded={success}). "
                f"First failures: {sample}"
            )

    def bulk_set_theme_ids(
        self,
        items: list[tuple[ChunkId, list[ThemeId]]],
    ) -> None:
        """Set `theme_ids` only — leave `shelf_ids` untouched.

        One `_bulk` round-trip per call. Used by Layer B persist so a
        concurrent `fs.attach()` writing `shelf_ids` doesn't race against
        our read-then-overwrite. `refresh=False` — Layer B's caller drives
        the final refresh.
        """
        if not items:
            return
        from elasticsearch.helpers import bulk  # type: ignore[import-not-found]

        actions = [
            {
                "_op_type": "update",
                "_index": self.index,
                "_id": chunk_id,
                "doc": {"theme_ids": list(theme_ids)},
            }
            for chunk_id, theme_ids in items
        ]
        bulk(self._es, actions, refresh=False)

    # ------------------------------------------------------------------ reads

    def get(self, chunk_id: ChunkId) -> Chunk | None:
        try:
            resp = self._es.get(index=self.index, id=chunk_id)
        except Exception:
            return None
        if not resp.get("found"):
            return None
        src = resp["_source"]
        # ES 9.x strips `dense_vector` from `_source` regardless of
        # `index_options` — fetch the vector via the `fields` API and merge
        # it back so `Chunk.embedding` round-trips. The `get` API doesn't
        # accept `fields`; route via a single-hit `search` instead.
        if "embedding" not in src and src.get("embedding_model"):
            src = {**src, "embedding": self._fetch_embedding(chunk_id)}
        return _doc_to_chunk(src)

    def get_many(self, chunk_ids: list[ChunkId], *, batch_size: int = 500) -> list[Chunk]:
        if not chunk_ids:
            return []
        # Batch the fetch: a single `_mget` over thousands of ids (each pulling a
        # 768-dim dense_vector via the paired vectors fetch) builds one huge
        # response that can OOM a small ES heap. Chunking it bounds peak memory.
        #
        # Two round-trips per batch (one `_mget` for the source doc, one `_search`
        # for the vectors) — `_mget` doesn't honor `fields`. The vectors fetch is
        # keyed by `_id` so it returns only chunks we asked for.
        out: list[Chunk] = []
        for start in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[start:start + batch_size]
            resp = self._es.mget(index=self.index, body={"ids": batch})
            docs = [d for d in resp.get("docs", []) if d.get("found")]
            if not docs:
                continue
            embeddings = self._fetch_embeddings_bulk(
                [d["_id"] for d in docs if d["_source"].get("embedding_model")]
            )
            for d in docs:
                src = d["_source"]
                if "embedding" not in src and d["_id"] in embeddings:
                    src = {**src, "embedding": embeddings[d["_id"]]}
                out.append(_doc_to_chunk(src))
        return out

    def knn_search_chunks(
        self,
        query_vector: list[float],
        *,
        k: int,
        exclude_ids: list[ChunkId] | None = None,
        candidate_ids: list[ChunkId] | None = None,
    ) -> list[tuple[ChunkId, float]]:
        """kNN search using ES ``knn`` query on the ``embedding`` field.

        ``exclude_ids`` becomes a ``must_not.ids`` filter; ``candidate_ids``
        becomes an ``ids`` filter (the kNN search is restricted to these).
        ES returns ``_score = (cosine + 1) / 2`` for ``similarity='cosine'``
        docs; we map back to plain cosine in [-1, 1] so callers see consistent
        semantics across ChunkStore implementations.
        """
        # num_candidates governs HNSW recall — 10x k is conservative.
        num_candidates = max(50, k * 10)

        filters: list[dict[str, Any]] = []
        if candidate_ids:
            filters.append({"ids": {"values": list(candidate_ids)}})
        must_not: list[dict[str, Any]] = []
        if exclude_ids:
            must_not.append({"ids": {"values": list(exclude_ids)}})

        knn_body: dict[str, Any] = {
            "field": "embedding",
            "query_vector": list(query_vector),
            "k": k,
            "num_candidates": num_candidates,
        }
        if filters or must_not:
            knn_body["filter"] = {
                "bool": {
                    "filter": filters,
                    "must_not": must_not,
                }
            }
        body: dict[str, Any] = {
            "size": k,
            "_source": False,
            "knn": knn_body,
        }
        resp = self._es.search(index=self.index, body=body)
        out: list[tuple[ChunkId, float]] = []
        for h in resp["hits"]["hits"]:
            es_score = h["_score"]
            cosine = (es_score * 2.0) - 1.0
            out.append((h["_id"], cosine))
        return out

    def search(
        self,
        query: str,
        theme_ids: list[ThemeId] | None = None,
        shelf_ids: list[ShelfId] | None = None,
        k: int = 10,
        use_vector: bool = True,
        use_bm25: bool = True,
    ) -> list[Chunk]:
        filters: list[dict[str, Any]] = []
        if shelf_ids:
            filters.append({"terms": {"shelf_ids": list(shelf_ids)}})
        if theme_ids:
            filters.append({"terms": {"theme_ids": list(theme_ids)}})

        bm25_hits: list[tuple[str, dict[str, Any]]] = []
        if use_bm25:
            body: dict[str, Any] = {
                "size": k,
                "query": {
                    "bool": {
                        "must": [{"match": {"text": query}}] if query else [{"match_all": {}}],
                        "filter": filters,
                    }
                },
            }
            resp = self._es.search(index=self.index, body=body)
            bm25_hits = [(h["_id"], h["_source"]) for h in resp["hits"]["hits"]]

        vec_hits: list[tuple[str, dict[str, Any]]] = []
        # ES search by text alone uses BM25 only. The vector path here is meant
        # for callers that pre-embed the query and pass it via a future
        # `query_vector` kwarg. For protocol parity we fall back to a kNN over
        # the highest-scoring BM25 hit's embedding when BM25 returned anything.
        if use_vector and query and bm25_hits:
            seed_vec = bm25_hits[0][1].get("embedding")
            if seed_vec:
                knn_body = {
                    "size": k,
                    "knn": {
                        "field": "embedding",
                        "query_vector": seed_vec,
                        "k": k,
                        "num_candidates": max(50, k * 5),
                        "filter": filters,
                    },
                }
                resp = self._es.search(index=self.index, body=knn_body)
                vec_hits = [(h["_id"], h["_source"]) for h in resp["hits"]["hits"]]

        # Reciprocal-rank fusion across the two ranked lists.
        rankings = [r for r in ([h[0] for h in bm25_hits], [h[0] for h in vec_hits]) if r]
        if not rankings:
            return []
        fused = _rrf(rankings)[:k]
        by_id = {h[0]: h[1] for h in (*bm25_hits, *vec_hits)}
        return [_doc_to_chunk(by_id[cid]) for cid in fused if cid in by_id]

    def scan(self) -> list[Chunk]:
        return list(self._iter_all())

    def iter_chunks(self, batch_size: int = 1000) -> Iterable[list[Chunk]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batch: list[Chunk] = []
        for chunk in self._iter_all(page=batch_size):
            batch.append(chunk)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # ------------------------------------------------------------------ private

    def _iter_all(self, *, page: int = _SCAN_PAGE) -> Iterator[Chunk]:
        """Stream every chunk via `search_after`.

        Sort key is `chunk_id` (keyword) — totally ordered, stable across
        requests, mappable to a unique-per-document value. We previously used
        `_doc` because it's the fastest sort to compute, but `_doc` is NOT
        stable across multiple search calls: ES makes no guarantee about
        monotonicity of `_doc` across segments or after a segment merge.
        That broke `search_after` paging when `attach()` ran right after
        `clear_attachments()` — the `update_by_query` triggered segment churn,
        and the very next paginated scan terminated after 3 pages with the
        cursor pointing at a now-invalid `_doc`, silently dropping ~75% of
        the corpus.

        `chunk_id` is a `keyword` with no field-data issues; it's also the
        index's `_id`, so sorting by it is essentially an index lookup. Tiny
        speed hit vs `_doc`; correctness is non-negotiable.
        """
        after: list[Any] | None = None
        while True:
            body: dict[str, Any] = {
                "size": page,
                "query": {"match_all": {}},
                "sort": [{"chunk_id": "asc"}],
                # `dense_vector` is excluded from `_source` on ES 9.x —
                # request it explicitly via the `fields` API and merge
                # it back into the source dict before validation.
                "fields": ["embedding"],
            }
            if after is not None:
                body["search_after"] = after
            resp = self._es.search(index=self.index, body=body)
            hits = resp["hits"]["hits"]
            if not hits:
                return
            for h in hits:
                src = h["_source"]
                vec = (h.get("fields") or {}).get("embedding")
                if vec is not None and "embedding" not in src:
                    src = {**src, "embedding": vec}
                yield _doc_to_chunk(src)
            after = hits[-1]["sort"]

    def _fetch_embedding(self, chunk_id: ChunkId) -> list[float] | None:
        """Look up one chunk's vector via the `fields` API."""
        resp = self._es.search(
            index=self.index,
            body={
                "size": 1,
                "_source": False,
                "fields": ["embedding"],
                "query": {"term": {"_id": chunk_id}},
            },
        )
        hits = resp["hits"]["hits"]
        if not hits:
            return None
        return (hits[0].get("fields") or {}).get("embedding")

    def _fetch_embeddings_bulk(
        self, chunk_ids: list[ChunkId]
    ) -> dict[ChunkId, list[float]]:
        """Batch-fetch `embedding` for `chunk_ids` via the `fields` API.

        ES caps `terms` clauses at `index.max_terms_count` (default 65 536),
        so we page through `chunk_ids` rather than risk a request-time error
        on jumbo calls. Returns a `chunk_id → vector` dict; missing ids are
        simply absent (caller handles the fallback to `embedding=None`).
        """
        if not chunk_ids:
            return {}
        out: dict[ChunkId, list[float]] = {}
        page = 1024
        for start in range(0, len(chunk_ids), page):
            window = chunk_ids[start : start + page]
            resp = self._es.search(
                index=self.index,
                body={
                    "size": len(window),
                    "_source": False,
                    "fields": ["embedding"],
                    "query": {"ids": {"values": window}},
                },
            )
            for h in resp["hits"]["hits"]:
                vec = (h.get("fields") or {}).get("embedding")
                if vec is not None:
                    out[h["_id"]] = vec
        return out


# ---------------------------------------------------------------------- helpers


def _chunk_to_doc(chunk: Chunk) -> dict[str, Any]:
    """Pydantic → ES doc. Pydantic does the date→ISO + nested serialization."""
    return chunk.model_dump(mode="json")


def _doc_to_chunk(doc: dict[str, Any]) -> Chunk:
    return Chunk.model_validate(doc)


def _rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal-rank fusion; same recipe as `InMemoryChunkStore._rrf`."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
