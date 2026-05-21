"""`ElasticEntityStore` — `EntityStore` backed by Elasticsearch 8.x.

Sibling to `ElasticChunkStore`. Lives in a separate file because the index
mapping and the search semantics are different enough to keep the two
adapters from cross-contaminating; both use the same `Elasticsearch` client
construction style (api_key / basic_auth via constructor, env fallback).

Index layout:

  - `ontology_id`   keyword (also the ES `_id`)
  - `prefix`        keyword
  - `label`         text (BM25-analyzed) + `.keyword` subfield for exact match
  - `synonyms`      text (multivalue, BM25-analyzed)
  - `ancestor_ids`  keyword[]
  - `facet_hint`    keyword
  - `mention_count`, `chunk_count`  integer
  - `last_seen`     date

`init()` creates the index with this mapping if missing. Bulk upsert.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from foodscholar.io.entity import Entity
from foodscholar.io.ontology import OntologyId
from foodscholar.logging import get_logger

_log = get_logger("foodscholar.storage.elastic_entities")

_BULK_PAGE = 500
_SCAN_PAGE = 500


class ElasticEntityStore:
    """ES-backed implementation of the `EntityStore` protocol."""

    def __init__(
        self,
        url: str,
        index: str,
        *,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        if not url or not index:
            raise ValueError("ElasticEntityStore needs both `url` and `index`")
        try:
            from elasticsearch import Elasticsearch  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "the 'elasticsearch>=8' package is required for ElasticEntityStore. "
                "Install with: pip install 'foodscholar[elastic]'"
            ) from e
        self._url = url
        self.index = index

        client_kwargs: dict[str, Any] = {"request_timeout": 60}
        if username and password is not None:
            client_kwargs["basic_auth"] = (username, password)
        else:
            import os as _os

            effective_key = api_key or _os.environ.get("ELASTICSEARCH_API_KEY")
            if effective_key:
                client_kwargs["api_key"] = effective_key
        self._es = Elasticsearch(url, **client_kwargs)
        # Track whether `init()` (or a self-heal in upsert) has provisioned
        # the index with the explicit mapping in this process. Without this,
        # the very first `upsert` against a fresh cluster would let ES create
        # the index with dynamic text/keyword inference — which breaks sort
        # and term aggregations.
        self._ensured_init = False

    # ------------------------------------------------------------------ admin

    def init(self) -> None:
        if self._es.indices.exists(index=self.index):
            self._ensured_init = True
            return
        self._es.indices.create(
            index=self.index,
            body={
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                },
                # `dynamic: strict` so an unexpected field at upsert time raises
                # rather than silently auto-inferring to text — the bug we just
                # debugged. Every field a Pydantic Entity carries must be listed.
                "mappings": {
                    "dynamic": "strict",
                    "properties": {
                        "ontology_id": {"type": "keyword"},
                        "prefix": {"type": "keyword"},
                        "label": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}},
                        },
                        "synonyms": {"type": "text"},
                        "ancestor_ids": {"type": "keyword"},
                        "facet_hint": {"type": "keyword"},
                        "mention_count": {"type": "integer"},
                        "chunk_count": {"type": "integer"},
                        "chunk_ids": {"type": "keyword"},
                        "last_seen": {"type": "date"},
                    },
                },
            },
        )
        self._ensured_init = True
        _log.info("elastic_entities.index_created", index=self.index)

    # ------------------------------------------------------------------ writes

    def upsert(self, entities: Iterable[Entity]) -> None:
        from elasticsearch.helpers import bulk  # type: ignore[import-not-found]

        # Defensive: if init() wasn't called explicitly, run it once now so
        # ES doesn't auto-create the index with dynamic text inference (the
        # bug that triggered fielddata errors on ontology_id sort/aggs).
        if not self._ensured_init:
            self.init()

        actions: list[dict[str, Any]] = []
        for e in entities:
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.index,
                    "_id": e.ontology_id,
                    "_source": _entity_to_doc(e),
                }
            )
            if len(actions) >= _BULK_PAGE:
                bulk(self._es, actions, refresh=False)
                actions = []
        if actions:
            bulk(self._es, actions, refresh="wait_for")

    # ------------------------------------------------------------------ reads

    def get(self, ontology_id: OntologyId) -> Entity | None:
        try:
            resp = self._es.get(index=self.index, id=ontology_id)
        except Exception:
            return None
        if not resp.get("found"):
            return None
        return _doc_to_entity(resp["_source"])

    def get_many(self, ontology_ids: list[OntologyId]) -> list[Entity]:
        if not ontology_ids:
            return []
        resp = self._es.mget(index=self.index, body={"ids": ontology_ids})
        return [_doc_to_entity(d["_source"]) for d in resp.get("docs", []) if d.get("found")]

    def list_by_prefix(self, prefix: str, *, k: int = 100) -> list[Entity]:
        body: dict[str, Any] = {
            "size": k,
            "query": {"term": {"prefix": prefix}},
            "sort": [{"chunk_count": "desc"}, {"ontology_id": "asc"}],
        }
        resp = self._es.search(index=self.index, body=body)
        return [_doc_to_entity(h["_source"]) for h in resp["hits"]["hits"]]

    def search(
        self, query: str, *, prefix: str | None = None, k: int = 10
    ) -> list[Entity]:
        must: list[dict[str, Any]] = [
            {"multi_match": {"query": query, "fields": ["label^2", "synonyms"]}}
        ]
        body: dict[str, Any] = {
            "size": k,
            "query": {
                "bool": {
                    "must": must,
                    "filter": [{"term": {"prefix": prefix}}] if prefix else [],
                }
            },
        }
        resp = self._es.search(index=self.index, body=body)
        return [_doc_to_entity(h["_source"]) for h in resp["hits"]["hits"]]

    def scan(self) -> list[Entity]:
        out: list[Entity] = []
        after: list[Any] | None = None
        while True:
            # Sort by `_doc` (native index order) — fastest for full scans and
            # avoids depending on any user-managed field being keyword-typed.
            body: dict[str, Any] = {
                "size": _SCAN_PAGE,
                "query": {"match_all": {}},
                "sort": ["_doc"],
            }
            if after is not None:
                body["search_after"] = after
            resp = self._es.search(index=self.index, body=body)
            hits = resp["hits"]["hits"]
            if not hits:
                return out
            for h in hits:
                out.append(_doc_to_entity(h["_source"]))
            after = hits[-1]["sort"]


# ---------------------------------------------------------------------- helpers


def _entity_to_doc(e: Entity) -> dict[str, Any]:
    return e.model_dump(mode="json")


def _doc_to_entity(doc: dict[str, Any]) -> Entity:
    return Entity.model_validate(doc)
