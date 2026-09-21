"""`ElasticRelationStore` — `RelationStore` backed by Elasticsearch 8.x.

Sibling to `ElasticEntityStore`, in its own file for the same reason: the
mapping and the query semantics are different enough that sharing a module
would let the two adapters cross-contaminate.

Index layout (`dynamic: strict`, so an unexpected field raises rather than
auto-inferring to `text`)::

    relation_id                              keyword  (also the ES _id)
    subject_id / object_id / predicate       keyword  (+ predicate.text for BM25)
    subject_surfaces / object_surfaces       text
    predicate_surfaces                       text
    chunk_ids                                keyword[]   <- the `terms` filter
    chunk_count / mention_count              integer
    subject_linked / object_linked           boolean
    extractor_version / dedupe_version       keyword
    last_seen                                date

No `dense_vector` here. Relation embeddings for retrieval's triplet branch are
a retrieval-side concern, and ES 9.4 strips `dense_vector` from `_source`
anyway (see AGENTS.md) — adding one now would buy a read-back problem for no
current consumer.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Literal

from foodscholar.io.chunk import ChunkId
from foodscholar.io.relation import Relation
from foodscholar.logging import get_logger

_log = get_logger("foodscholar.storage.elastic_relations")

_DEFAULT_BULK_SIZE = 500
_SCAN_PAGE = 500
# ES caps a `terms` clause at 65_536 by default; stay well under and page.
_TERMS_PAGE = 1024


class ElasticRelationStore:
    """ES-backed implementation of the `RelationStore` protocol."""

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
            raise ValueError("ElasticRelationStore needs both `url` and `index`")
        if bulk_size <= 0:
            raise ValueError(f"bulk_size must be positive, got {bulk_size}")
        self._bulk_size = bulk_size
        try:
            from elasticsearch import Elasticsearch
        except ImportError as e:
            raise ImportError(
                "the 'elasticsearch>=8' package is required for "
                "ElasticRelationStore. Install with: pip install "
                "'foodscholar[elastic]'"
            ) from e
        self._url = url
        self.index = index

        client_kwargs: dict[str, Any] = {"request_timeout": 60}
        if username and password is not None:
            client_kwargs["basic_auth"] = (username, password)
        else:
            import os

            effective_key = api_key or os.environ.get("ELASTICSEARCH_API_KEY")
            if effective_key:
                client_kwargs["api_key"] = effective_key
        self._es = Elasticsearch(url, **client_kwargs)
        self._ensured_init = False

    # ------------------------------------------------------------------ admin

    def init(self) -> None:
        """Idempotent; tolerates a concurrent creator racing exists()→create()."""
        from elasticsearch import BadRequestError

        if self._es.indices.exists(index=self.index):
            self._ensured_init = True
            return
        try:
            self._create_index()
        except BadRequestError as e:
            if getattr(e, "error", "") != "resource_already_exists_exception":
                raise
        self._ensured_init = True
        _log.info("elastic_relations.index_created", index=self.index)

    def _create_index(self) -> None:
        self._es.indices.create(
            index=self.index,
            body={
                "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                "mappings": {
                    "dynamic": "strict",
                    "properties": {
                        "relation_id": {"type": "keyword"},
                        "subject_id": {"type": "keyword"},
                        "object_id": {"type": "keyword"},
                        "predicate": {
                            "type": "keyword",
                            "fields": {"text": {"type": "text"}},
                        },
                        "subject_surfaces": {"type": "text"},
                        "object_surfaces": {"type": "text"},
                        "predicate_surfaces": {"type": "text"},
                        "chunk_ids": {"type": "keyword"},
                        "chunk_count": {"type": "integer"},
                        "mention_count": {"type": "integer"},
                        "subject_linked": {"type": "boolean"},
                        "object_linked": {"type": "boolean"},
                        "extractor_version": {"type": "keyword"},
                        "dedupe_version": {"type": "keyword"},
                        "last_seen": {"type": "date"},
                    },
                },
            },
        )

    def _ensure_init(self) -> None:
        if not self._ensured_init:
            self.init()

    # ------------------------------------------------------------------ writes

    def upsert(self, relations: Iterable[Relation]) -> None:
        from elasticsearch.helpers import bulk

        self._ensure_init()
        actions = [
            {
                "_op_type": "index",
                "_index": self.index,
                "_id": r.relation_id,
                "_source": _to_source(r),
            }
            for r in relations
        ]
        if not actions:
            return
        bulk(self._es, actions, chunk_size=self._bulk_size, refresh=True)

    def clear(self) -> None:
        """Delete every relation, keeping the index and its mapping."""
        self._ensure_init()
        self._es.delete_by_query(
            index=self.index, body={"query": {"match_all": {}}}, refresh=True
        )

    # ------------------------------------------------------------------ reads

    def get(self, relation_id: str) -> Relation | None:
        from elasticsearch import NotFoundError

        try:
            resp = self._es.get(index=self.index, id=relation_id)
        except NotFoundError:
            return None
        return _from_source(resp["_source"])

    def get_many(self, relation_ids: list[str]) -> list[Relation]:
        if not relation_ids:
            return []
        out: list[Relation] = []
        for start in range(0, len(relation_ids), _TERMS_PAGE):
            window = relation_ids[start : start + _TERMS_PAGE]
            resp = self._es.search(
                index=self.index,
                body={"size": len(window), "query": {"ids": {"values": window}}},
            )
            out.extend(_from_source(h["_source"]) for h in resp["hits"]["hits"])
        return out

    def for_entity(
        self,
        ontology_id: str,
        *,
        direction: Literal["out", "in", "both"] = "both",
        k: int = 100,
    ) -> list[Relation]:
        if direction == "out":
            query: dict[str, Any] = {"term": {"subject_id": ontology_id}}
        elif direction == "in":
            query = {"term": {"object_id": ontology_id}}
        else:
            query = {
                "bool": {
                    "should": [
                        {"term": {"subject_id": ontology_id}},
                        {"term": {"object_id": ontology_id}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        return self._search(query, size=k)

    def for_chunks(self, chunk_ids: list[ChunkId]) -> list[Relation]:
        """One `terms` filter per page — retrieval's hot path.

        Deliberately not a loop over chunk ids: the triplet-similarity branch
        asks for the relations of a whole candidate set at once.
        """
        if not chunk_ids:
            return []
        seen: dict[str, Relation] = {}
        for start in range(0, len(chunk_ids), _TERMS_PAGE):
            window = chunk_ids[start : start + _TERMS_PAGE]
            for relation in self._search(
                {"terms": {"chunk_ids": window}}, size=_SCAN_PAGE, scroll_all=True
            ):
                seen[relation.relation_id] = relation
        out = list(seen.values())
        out.sort(key=lambda r: (r.chunk_count, r.mention_count), reverse=True)
        return out

    def by_predicate(self, predicate: str, *, k: int = 100) -> list[Relation]:
        return self._search({"term": {"predicate": predicate}}, size=k)

    def scan(self) -> list[Relation]:
        return [r for batch in self.iter_relations(_SCAN_PAGE) for r in batch]

    def iter_relations(self, batch_size: int = 1000) -> Iterator[list[Relation]]:
        from elasticsearch.helpers import scan as es_scan

        self._ensure_init()
        batch: list[Relation] = []
        for hit in es_scan(
            self._es,
            index=self.index,
            query={"query": {"match_all": {}}},
            preserve_order=False,
        ):
            batch.append(_from_source(hit["_source"]))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # ------------------------------------------------------------------ helpers

    def _search(
        self, query: dict[str, Any], *, size: int, scroll_all: bool = False
    ) -> list[Relation]:
        self._ensure_init()
        if scroll_all:
            from elasticsearch.helpers import scan as es_scan

            return [
                _from_source(hit["_source"])
                for hit in es_scan(
                    self._es,
                    index=self.index,
                    query={"query": query},
                    preserve_order=False,
                )
            ]
        resp = self._es.search(
            index=self.index,
            body={
                "size": size,
                "query": query,
                "sort": [
                    {"chunk_count": "desc"},
                    {"mention_count": "desc"},
                    {"relation_id": "asc"},
                ],
            },
        )
        return [_from_source(h["_source"]) for h in resp["hits"]["hits"]]


def _to_source(r: Relation) -> dict[str, Any]:
    return {
        "relation_id": r.relation_id,
        "subject_id": r.subject_id,
        "object_id": r.object_id,
        "predicate": r.predicate,
        "subject_surfaces": list(r.subject_surfaces),
        "object_surfaces": list(r.object_surfaces),
        "predicate_surfaces": list(r.predicate_surfaces),
        "chunk_ids": list(r.chunk_ids),
        "chunk_count": r.chunk_count,
        "mention_count": r.mention_count,
        "subject_linked": r.subject_linked,
        "object_linked": r.object_linked,
        "extractor_version": r.extractor_version,
        "dedupe_version": r.dedupe_version,
        "last_seen": r.last_seen.isoformat(),
    }


def _from_source(src: dict[str, Any]) -> Relation:
    return Relation(
        relation_id=src["relation_id"],
        subject_id=src["subject_id"],
        object_id=src["object_id"],
        predicate=src["predicate"],
        subject_surfaces=tuple(src.get("subject_surfaces") or ()),
        object_surfaces=tuple(src.get("object_surfaces") or ()),
        predicate_surfaces=tuple(src.get("predicate_surfaces") or ()),
        chunk_ids=tuple(src.get("chunk_ids") or ()),
        chunk_count=int(src.get("chunk_count") or 0),
        mention_count=int(src.get("mention_count") or 0),
        subject_linked=bool(src.get("subject_linked", True)),
        object_linked=bool(src.get("object_linked", True)),
        extractor_version=src.get("extractor_version") or "",
        dedupe_version=src.get("dedupe_version") or "",
        last_seen=src["last_seen"],
    )
