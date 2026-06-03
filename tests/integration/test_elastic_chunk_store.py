"""Integration tests for ElasticChunkStore.knn_search_chunks.

Requires a live ES cluster at localhost:9200.  The test is gated via
``pytest.importorskip("elasticsearch")`` — if the package is missing the whole
module is skipped.  A fixture tries to connect and also skips if the cluster
is unreachable or unable to allocate a new shard (e.g. disk full).

Vectors are built with stdlib ``random`` (seeded) + a manual L2 normaliser so
there is no numpy dependency — the heavy-ML path lives in
``test_layer_b_pipeline.py``.
"""

from __future__ import annotations

import math
import random
import uuid

import pytest

pytest.importorskip("elasticsearch")

from elasticsearch import Elasticsearch

from foodscholar.io.chunk import Chunk
from foodscholar.storage.elastic import ElasticChunkStore

_ES_URL = "http://localhost:9200"
_INDEX_PREFIX = "fs_test_knn_"


# ------------------------------------------------------------------ helpers


def _gauss_vector(rng: random.Random, dim: int) -> list[float]:
    """Return a unit-length vector sampled from N(0,1)^dim."""
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _make_chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=f"synthetic test chunk {cid}",
        source_doc_id="test-doc",
        source_type="abstract",
        section_type="other",
        embedding=vec,
        embedding_model="test-bge-768",
    )


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def store():
    """Provision a temp index, yield the store, delete the index on teardown.

    Skips if ES is unreachable or if the cluster cannot allocate a new shard
    (e.g. disk watermark exceeded on a single-node cluster).
    """
    es = Elasticsearch(_ES_URL, request_timeout=120)
    try:
        es.info()
    except Exception:
        pytest.skip("Elasticsearch not reachable at localhost:9200")

    index = f"{_INDEX_PREFIX}{uuid.uuid4().hex[:8]}"
    s = ElasticChunkStore(url=_ES_URL, index=index)

    try:
        s.init()
    except Exception as exc:
        pytest.skip(f"Could not create test index {index!r}: {exc}")

    # Verify the shard is actually assigned (single-node with disk pressure
    # can create an index but leave the primary UNASSIGNED).
    # `wait_for_status="green"` raises ApiError(408) when it times out, so we
    # catch any exception here and treat it as a skip signal.
    try:
        health = es.cluster.health(index=index, wait_for_status="green", timeout="5s")
        shard_ok = health.get("status") == "green"
    except Exception:
        shard_ok = False
    if not shard_ok:
        try:
            es.indices.delete(index=index, ignore_unavailable=True)
        except Exception:
            pass
        pytest.skip(
            f"Test index shard unassigned for {index!r}; "
            "likely disk watermark exceeded — run with more free space."
        )

    yield s

    # Teardown — best-effort; ignore errors if index was already deleted.
    try:
        es.indices.delete(index=index, ignore_unavailable=True)
    except Exception:
        pass


# ------------------------------------------------------------------ tests


def test_knn_search_chunks_returns_top_k_sorted(store: ElasticChunkStore):
    """knn_search_chunks returns k results sorted by cosine descending."""
    rng = random.Random(42)
    dim = 768

    # Build 5 random unit vectors.
    vecs = [_gauss_vector(rng, dim) for _ in range(5)]
    chunk_ids = [f"knn-chunk-{i}" for i in range(5)]
    chunks = [_make_chunk(cid, vec) for cid, vec in zip(chunk_ids, vecs)]

    store.upsert(chunks)

    query_id = chunk_ids[0]
    query_vec = vecs[0]

    results = store.knn_search_chunks(
        query_vector=query_vec,
        k=3,
        exclude_ids=[query_id],
    )

    # Must return exactly k results (4 remaining chunks, k=3).
    assert len(results) == 3, f"expected 3 results, got {len(results)}: {results}"

    # Query chunk must not appear.
    result_ids = [cid for cid, _ in results]
    assert query_id not in result_ids, f"query_id {query_id!r} appeared in results"

    # Scores must be sorted descending.
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), f"scores not sorted desc: {scores}"

    # Scores must be in cosine range [-1, 1].
    for s in scores:
        assert -1.0 <= s <= 1.0, f"score out of [-1,1]: {s}"


def test_knn_search_chunks_exclude_ids_removes_specified(store: ElasticChunkStore):
    """exclude_ids=[id] prevents those ids from appearing in the result set."""
    rng = random.Random(7)
    dim = 768

    vecs = [_gauss_vector(rng, dim) for _ in range(4)]
    chunk_ids = [f"excl-chunk-{i}" for i in range(4)]
    chunks = [_make_chunk(cid, vec) for cid, vec in zip(chunk_ids, vecs)]

    store.upsert(chunks)

    excluded = [chunk_ids[0], chunk_ids[1]]
    results = store.knn_search_chunks(
        query_vector=vecs[0],
        k=10,
        exclude_ids=excluded,
    )

    result_ids = [cid for cid, _ in results]
    for eid in excluded:
        assert eid not in result_ids, f"excluded id {eid!r} still in results: {result_ids}"


def test_init_idempotent_under_create_race(store: ElasticChunkStore):
    """init() must not raise if the index appears between its exists() check and
    create() — the `resource_already_exists` race. The `store` fixture already
    created the index; forcing exists()->False drives init() down the create path
    against an existing index, which must be swallowed rather than raised."""
    store._es.indices.exists = lambda index: False  # force the create branch
    store.init()  # must NOT raise resource_already_exists_exception
    assert store._ensured_init is True
