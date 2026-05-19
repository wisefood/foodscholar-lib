"""Tests for DenseIndex — the linker's dense-tier kNN backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from foodscholar.annotate.dense_index import DenseIndex
from foodscholar.annotate.embedder import HashEmbedder
from foodscholar.ontology import FoodOnAPI, load_ontology

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def api() -> FoodOnAPI:
    return FoodOnAPI(load_ontology(FIXTURES / "mini_foodon.obo"), prefix_filter=None)


def test_build_excludes_obsolete(api: FoodOnAPI) -> None:
    idx = DenseIndex.build(api, HashEmbedder(dim=16))
    # mini fixture: 11 terms, 1 obsolete
    assert idx.size == 10


def test_query_returns_k_results(api: FoodOnAPI) -> None:
    embedder = HashEmbedder(dim=16)
    idx = DenseIndex.build(api, embedder)
    [q] = embedder.embed(["olive oil"])
    hits = idx.query(q, k=3)
    assert len(hits) == 3
    # scores descending
    assert hits[0][1] >= hits[1][1] >= hits[2][1]


def test_query_top_hit_is_self_for_exact_text(api: FoodOnAPI) -> None:
    # HashEmbedder is deterministic — embedding the exact term text should
    # produce the highest cosine against that term's own row.
    embedder = HashEmbedder(dim=32)
    idx = DenseIndex.build(api, embedder)
    olive_oil_term = next(t for t in api if t.label == "olive oil")
    text = " ".join([olive_oil_term.label, *olive_oil_term.synonyms])
    [q] = embedder.embed([text])
    hits = idx.query(q, k=1)
    assert hits[0][0] == olive_oil_term.id
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_query_zero_vector_returns_empty(api: FoodOnAPI) -> None:
    idx = DenseIndex.build(api, HashEmbedder(dim=8))
    assert idx.query([0.0] * 8, k=3) == []


def test_cache_round_trip(api: FoodOnAPI, tmp_path: Path) -> None:
    cache = tmp_path / "terms.npz"
    embedder = HashEmbedder(dim=16)

    first = DenseIndex.build(api, embedder, cache_path=cache)
    assert cache.exists()

    # Second build should load from cache and behave identically.
    second = DenseIndex.build(api, embedder, cache_path=cache)
    assert second.size == first.size

    [q] = embedder.embed(["apple"])
    assert first.query(q, k=3) == second.query(q, k=3)


def test_cache_invalidates_on_embedder_change(api: FoodOnAPI, tmp_path: Path) -> None:
    cache = tmp_path / "terms.npz"
    DenseIndex.build(api, HashEmbedder(dim=16), cache_path=cache)

    # Different embedder dim → different vectors → cache fingerprint mismatch.
    rebuilt = DenseIndex.build(api, HashEmbedder(dim=32), cache_path=cache)
    [q] = HashEmbedder(dim=32).embed(["apple"])
    # If the stale cache had been used, dim-16 vectors would break this query.
    hits = rebuilt.query(q, k=1)
    assert len(hits) == 1


def test_empty_ontology_index(tmp_path: Path) -> None:
    empty = FoodOnAPI([], prefix_filter=None)
    idx = DenseIndex.build(empty, HashEmbedder(dim=8))
    assert idx.size == 0
    assert idx.query([0.1] * 8, k=3) == []
