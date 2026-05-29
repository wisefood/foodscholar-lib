"""Tests for the `bulk_size` tuning knob on ElasticChunkStore / ElasticEntityStore.

These don't talk to a live ES cluster — they patch `Elasticsearch` so we can
verify that:
  - The constructor accepts and stores the configured `bulk_size`.
  - Invalid values raise `ValueError`.
  - `cfg.storage.chunk_store.bulk_size` flows through `FoodScholar.from_config`
    to both the chunk store and the paired entity store.
"""

from __future__ import annotations

from typing import Any

import pytest

from foodscholar.config import ChunkStoreConfig, FoodScholarConfig


class _FakeES:
    """Just enough to satisfy the adapter ctor — no real HTTP."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def _patch_es(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the `Elasticsearch` import target in both store modules."""
    monkeypatch.setattr(
        "elasticsearch.Elasticsearch", _FakeES, raising=True
    )


def test_chunk_store_constructor_records_bulk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_es(monkeypatch)
    from foodscholar.storage.elastic import ElasticChunkStore

    store = ElasticChunkStore(
        url="http://localhost:9200",
        index="foodscholar_chunks",
        bulk_size=2500,
    )
    assert store._bulk_size == 2500


def test_chunk_store_constructor_defaults_to_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_es(monkeypatch)
    from foodscholar.storage.elastic import ElasticChunkStore

    store = ElasticChunkStore(
        url="http://localhost:9200", index="foodscholar_chunks"
    )
    assert store._bulk_size == 500


@pytest.mark.parametrize("bad", [0, -1, -1000])
def test_chunk_store_rejects_non_positive_bulk_size(
    monkeypatch: pytest.MonkeyPatch, bad: int
) -> None:
    _patch_es(monkeypatch)
    from foodscholar.storage.elastic import ElasticChunkStore

    with pytest.raises(ValueError, match="bulk_size must be positive"):
        ElasticChunkStore(
            url="http://localhost:9200",
            index="foodscholar_chunks",
            bulk_size=bad,
        )


def test_entity_store_honors_bulk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_es(monkeypatch)
    from foodscholar.storage.elastic_entities import ElasticEntityStore

    store = ElasticEntityStore(
        url="http://localhost:9200",
        index="foodscholar_chunks_entities",
        bulk_size=4000,
    )
    assert store._bulk_size == 4000


def test_config_default_bulk_size_is_500() -> None:
    cfg = ChunkStoreConfig(backend="elastic", url="http://x", index="i")
    assert cfg.bulk_size == 500


def test_config_accepts_custom_bulk_size() -> None:
    cfg = ChunkStoreConfig(
        backend="elastic", url="http://x", index="i", bulk_size=2500
    )
    assert cfg.bulk_size == 2500


def test_from_config_threads_bulk_size_to_both_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single cfg.storage.chunk_store.bulk_size drives the chunk store AND
    the paired entity store (which shares the chunk-store config)."""
    _patch_es(monkeypatch)
    from foodscholar import FoodScholar

    cfg = FoodScholarConfig.model_validate(
        {
            "corpus": {"chunks_path": "data/chunks.parquet"},
            "storage": {
                "chunk_store": {
                    "backend": "elastic",
                    "url": "http://localhost:9200",
                    "index": "foodscholar_chunks",
                    "bulk_size": 1750,
                },
                "graph_store": {"backend": "memory"},
            },
        }
    )
    fs = FoodScholar.from_config(cfg)
    assert fs.chunk_store._bulk_size == 1750  # type: ignore[attr-defined]
    assert fs.entity_store._bulk_size == 1750  # type: ignore[attr-defined]
