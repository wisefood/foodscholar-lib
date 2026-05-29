"""Tests for embedder adapters."""

from __future__ import annotations

from foodscholar.annotate.embedder import HashEmbedder
from foodscholar.storage.protocols import Embedder


def test_hash_embedder_implements_protocol() -> None:
    assert isinstance(HashEmbedder(), Embedder)


def test_hash_embedder_is_deterministic() -> None:
    a = HashEmbedder().embed(["olive oil"])
    b = HashEmbedder().embed(["olive oil"])
    assert a == b


def test_hash_embedder_dim_is_configurable() -> None:
    e = HashEmbedder(dim=16)
    out = e.embed(["x"])
    assert e.dim == 16
    assert len(out[0]) == 16


def test_hash_embedder_different_text_different_vector() -> None:
    e = HashEmbedder()
    [a, b] = e.embed(["olive oil", "apple"])
    assert a != b
