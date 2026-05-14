"""Tests for embedder adapters."""

from __future__ import annotations

from foodscholar.annotate.embedder import HashEmbedder, SourceTypeRouter
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


def test_source_type_router_picks_scientific_for_abstracts() -> None:
    sci = HashEmbedder(dim=8)
    gen = HashEmbedder(dim=16)
    router = SourceTypeRouter(scientific=sci, general=gen)
    vec, model_id = router.embed_chunk("text", "abstract")
    assert len(vec) == 8
    assert model_id == sci.model_id


def test_source_type_router_picks_general_for_textbook() -> None:
    sci = HashEmbedder(dim=8)
    gen = HashEmbedder(dim=16)
    router = SourceTypeRouter(scientific=sci, general=gen)
    vec, model_id = router.embed_chunk("text", "textbook")
    assert len(vec) == 16
    assert model_id == gen.model_id


def test_source_type_router_picks_general_for_guide() -> None:
    sci = HashEmbedder(dim=8)
    gen = HashEmbedder(dim=16)
    router = SourceTypeRouter(scientific=sci, general=gen)
    _vec, model_id = router.embed_chunk("text", "guide")
    assert model_id == gen.model_id


def test_source_type_router_satisfies_embedder_protocol() -> None:
    router = SourceTypeRouter(HashEmbedder(), HashEmbedder())
    assert isinstance(router, Embedder)
    assert len(router.embed(["a", "b"])) == 2
