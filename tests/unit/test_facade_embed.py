"""Tests for `fs.embed()` — fill in chunk-text vectors for chunks already in the store.

The default contract:

  - ingest stores chunks with `embedding = None`
  - `fs.embed()` walks the store, encodes each chunk with the configured
    embedder, writes back via `chunk_store.update_embedding` (only the
    embedding + embedding_model, never the annotations)
  - `only_missing=True` (default) skips chunks whose embedding_model is
    already a real (non-mock) id
  - `only_missing=False` re-encodes every chunk
"""

from __future__ import annotations

from foodscholar import FoodScholar
from foodscholar.facade import _is_real_embedding
from foodscholar.io.chunk import Chunk


def _chunk(chunk_id: str, *, embedding=None, model: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        source_doc_id="d",
        source_type="abstract",
        section_type="abstract",
        embedding=embedding,
        embedding_model=model,
    )


def _fs(**overrides) -> FoodScholar:  # type: ignore[no-untyped-def]
    fs = FoodScholar.in_memory()
    for k, v in overrides.items():
        setattr(fs, k, v)
    return fs


def test_is_real_embedding_helper() -> None:
    assert _is_real_embedding("allenai/specter2_base") is True
    assert _is_real_embedding("BAAI/bge-large-en-v1.5") is True
    assert _is_real_embedding("router(scientific=specter2;general=bge)") is True
    assert _is_real_embedding("mock-embedder-v0") is False
    assert _is_real_embedding("hash-embedder-v0") is False
    assert _is_real_embedding(None) is False
    assert _is_real_embedding("") is False


def test_embed_fills_in_missing_vectors() -> None:
    fs = _fs()
    fs.upsert_chunks([_chunk("c1"), _chunk("c2")])

    meta = fs.embed()
    assert meta.phase == "embed"
    assert meta.record_count == 2

    for cid in ("c1", "c2"):
        c = fs.chunk_store.get(cid)
        assert c is not None
        assert c.embedding is not None
        assert len(c.embedding) > 0
        assert c.embedding_model == "mock-embedder-v0"


def test_embed_only_missing_default_skips_already_embedded() -> None:
    """Chunks whose embedding_model looks 'real' must not be re-encoded."""
    fs = _fs()
    fs.upsert_chunks(
        [
            _chunk("c1", embedding=[0.1, 0.2], model="allenai/specter2_base"),
            _chunk("c2"),
        ]
    )
    meta = fs.embed()
    assert meta.record_count == 1  # only c2 encoded

    c1 = fs.chunk_store.get("c1")
    assert c1 is not None
    assert c1.embedding == [0.1, 0.2]
    assert c1.embedding_model == "allenai/specter2_base"

    c2 = fs.chunk_store.get("c2")
    assert c2 is not None
    assert c2.embedding is not None
    assert c2.embedding_model == "mock-embedder-v0"


def test_embed_force_overwrites_when_only_missing_false() -> None:
    fs = _fs()
    fs.upsert_chunks(
        [_chunk("c1", embedding=[0.9, 0.8], model="allenai/specter2_base")]
    )
    fs.embed(only_missing=False)
    c1 = fs.chunk_store.get("c1")
    assert c1 is not None
    assert c1.embedding_model == "mock-embedder-v0"
    assert c1.embedding != [0.9, 0.8]


def test_embed_overwrites_mock_under_default_only_missing() -> None:
    """A mock vector is NOT considered real — only_missing=True still re-encodes it."""
    fs = _fs()
    fs.upsert_chunks(
        [_chunk("c1", embedding=[0.1] * 8, model="hash-embedder-v0")]
    )
    meta = fs.embed()
    assert meta.record_count == 1
    c1 = fs.chunk_store.get("c1")
    assert c1 is not None
    assert c1.embedding_model == "mock-embedder-v0"


def test_embed_uses_source_type_router_when_present() -> None:
    """A SourceTypeRouter on fs.embedder routes per chunk.source_type."""
    from foodscholar.annotate.embedder import HashEmbedder, SourceTypeRouter

    scientific = HashEmbedder(dim=8)
    scientific.model_id = "fake-specter"  # type: ignore[attr-defined]
    general = HashEmbedder(dim=12)
    general.model_id = "fake-bge"  # type: ignore[attr-defined]
    router = SourceTypeRouter(scientific=scientific, general=general)

    fs = FoodScholar.in_memory()
    fs.embedder = router  # bypass lazy build
    fs.upsert_chunks(
        [
            Chunk(
                chunk_id="abs1",
                text="an abstract chunk",
                source_doc_id="d",
                source_type="abstract",
                section_type="abstract",
            ),
            Chunk(
                chunk_id="gd1",
                text="a guide chunk",
                source_doc_id="d",
                source_type="guide",
                section_type="guideline",
            ),
        ]
    )
    fs.embed()
    abs1 = fs.chunk_store.get("abs1")
    gd1 = fs.chunk_store.get("gd1")
    assert abs1 is not None and gd1 is not None
    assert abs1.embedding_model == "fake-specter"
    assert gd1.embedding_model == "fake-bge"
    # dims follow the per-source-type embedder
    assert len(abs1.embedding or []) == 8
    assert len(gd1.embedding or []) == 12


def test_embed_does_not_touch_annotations() -> None:
    """update_embedding only patches the vector fields; mentions stay."""
    from foodscholar.io.chunk import EntityLink, Mention

    fs = _fs()
    m = Mention(text="olive oil", start=0, end=9, score=1.0, ner_model_version="x")
    link = EntityLink(
        mention=m,
        ontology_id="FOODON:03309927",
        confidence=0.9,
        method="dense",
        linker_version="v",
    )
    fs.upsert_chunks(
        [
            Chunk(
                chunk_id="c1",
                text="olive oil is great",
                source_doc_id="d",
                source_type="abstract",
                section_type="abstract",
                mentions=[m],
                entity_links=[link],
                foodon_ids=["FOODON:03309927"],
                enrichment_version="annotate-v2",
            )
        ]
    )
    fs.embed()
    c1 = fs.chunk_store.get("c1")
    assert c1 is not None
    assert c1.mentions == [m]
    assert c1.entity_links == [link]
    assert c1.foodon_ids == ["FOODON:03309927"]
    assert c1.enrichment_version == "annotate-v2"
    assert c1.embedding is not None


def test_embed_returns_zero_when_store_empty() -> None:
    fs = _fs()
    meta = fs.embed()
    assert meta.record_count == 0
