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
    assert _is_real_embedding("BAAI/bge-base-en-v1.5") is True
    assert _is_real_embedding("BAAI/bge-large-en-v1.5") is True
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
            _chunk("c1", embedding=[0.1, 0.2], model="BAAI/bge-base-en-v1.5"),
            _chunk("c2"),
        ]
    )
    meta = fs.embed()
    assert meta.record_count == 1  # only c2 encoded

    c1 = fs.chunk_store.get("c1")
    assert c1 is not None
    assert c1.embedding == [0.1, 0.2]
    assert c1.embedding_model == "BAAI/bge-base-en-v1.5"

    c2 = fs.chunk_store.get("c2")
    assert c2 is not None
    assert c2.embedding is not None
    assert c2.embedding_model == "mock-embedder-v0"


def test_embed_force_overwrites_when_only_missing_false() -> None:
    fs = _fs()
    fs.upsert_chunks(
        [_chunk("c1", embedding=[0.9, 0.8], model="BAAI/bge-base-en-v1.5")]
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


# --------------------------------------------------------------------- bulk path


def test_memory_store_update_embeddings_bulk_round_trips() -> None:
    """update_embeddings_bulk is the new hot path for fs.embed() over a tunneled
    cluster. The in-memory implementation is a simple loop — verify the contract."""
    fs = _fs()
    fs.upsert_chunks([_chunk("c1"), _chunk("c2"), _chunk("c3")])
    fs.chunk_store.update_embeddings_bulk(
        [
            ("c1", [0.1] * 8, "test-model"),
            ("c2", [0.2] * 8, "test-model"),
            ("c3", [0.3] * 8, "other-model"),
        ]
    )
    assert fs.chunk_store.get("c1").embedding == [0.1] * 8  # type: ignore[union-attr]
    assert fs.chunk_store.get("c1").embedding_model == "test-model"  # type: ignore[union-attr]
    assert fs.chunk_store.get("c3").embedding_model == "other-model"  # type: ignore[union-attr]


def test_memory_store_update_embeddings_bulk_empty_is_noop() -> None:
    fs = _fs()
    fs.upsert_chunks([_chunk("c1")])
    fs.chunk_store.update_embeddings_bulk([])
    assert fs.chunk_store.get("c1").embedding is None  # type: ignore[union-attr]


def test_embed_uses_bulk_writeback_one_call_per_flush() -> None:
    """fs.embed() should call update_embeddings_bulk ONCE per flush — not N
    times via per-doc update_embedding. This is the network-side win on a
    tunneled ES."""
    fs = _fs()
    fs.upsert_chunks([_chunk(f"c{i}") for i in range(10)])

    bulk_calls: list[int] = []
    single_calls: list[str] = []
    original_bulk = fs.chunk_store.update_embeddings_bulk
    original_single = fs.chunk_store.update_embedding

    def spy_bulk(items):  # type: ignore[no-untyped-def]
        bulk_calls.append(len(items))
        return original_bulk(items)

    def spy_single(chunk_id, embedding, embedding_model):  # type: ignore[no-untyped-def]
        single_calls.append(chunk_id)
        return original_single(chunk_id, embedding, embedding_model)

    fs.chunk_store.update_embeddings_bulk = spy_bulk  # type: ignore[method-assign]
    fs.chunk_store.update_embedding = spy_single  # type: ignore[method-assign]

    # batch_size=4 → expect 3 flushes (4, 4, 2)
    fs.embed(batch_size=4)

    assert bulk_calls == [4, 4, 2], f"expected three flushes, got {bulk_calls}"
    assert single_calls == [], "fs.embed() must not fall back to per-doc updates"


def test_embed_batches_one_encode_call_per_flush() -> None:
    """fs.embed() should issue ONE encode call per flush — not one encode per
    chunk. This is the GPU-side win that lets a batched accelerator amortize
    kernel launch overhead."""
    from foodscholar.annotate.embedder import HashEmbedder

    class CountingEmbedder(HashEmbedder):
        def __init__(self, *, dim: int, model_id: str) -> None:
            super().__init__(dim=dim)
            self.model_id = model_id  # type: ignore[misc]
            self.calls: list[int] = []

        def embed(self, texts):  # type: ignore[no-untyped-def]
            self.calls.append(len(texts))
            return super().embed(texts)

    embedder = CountingEmbedder(dim=8, model_id="fake-bge")

    fs = FoodScholar.in_memory()
    fs.embedder = embedder

    # 8 chunks in a single flush — must produce ONE encode call of size 8.
    chunks: list[Chunk] = []
    for i in range(3):
        chunks.append(
            Chunk(
                chunk_id=f"abs{i}",
                text=f"abstract {i}",
                source_doc_id="d",
                source_type="abstract",
                section_type="abstract",
            )
        )
    for i in range(5):
        chunks.append(
            Chunk(
                chunk_id=f"gd{i}",
                text=f"guide {i}",
                source_doc_id="d",
                source_type="guide",
                section_type="guideline",
            )
        )
    fs.upsert_chunks(chunks)
    fs.embed(batch_size=16)  # one flush

    assert embedder.calls == [8], f"embedder got {embedder.calls}"
