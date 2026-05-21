from foodscholar.corpus import ChunkAnnotation, merge_annotations
from foodscholar.io.chunk import Chunk
from foodscholar.storage.memory import InMemoryChunkStore


def _store() -> InMemoryChunkStore:
    store = InMemoryChunkStore()
    store.upsert(
        [
            Chunk(
                chunk_id="c1",
                text="Olive oil is discussed.",
                source_doc_id="d1",
                source_type="abstract",
                section_type="abstract",
            )
        ]
    )
    return store


def test_merge_annotations_updates_foodon_ids() -> None:
    store = _store()
    n = merge_annotations(
        store,
        [
            ChunkAnnotation(
                chunk_id="c1",
                foodon_ids=["FOODON:2", "FOODON:1", "FOODON:1"],
                enrichment_version="fixture-v1",
            )
        ],
    )

    chunk = store.get("c1")
    assert n == 1
    assert chunk is not None
    assert chunk.foodon_ids == ["FOODON:1", "FOODON:2"]
    assert chunk.enrichment_version == "fixture-v1"


def test_merge_annotations_raises_for_unknown_chunk_in_strict_mode() -> None:
    store = _store()
    try:
        merge_annotations(store, [ChunkAnnotation(chunk_id="missing")])
    except KeyError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_merge_annotations_skips_unknown_chunk_in_non_strict_mode() -> None:
    store = _store()
    assert merge_annotations(
        store, [ChunkAnnotation(chunk_id="missing")], strict=False
    ) == 0
