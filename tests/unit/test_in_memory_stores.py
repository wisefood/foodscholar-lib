from foodscholar.io.chunk import EntityLink, Mention
from foodscholar.io.graph import Card, Shelf, Theme
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore
from foodscholar.storage.protocols import ChunkStore, GraphStore


def test_chunk_store_implements_protocol() -> None:
    store = InMemoryChunkStore()
    assert isinstance(store, ChunkStore)


def test_graph_store_implements_protocol() -> None:
    store = InMemoryGraphStore()
    assert isinstance(store, GraphStore)


def test_chunk_upsert_and_get(mini_chunks) -> None:  # type: ignore[no-untyped-def]
    store = InMemoryChunkStore()
    store.upsert(mini_chunks)
    assert store.get("c1") is not None
    assert store.get("nope") is None
    assert len(store.get_many(["c1", "c2", "nope"])) == 2


def test_chunk_search_bm25_filters(mini_chunks) -> None:  # type: ignore[no-untyped-def]
    store = InMemoryChunkStore()
    store.upsert(mini_chunks)
    hits = store.search("olive oil heart", k=3)
    assert any(c.chunk_id == "c1" for c in hits)


def test_chunk_search_with_shelf_filter(mini_chunks) -> None:  # type: ignore[no-untyped-def]
    store = InMemoryChunkStore()
    store.upsert(mini_chunks)
    store.update_attachments("c1", shelf_ids=["s-med"], theme_ids=[])
    hits = store.search("cardiovascular", k=5, shelf_ids=["s-med"])
    assert {h.chunk_id for h in hits} == {"c1"}


def test_chunk_store_iter_chunks_batches(mini_chunks) -> None:  # type: ignore[no-untyped-def]
    store = InMemoryChunkStore()
    store.upsert(mini_chunks)
    batches = list(store.iter_chunks(batch_size=2))
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [c.chunk_id for batch in batches for c in batch] == [c.chunk_id for c in mini_chunks]


def test_chunk_store_iter_chunks_rejects_non_positive_batch_size() -> None:
    store = InMemoryChunkStore()
    import pytest

    with pytest.raises(ValueError):
        list(store.iter_chunks(batch_size=0))


def test_chunk_store_update_annotations(mini_chunks) -> None:  # type: ignore[no-untyped-def]
    store = InMemoryChunkStore()
    store.upsert(mini_chunks)
    mention = Mention(
        text="olive oil",
        start=0,
        end=9,
        score=0.99,
        ner_model_version="fixture-ner",
        entity_type="food",
    )
    link = EntityLink(
        mention=mention,
        ontology_id="TEST:0000008",
        confidence=0.95,
        method="lexical_exact",
        linker_version="fixture-linker",
    )

    store.update_annotations("c1", [mention], [link], ["TEST:0000008"], "fixture-v1")
    store.update_annotations("missing", [], [], [], "fixture-v1")

    chunk = store.get("c1")
    assert chunk is not None
    assert chunk.mentions == [mention]
    assert chunk.entity_links == [link]
    assert chunk.foodon_ids == ["TEST:0000008"]
    assert chunk.enrichment_version == "fixture-v1"


def test_graph_store_shelves_and_neighbors() -> None:
    g = InMemoryGraphStore()
    root = Shelf(shelf_id="s-root", label="Foods", facet="foods", depth=0)
    child = Shelf(
        shelf_id="s-grain", label="Whole grains", facet="foods", depth=1, parent_shelf_id="s-root"
    )
    g.upsert_shelves([root, child])
    assert g.get_shelf("s-root") == root
    assert "s-grain" in g.get_neighbors("s-root", hops=1)


def test_graph_store_theme_chunk_attachment() -> None:
    g = InMemoryGraphStore()
    s = Shelf(shelf_id="s-med", label="Mediterranean", facet="dietary_patterns", depth=1)
    t = Theme(
        theme_id="t-1",
        label="Olive oil",
        shelf_ids=["s-med"],
        discovered_by="leiden",
        discovery_version="v0",
        facet="dietary_patterns",
        discovery_pass="global_similarity",
    )
    g.upsert_shelves([s])
    g.upsert_themes([t])
    g.attach_chunks_to_theme("t-1", ["c1", "c5"])
    assert set(g.get_chunks_for_theme("t-1")) == {"c1", "c5"}
    assert g.get_themes_for_shelf("s-med") == [t]


def test_graph_store_card_roundtrip() -> None:
    g = InMemoryGraphStore()
    card = Card(
        card_id="card-1",
        target_id="s-med",
        target_type="shelf",
        title="Mediterranean",
        summary="...",
        evidence_quality="high",
        cited_chunk_ids=["c1"],
        llm_model="mock-llm-v0",
        prompt_version="v1",
    )
    g.upsert_cards([card])
    assert g.get_card("s-med", "shelf") == card
    assert g.get_card("nope", "shelf") is None
