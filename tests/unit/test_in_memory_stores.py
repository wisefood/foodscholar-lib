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
