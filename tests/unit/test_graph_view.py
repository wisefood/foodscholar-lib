import pytest

from foodscholar import FoodScholar
from foodscholar.graph_view import CardHandle, ShelfHandle, ThemeHandle
from foodscholar.io.chunk import Chunk
from foodscholar.io.graph import Card, Shelf


@pytest.fixture
def fs() -> FoodScholar:
    return FoodScholar.in_memory()


def test_graph_attached_to_facade(fs: FoodScholar) -> None:
    assert fs.graph is not None
    assert fs.graph.summary() == {"shelves": 0, "themes": 0, "roots": 0}


def test_add_shelf_kwargs_returns_handle(fs: FoodScholar) -> None:
    h = fs.graph.add_shelf(
        shelf_id="s-med", label="Mediterranean diet", facet="dietary_patterns", depth=1
    )
    assert isinstance(h, ShelfHandle)
    assert h.shelf_id == "s-med"
    assert h.label == "Mediterranean diet"
    assert h.facet == "dietary_patterns"


def test_add_shelf_model_arg(fs: FoodScholar) -> None:
    s = Shelf(shelf_id="s-foo", label="Foo", facet="foods", depth=0)
    h = fs.graph.add_shelf(s)
    assert h.shelf_id == "s-foo"
    assert fs.graph.shelf("s-foo") is not None


def test_listings_and_facet_filter(fs: FoodScholar) -> None:
    fs.graph.add_shelf(shelf_id="s-foods", label="Foods", facet="foods", depth=0)
    fs.graph.add_shelf(
        shelf_id="s-med",
        label="Mediterranean",
        facet="dietary_patterns",
        depth=1,
        parent_shelf_id="s-foods",
    )
    fs.graph.add_shelf(shelf_id="s-allergy", label="Allergies", facet="allergies", depth=1)

    all_shelves = fs.graph.shelves()
    assert {h.shelf_id for h in all_shelves} == {"s-foods", "s-med", "s-allergy"}

    foods = fs.graph.shelves(facet="foods")
    assert [h.shelf_id for h in foods] == ["s-foods"]

    roots = fs.graph.roots()
    assert {h.shelf_id for h in roots} == {"s-foods", "s-allergy"}


def test_shelf_handle_navigation(fs: FoodScholar) -> None:
    fs.graph.add_shelf(shelf_id="s-foods", label="Foods", facet="foods", depth=0)
    fs.graph.add_shelf(
        shelf_id="s-med",
        label="Mediterranean",
        facet="dietary_patterns",
        depth=1,
        parent_shelf_id="s-foods",
    )

    parent = fs.graph.shelf("s-med").parent()
    assert parent is not None
    assert parent.shelf_id == "s-foods"

    children = fs.graph.shelf("s-foods").children()
    assert [h.shelf_id for h in children] == ["s-med"]


def test_attach_chunks_to_shelf_denormalizes(fs: FoodScholar) -> None:
    fs.graph.add_shelf(shelf_id="s-med", label="Mediterranean", facet="dietary_patterns", depth=1)
    fs.upsert_chunks(
        [
            Chunk(
                chunk_id="c1",
                text="olive oil",
                source_doc_id="d1",
                source_type="abstract",
                section_type="abstract",
            )
        ]
    )
    fs.graph.attach_chunks(["c1"], shelf="s-med")

    c1 = fs.chunk_store.get("c1")
    assert c1.shelf_ids == ["s-med"]

    chunks = fs.graph.shelf("s-med").chunks()
    assert [c.chunk_id for c in chunks] == ["c1"]


def test_attach_chunks_is_idempotent_and_merges(fs: FoodScholar) -> None:
    fs.graph.add_shelf(shelf_id="s-a", label="A", facet="foods", depth=0)
    fs.graph.add_shelf(shelf_id="s-b", label="B", facet="foods", depth=0)
    fs.upsert_chunks(
        [
            Chunk(
                chunk_id="c1",
                text="t",
                source_doc_id="d",
                source_type="abstract",
                section_type="abstract",
            )
        ]
    )
    fs.graph.attach_chunks(["c1"], shelf="s-a")
    fs.graph.attach_chunks(["c1"], shelf="s-a")  # repeat
    fs.graph.attach_chunks(["c1"], shelf="s-b")

    assert set(fs.chunk_store.get("c1").shelf_ids) == {"s-a", "s-b"}


def test_attach_chunks_requires_one_target(fs: FoodScholar) -> None:
    with pytest.raises(ValueError, match=r"shelf=|theme="):
        fs.graph.attach_chunks(["c1"])
    with pytest.raises(ValueError, match="not both"):
        fs.graph.attach_chunks(["c1"], shelf="s", theme="t")


def test_theme_navigation(fs: FoodScholar) -> None:
    fs.graph.add_shelf(shelf_id="s-med", label="Mediterranean", facet="dietary_patterns", depth=1)
    fs.graph.add_theme(
        theme_id="t-olive",
        label="Olive oil",
        shelf_ids=["s-med"],
        discovered_by="leiden",
        discovery_version="v0",
        facet="dietary_patterns",
        discovery_pass="similarity",
    )
    fs.upsert_chunks(
        [
            Chunk(
                chunk_id="c1",
                text="t",
                source_doc_id="d",
                source_type="abstract",
                section_type="abstract",
            )
        ]
    )
    fs.graph.attach_chunks(["c1"], theme="t-olive")

    theme = fs.graph.theme("t-olive")
    assert isinstance(theme, ThemeHandle)
    assert [s.shelf_id for s in theme.shelves()] == ["s-med"]
    assert [c.chunk_id for c in theme.chunks()] == ["c1"]
    assert fs.chunk_store.get("c1").theme_ids == ["t-olive"]


def test_card_handle_and_target(fs: FoodScholar) -> None:
    fs.graph.add_shelf(shelf_id="s-med", label="Mediterranean", facet="dietary_patterns", depth=1)
    fs.upsert_chunks(
        [
            Chunk(
                chunk_id="c1",
                text="olive oil",
                source_doc_id="d",
                source_type="abstract",
                section_type="abstract",
            )
        ]
    )
    card = fs.graph.add_card(
        Card(
            card_id="card-1",
            target_id="s-med",
            target_type="shelf",
            title="Med",
            summary="...",
            evidence_quality="high",
            cited_chunk_ids=["c1"],
            llm_model="mock",
            prompt_version="v1",
        )
    )
    assert isinstance(card, CardHandle)
    target = card.target()
    assert target is not None
    assert target.shelf_id == "s-med"
    cited = card.cited_chunks()
    assert [c.chunk_id for c in cited] == ["c1"]

    assert fs.graph.shelf("s-med").card().card_id == "card-1"


def test_search_scoped_to_shelf(fs: FoodScholar) -> None:
    fs.graph.add_shelf(shelf_id="s-med", label="Mediterranean", facet="dietary_patterns", depth=1)
    fs.upsert_chunks(
        [
            Chunk(
                chunk_id="c1",
                text="olive oil reduces cardiovascular risk",
                source_doc_id="d1",
                source_type="abstract",
                section_type="abstract",
            ),
            Chunk(
                chunk_id="c2",
                text="iron rich foods",
                source_doc_id="d2",
                source_type="textbook",
                section_type="textbook",
            ),
        ]
    )
    fs.graph.attach_chunks(["c1"], shelf="s-med")

    hits = fs.graph.search("olive oil cardiovascular", shelf="s-med", k=5)
    assert [h.chunk_id for h in hits] == ["c1"]


def test_lookup_misses_return_none(fs: FoodScholar) -> None:
    assert fs.graph.shelf("nope") is None
    assert fs.graph.theme("nope") is None
    assert fs.graph.card("nope", "shelf") is None
    assert fs.graph.chunk("nope") is None


def test_handle_repr(fs: FoodScholar) -> None:
    h = fs.graph.add_shelf(shelf_id="s-1", label="One", facet="foods", depth=0)
    assert "ShelfHandle" in repr(h)
    assert "s-1" in repr(h)
