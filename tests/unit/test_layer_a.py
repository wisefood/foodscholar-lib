from pathlib import Path

from foodscholar.config import LayerAConfig
from foodscholar.io.chunk import Chunk
from foodscholar.layer_a import build_layer_a, build_shelves, shelf_id_for_foodon
from foodscholar.ontology import FoodOnAPI, load_ontology
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore


def _mini_foodon() -> FoodOnAPI:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"
    return FoodOnAPI(load_ontology(path), prefix_filter=None)


def _chunk(chunk_id: str, foodon_ids: list[str]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="fixture text",
        source_doc_id="fixture-doc",
        source_type="abstract",
        section_type="abstract",
        foodon_ids=foodon_ids,
    )


def _store() -> InMemoryChunkStore:
    store = InMemoryChunkStore()
    store.upsert(
        [
            _chunk("c1", ["TEST:0000008"]),
            _chunk("c2", ["TEST:0000006"]),
            _chunk("c3", ["TEST:0000008", "TEST:9999999"]),
        ]
    )
    return store


def test_build_shelves_propagates_foodon_support_to_ancestors() -> None:
    shelves = build_shelves(
        _store(),
        _mini_foodon(),
        LayerAConfig(min_support=2, max_depth=5),
    )

    by_id = {s.shelf_id: s for s in shelves}
    assert shelf_id_for_foodon("TEST:0000006") not in by_id
    assert by_id[shelf_id_for_foodon("TEST:0000001")].chunk_count == 3
    assert by_id[shelf_id_for_foodon("TEST:0000008")].chunk_count == 2
    assert by_id[shelf_id_for_foodon("TEST:0000008")].parent_shelf_id == shelf_id_for_foodon(
        "TEST:0000007"
    )
    assert [s.depth for s in shelves] == sorted(s.depth for s in shelves)


def test_build_shelves_honors_blacklist_and_reparents_to_nearest_included_ancestor() -> None:
    shelves = build_shelves(
        _store(),
        _mini_foodon(),
        LayerAConfig(min_support=1, max_depth=5, blacklist_terms=["plant food"]),
    )

    by_id = {s.shelf_id: s for s in shelves}
    assert shelf_id_for_foodon("TEST:0000002") not in by_id
    assert by_id[shelf_id_for_foodon("TEST:0000004")].parent_shelf_id == shelf_id_for_foodon(
        "TEST:0000001"
    )


def test_build_layer_a_upserts_shelves_and_returns_artifact_meta() -> None:
    from foodscholar import FoodScholar

    fs = FoodScholar.in_memory()
    fs.config.layer_a.min_support = 2
    fs.attach_ontology(_mini_foodon())
    fs.upsert_chunks(_store().scan())

    meta = fs.build_layer_a()

    assert meta.phase == "build-layer-a"
    assert meta.record_count == 5
    assert fs.graph_store.get_shelf(shelf_id_for_foodon("TEST:0000008")) is not None


def test_build_layer_a_function_writes_to_graph_store() -> None:
    from foodscholar import FoodScholarConfig

    chunk_store = _store()
    graph_store = InMemoryGraphStore()
    config = FoodScholarConfig.model_validate(
        {
            "corpus": {"chunks_path": "data/chunks.parquet"},
            "storage": {
                "chunk_store": {"backend": "memory"},
                "graph_store": {"backend": "memory"},
            },
            "layer_a": {"min_support": 2},
        }
    )

    meta = build_layer_a(
        chunk_store,
        graph_store,
        _mini_foodon(),
        config=config.layer_a,
        full_config=config,
    )

    assert meta.record_count == len(graph_store.list_shelves())
    assert graph_store.get_shelf(shelf_id_for_foodon("TEST:0000007")) is not None
