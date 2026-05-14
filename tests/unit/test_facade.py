from pathlib import Path

import pytest

from foodscholar import FoodScholar, FoodScholarConfig
from foodscholar.io.chunk import Chunk
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore


def test_in_memory_constructs_with_defaults() -> None:
    fs = FoodScholar.in_memory()
    assert isinstance(fs.chunk_store, InMemoryChunkStore)
    assert isinstance(fs.graph_store, InMemoryGraphStore)
    assert fs.embedder.model_id == "mock-embedder-v0"
    assert fs.llm.model_id == "mock-llm-v0"
    info = fs.info()
    assert info["chunk_store"] == "memory"
    assert info["graph_store"] == "memory"


def test_in_memory_accepts_overrides() -> None:
    class Stub:
        model_id = "stub-llm"

        def generate(self, prompt: str, max_tokens: int = 1024) -> str:
            return ""

    fs = FoodScholar.in_memory(llm=Stub())
    assert fs.llm.model_id == "stub-llm"


def test_from_config_memory_backends(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "corpus:\n"
        "  chunks_path: data/chunks.parquet\n"
        "storage:\n"
        "  chunk_store:\n"
        "    backend: memory\n"
        "  graph_store:\n"
        "    backend: memory\n"
    )
    fs = FoodScholar.from_config(cfg)
    assert isinstance(fs.chunk_store, InMemoryChunkStore)
    assert isinstance(fs.graph_store, InMemoryGraphStore)


def test_from_config_accepts_pydantic_config() -> None:
    cfg = FoodScholarConfig.model_validate(
        {
            "corpus": {"chunks_path": "data/chunks.parquet"},
            "storage": {
                "chunk_store": {"backend": "memory"},
                "graph_store": {"backend": "memory"},
            },
        }
    )
    fs = FoodScholar.from_config(cfg)
    assert fs.config is cfg
    assert isinstance(fs.chunk_store, InMemoryChunkStore)


def test_upsert_chunks_routes_to_store() -> None:
    fs = FoodScholar.in_memory()
    fs.upsert_chunks(
        [
            Chunk(
                chunk_id="c1",
                text="olive oil and heart health",
                source_doc_id="d1",
                source_type="abstract",
                section_type="abstract",
            )
        ]
    )
    assert fs.chunk_store.get("c1") is not None


def test_deferred_phases_raise_not_implemented() -> None:
    fs = FoodScholar.in_memory()
    for method in ["annotate", "build_layer_a", "attach", "build_layer_b", "build_layer_c"]:
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            getattr(fs, method)()


def test_build_stops_at_first_deferred_phase() -> None:
    fs = FoodScholar.in_memory()
    with pytest.raises(NotImplementedError, match="'annotate'"):
        fs.build()


def test_query_raises_until_retrieval_lands() -> None:
    fs = FoodScholar.in_memory()
    with pytest.raises(NotImplementedError, match="'query'"):
        fs.query("anything")


def test_init_in_memory_is_noop() -> None:
    fs = FoodScholar.in_memory()
    fs.init()  # should not raise
