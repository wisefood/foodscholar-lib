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
    for method in ["build_layer_a", "attach", "build_layer_b", "build_layer_c"]:
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            getattr(fs, method)()


def test_build_stops_at_first_deferred_phase() -> None:
    """`fs.build()` runs annotate (real) then trips on build-layer-a (deferred)."""
    from pathlib import Path

    from foodscholar import FoodOnAPI
    from foodscholar.ontology import load_ontology

    fs = FoodScholar.in_memory()
    fs.attach_ontology(
        FoodOnAPI(
            load_ontology(Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"),
            prefix_filter=None,
        )
    )
    with pytest.raises(NotImplementedError, match="'build-layer-a'"):
        fs.build()


def test_query_raises_until_retrieval_lands() -> None:
    fs = FoodScholar.in_memory()
    with pytest.raises(NotImplementedError, match="'query'"):
        fs.query("anything")


def test_init_in_memory_is_noop() -> None:
    fs = FoodScholar.in_memory()
    fs.init()  # should not raise


def _fs_with_mini_ontology(**linker_overrides) -> FoodScholar:  # type: ignore[no-untyped-def]
    from pathlib import Path

    from foodscholar import FoodOnAPI
    from foodscholar.ontology import load_ontology

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"
    fs = FoodScholar.in_memory()
    fs.attach_ontology(FoodOnAPI(load_ontology(fixture), prefix_filter=None))
    for k, v in linker_overrides.items():
        setattr(fs.config.annotate.linker, k, v)
    return fs


def test_linker_defaults_to_lexical_only() -> None:
    """No dense_model, llm_select off → linker has no dense index, no LLM."""
    fs = _fs_with_mini_ontology()
    linker = fs.linker
    assert linker._dense_index is None
    assert linker._llm is None


def test_linker_llm_select_wires_facade_llm() -> None:
    """cfg.annotate.linker.llm_select=True → linker uses the facade's LLM."""
    fs = _fs_with_mini_ontology(llm_select=True)
    linker = fs.linker
    assert linker._llm is fs.llm


def test_ner_defaults_to_keyword() -> None:
    """cfg.annotate.ner defaults to 'keyword' → fs.ner is a KeywordNER."""
    from foodscholar.annotate.ner import KeywordNER

    fs = _fs_with_mini_ontology()
    assert isinstance(fs.ner, KeywordNER)


def test_ner_agentic_selector_builds_agentic_ner() -> None:
    """cfg.annotate.ner='agentic' → fs.ner is an AgenticNER wrapping fs.llm."""
    from foodscholar.annotate.agent_ner import AgenticNER

    fs = _fs_with_mini_ontology()
    fs.config.annotate.ner = "agentic"
    ner = fs.ner
    assert isinstance(ner, AgenticNER)
    assert ner._llm is fs.llm


def _memory_config():  # type: ignore[no-untyped-def]
    from foodscholar import FoodScholarConfig

    return FoodScholarConfig.model_validate(
        {
            "corpus": {"chunks_path": "data/chunks.parquet"},
            "storage": {
                "chunk_store": {"backend": "memory"},
                "graph_store": {"backend": "memory"},
            },
        }
    )


def test_from_config_explicit_embedder_is_respected() -> None:
    """An embedder passed to from_config wins over the config-built one."""
    from foodscholar.annotate.embedder import HashEmbedder

    custom = HashEmbedder(dim=12)
    fs = FoodScholar.from_config(_memory_config(), embedder=custom)
    assert fs.embedder is custom


def test_from_config_embedder_degrades_to_mock_without_deps() -> None:
    """When the [annotate] embedder deps are absent, from_config falls back to
    the mock embedder rather than crashing — `_build_embedder` returns None and
    __init__ supplies _MockEmbedder."""
    import importlib.util

    fs = FoodScholar.from_config(_memory_config())
    if importlib.util.find_spec("sentence_transformers") is None:
        # No sentence-transformers in this env → SourceTypeRouter can't build.
        assert fs.embedder.model_id == "mock-embedder-v0"
    else:
        # Deps present → a real source-type router was built.
        assert fs.embedder.model_id.startswith("router(")
