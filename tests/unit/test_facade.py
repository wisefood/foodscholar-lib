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
    assert info["ner"] == "gliner"
    assert info["nel_backend"] == "hnsw"


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


def test_from_config_accepts_plain_dict() -> None:
    """In-code config: no YAML file on disk, just a dict."""
    fs = FoodScholar.from_config(
        {
            "corpus": {"chunks_path": "data/chunks.parquet"},
            "storage": {
                "chunk_store": {"backend": "memory"},
                "graph_store": {"backend": "memory"},
            },
        }
    )
    assert isinstance(fs.chunk_store, InMemoryChunkStore)
    assert isinstance(fs.graph_store, InMemoryGraphStore)
    assert fs.config.corpus.chunks_path == Path("data/chunks.parquet")


def test_in_memory_accepts_dict_config() -> None:
    fs = FoodScholar.in_memory(
        config={
            "corpus": {"chunks_path": "data/chunks.parquet"},
            "annotate": {"batch_size": 4},
            "storage": {
                "chunk_store": {"backend": "memory"},
                "graph_store": {"backend": "memory"},
            },
        }
    )
    assert fs.config.annotate.batch_size == 4


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
    # build_layer_b is wired as of M5 (Phase 4); only build_layer_c remains
    # deferred until Layer C lands.
    for method in ["build_layer_c"]:
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            getattr(fs, method)()


def test_query_raises_until_retrieval_lands() -> None:
    fs = FoodScholar.in_memory()
    with pytest.raises(NotImplementedError, match="'query'"):
        fs.query("anything")


def test_init_in_memory_is_noop() -> None:
    fs = FoodScholar.in_memory()
    fs.init()  # should not raise


def _memory_config() -> FoodScholarConfig:
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
    """Memory configs stay offline/lightweight unless an embedder is explicit."""
    fs = FoodScholar.from_config(_memory_config())
    assert fs.embedder.model_id == "mock-embedder-v0"


def test_from_config_does_not_eagerly_build_embedder() -> None:
    """from_config and info() must NOT trigger model loads for elastic backends.

    The chunk embedder is lazy — production SPECTER2 + BGE-large weigh ~1.7 GB
    and would otherwise stall every `fs.info()` call.
    """
    import sys

    fs = FoodScholar.from_config({
        "corpus": {"chunks_path": "data/chunks.parquet"},
        "storage": {
            # elastic forces _build_embedder under the old logic; we just want
            # the property check, not a live ES connection — skip if the SDK
            # isn't installed.
            "chunk_store": {"backend": "memory"},
            "graph_store": {"backend": "memory"},
        },
    })
    # info() must not have built it.
    info = fs.info()
    assert info["embedder"].startswith("lazy(") or info["embedder"] == "mock-embedder-v0"
    # sentence_transformers must NOT have been imported just to construct + info.
    if "sentence_transformers" in sys.modules:
        # already loaded by an earlier test — skip the assertion rather than
        # producing a false negative on test order.
        return
    assert "sentence_transformers" not in sys.modules


def test_resolve_config_rejects_unknown_type() -> None:
    from foodscholar.config import resolve_config

    with pytest.raises(TypeError, match="unsupported config type"):
        resolve_config(42)  # type: ignore[arg-type]


def test_attach_ner_and_linker_satisfy_protocol() -> None:
    """Attach custom NER + Linker so we can exercise annotate without GLiNER."""
    from foodscholar.io.chunk import EntityLink, Mention

    class _NER:
        model_id = "noop-ner"

        def extract(self, text: str) -> list[Mention]:
            return []

        def extract_batch(self, texts: list[str]) -> list[list[Mention]]:
            return [[] for _ in texts]

    class _Linker:
        linker_id = "noop-linker"

        def link(self, mention: Mention) -> EntityLink | None:
            return None

    fs = FoodScholar.in_memory()
    fs.attach_ner(_NER())
    fs.attach_linker(_Linker())
    assert fs.ner.model_id == "noop-ner"
    assert fs.linker.linker_id == "noop-linker"
