"""The `FoodScholar` facade — single entry point for the entire library.

Most users should not construct stores, embedders, or phase modules by hand:

    from foodscholar import FoodScholar

    fs = FoodScholar.from_config("config.yaml")
    fs.load_chunks("data/chunks.parquet")
    fs.build()
    answer = fs.query("Is olive oil heart-healthy?")

For notebooks and tests, the zero-config form skips backends entirely:

    fs = FoodScholar.in_memory()

The facade owns the wiring; phase modules stay pure (no I/O, no global state).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from foodscholar import __version__
from foodscholar.config import FoodScholarConfig, load_config
from foodscholar.graph_view import GraphView
from foodscholar.logging import configure_logging, get_logger
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore
from foodscholar.storage.protocols import ChunkStore, Embedder, GraphStore, LLMClient
from foodscholar.versioning import config_hash

if TYPE_CHECKING:
    from foodscholar.io.chunk import Chunk
    from foodscholar.ontology import FoodOnAPI
    from foodscholar.retrieval import Answer

_DEFERRED_TEMPLATE = (
    "phase '{phase}' is not implemented yet in foodscholar v{version}. "
    "See BRIEF.md §12 for the implementation order."
)


def _deferred(phase: str) -> NotImplementedError:
    return NotImplementedError(_DEFERRED_TEMPLATE.format(phase=phase, version=__version__))


class _MockEmbedder:
    """Deterministic toy embedder for the in-memory facade.

    Mirrors the test-suite MockEmbedder so notebooks behave like the unit tests.
    """

    model_id = "mock-embedder-v0"

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            out.append([h[i % len(h)] / 255.0 for i in range(self._dim)])
        return out


class _MockLLM:
    model_id = "mock-llm-v0"

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return "Mock answer citing [CHUNK]."


class FoodScholar:
    """User-facing facade for the library.

    Holds the four pluggable backends (`chunk_store`, `graph_store`, `embedder`,
    `llm`) plus the validated config. Every public method is also exposed via
    the `foodscholar` CLI so the two surfaces stay in lockstep.
    """

    def __init__(
        self,
        config: FoodScholarConfig,
        *,
        chunk_store: ChunkStore,
        graph_store: GraphStore,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.chunk_store = chunk_store
        self.graph_store = graph_store
        self.embedder: Embedder = embedder or _MockEmbedder()
        self.llm: LLMClient = llm or _MockLLM()
        self.config_hash = config_hash(config)
        self.graph = GraphView(chunk_store, graph_store)
        self._ontology: FoodOnAPI | None = None
        self._log = get_logger("foodscholar")

    # ------------------------------------------------------------------ factories

    @classmethod
    def in_memory(
        cls,
        *,
        config: FoodScholarConfig | None = None,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
    ) -> FoodScholar:
        """Zero-config facade backed by `InMemoryChunkStore` + `InMemoryGraphStore`.

        Intended for notebooks, tests, and quick experiments. Pass a `config` to
        override defaults; otherwise a minimal in-memory config is used.
        """
        configure_logging()
        if config is None:
            config = _minimal_memory_config()
        return cls(
            config=config,
            chunk_store=InMemoryChunkStore(),
            graph_store=InMemoryGraphStore(),
            embedder=embedder,
            llm=llm,
        )

    @classmethod
    def from_config(
        cls,
        config: str | Path | FoodScholarConfig,
        *,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
    ) -> FoodScholar:
        """Construct from a YAML path or an already-validated config object.

        Constructs whichever stores the config declares. `memory` works today;
        `elastic` and `neo4j` raise `NotImplementedError` until those adapters
        land.
        """
        configure_logging()
        cfg = config if isinstance(config, FoodScholarConfig) else load_config(config)

        chunk_backend = cfg.storage.chunk_store.backend
        if chunk_backend == "memory":
            chunk_store: ChunkStore = InMemoryChunkStore()
        elif chunk_backend == "elastic":
            from foodscholar.storage.elastic import ElasticChunkStore

            chunk_store = ElasticChunkStore(
                url=cfg.storage.chunk_store.url or "",
                index=cfg.storage.chunk_store.index or "",
            )
        else:
            raise ValueError(f"unknown chunk_store backend: {chunk_backend}")

        graph_backend = cfg.storage.graph_store.backend
        if graph_backend == "memory":
            graph_store: GraphStore = InMemoryGraphStore()
        elif graph_backend == "neo4j":
            from foodscholar.storage.neo4j import Neo4jGraphStore

            graph_store = Neo4jGraphStore(
                url=cfg.storage.graph_store.url or "",
                user=cfg.storage.graph_store.user or "",
                password=cfg.storage.graph_store.password or "",
            )
        else:
            raise ValueError(f"unknown graph_store backend: {graph_backend}")

        return cls(
            config=cfg,
            chunk_store=chunk_store,
            graph_store=graph_store,
            embedder=embedder,
            llm=llm,
        )

    # ------------------------------------------------------------------ ergonomics

    def info(self) -> dict[str, str]:
        """Return a small dict describing version, backends, and active models."""
        ontology = "loaded" if self._ontology else (
            "configured" if self.config.ontology else "none"
        )
        return {
            "foodscholar": __version__,
            "config_hash": self.config_hash,
            "chunk_store": self.config.storage.chunk_store.backend,
            "graph_store": self.config.storage.graph_store.backend,
            "embedder": self.embedder.model_id,
            "llm": self.llm.model_id,
            "ontology": ontology,
            "ner_model": self.config.annotate.ner_model,
            "prompt_version": self.config.layer_c.prompt_version,
        }

    def load_chunks(self, path: str | Path) -> int:
        """Read chunks from a parquet/jsonl path and upsert into the chunk store.

        Returns the number of chunks loaded.
        """
        from foodscholar.corpus import load_chunks

        chunks = load_chunks(path)
        self.chunk_store.upsert(chunks)
        self._log.info("corpus.loaded", n=len(chunks), config_hash=self.config_hash)
        return len(chunks)

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Upsert an explicit list of chunks (useful in tests and notebooks)."""
        self.chunk_store.upsert(chunks)

    # ------------------------------------------------------------------ ontology

    @property
    def ontology(self) -> FoodOnAPI:
        """Lazily-loaded read-only FoodOn API.

        On first access, reads `cfg.ontology.foodon_path` (cached at
        `cfg.ontology.cache_path` if set). Raises a clear error if the config
        has no ontology section or the file is missing.
        """
        if self._ontology is None:
            self._ontology = self._load_ontology()
        return self._ontology

    def load_ontology(self, *, refresh: bool = False) -> FoodOnAPI:
        """Eagerly load (or reload) the ontology declared in the config."""
        if refresh:
            self._ontology = None
        return self.ontology

    def attach_ontology(self, api: FoodOnAPI) -> None:
        """Attach a pre-built FoodOnAPI (notebooks/tests).

        Skips the loader entirely. Useful for unit tests that build an API
        from a small in-memory term list.
        """
        self._ontology = api

    def _load_ontology(self) -> FoodOnAPI:
        from foodscholar.ontology import FoodOnAPI, load_ontology

        if self.config.ontology is None:
            raise RuntimeError(
                "no ontology section in config — set `ontology.foodon_path` "
                "in your YAML, or call fs.attach_ontology(api) directly."
            )
        cfg = self.config.ontology
        terms = load_ontology(
            cfg.foodon_path,
            cache_path=cfg.cache_path,
            include_imports=cfg.include_imports,
        )
        self._log.info(
            "ontology.loaded",
            n_terms=len(terms),
            source=str(cfg.foodon_path),
            cached=cfg.cache_path is not None,
        )
        return FoodOnAPI(terms)

    # ------------------------------------------------------------------ phases (stubs)

    def init(self) -> None:
        """Provision backing stores declared by the config (ES index, Neo4j constraints).

        No-op for in-memory backends.
        """
        chunk_backend = self.config.storage.chunk_store.backend
        graph_backend = self.config.storage.graph_store.backend
        if chunk_backend == "memory" and graph_backend == "memory":
            self._log.info("init.in_memory_noop", config_hash=self.config_hash)
            return
        self._log.warning(
            "init.backend_not_implemented",
            chunk_backend=chunk_backend,
            graph_backend=graph_backend,
        )

    def annotate(self) -> None:
        """Run NER + entity linking + embeddings over the loaded chunks."""
        raise _deferred("annotate")

    def build_layer_a(self) -> None:
        """Build Layer A — the curated, multi-facet backbone from FoodOn."""
        raise _deferred("build-layer-a")

    def attach(self) -> None:
        """Write chunk→shelf attachments and denormalize shelf_ids onto chunks."""
        raise _deferred("attach")

    def build_layer_b(self) -> None:
        """Build Layer B — theme communities per shelf."""
        raise _deferred("build-layer-b")

    def build_layer_c(self) -> None:
        """Build Layer C — LLM write-up cards for every shelf and theme."""
        raise _deferred("build-layer-c")

    def build(self) -> None:
        """Run annotate → build-layer-a → attach → build-layer-b → build-layer-c."""
        self.annotate()
        self.build_layer_a()
        self.attach()
        self.build_layer_b()
        self.build_layer_c()

    def query(self, text: str) -> Answer:
        """Run the retrieval pipeline (§14) against the built graph."""
        raise _deferred("query")


def _minimal_memory_config() -> FoodScholarConfig:
    """Smallest valid config for `FoodScholar.in_memory()`.

    Inline here (not in `config.py`) so it's clear this is a facade convenience,
    not a public config preset.
    """
    return FoodScholarConfig.model_validate(
        {
            "corpus": {"chunks_path": "data/chunks.parquet"},
            "storage": {
                "chunk_store": {"backend": "memory"},
                "graph_store": {"backend": "memory"},
            },
        }
    )
