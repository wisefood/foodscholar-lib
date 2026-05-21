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
from foodscholar.storage.protocols import (
    NER,
    ChunkStore,
    Embedder,
    GraphStore,
    Linker,
    LLMClient,
)
from foodscholar.versioning import config_hash

if TYPE_CHECKING:
    from foodscholar.io.artifacts import ArtifactMeta
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
    """Built-in mock LLM for `in_memory()` and offline use.

    `generate_json` returns an empty object — enough to satisfy the protocol
    and let pipelines run without an LLM, but it produces no real annotations.
    Tests that exercise agentic behavior inject their own scripted client.
    """

    model_id = "mock-llm-v0"

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return "Mock answer citing [CHUNK]."

    def generate_json(
        self, prompt: str, schema: dict[str, object], max_tokens: int = 1024
    ) -> dict[str, object]:
        return {}


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
        self._ner: NER | None = None
        self._linker: Linker | None = None
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

        # Build the LLM client from cfg.llm unless the caller passed one
        # explicitly. No cfg.llm and no override → __init__ uses the mock.
        if llm is None and cfg.llm is not None:
            from foodscholar.llm import build_llm

            llm = build_llm(cfg.llm)

        # Build the chunk embedder from cfg.annotate unless the caller passed
        # one. SPECTER2 for abstracts, BGE-large for textbook/guide, routed by
        # source_type (BRIEF §2/§7). Falls back to the mock — with a loud
        # warning — if the [annotate] deps aren't installed.
        if embedder is None and chunk_backend != "memory":
            embedder = cls._build_embedder(cfg)

        return cls(
            config=cfg,
            chunk_store=chunk_store,
            graph_store=graph_store,
            embedder=embedder,
            llm=llm,
        )

    @staticmethod
    def _build_embedder(cfg: FoodScholarConfig) -> Embedder | None:
        """Construct the source-type-routed chunk embedder, or None to mock.

        SPECTER2 for abstracts, BGE-large for textbook/guide chunks, routed by
        `source_type` (BRIEF §2/§7). Returns None (→ `__init__` uses
        `_MockEmbedder`) when the `[annotate]` extra is missing OR the models
        fail to load — logging a warning so a production run can't silently
        produce a graph full of meaningless hash-vector embeddings.
        """
        log = get_logger("foodscholar")
        try:
            from foodscholar.annotate.embedder import HFEmbedder, SourceTypeRouter

            return SourceTypeRouter(
                scientific=HFEmbedder(cfg.annotate.scientific_embedder),
                general=HFEmbedder(cfg.annotate.general_embedder),
            )
        except Exception as e:
            log.warning(
                "embedder.unavailable",
                error=str(e),
                note="chunk embeddings will be MOCK — install: pip install 'foodscholar[annotate]'",
            )
            return None

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
            "ner": self.config.annotate.ner,
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

    # ------------------------------------------------------------------ annotate

    @property
    def ner(self) -> NER:
        """Lazily-built NER, selected by `cfg.annotate.ner`.

        `keyword` (default) → `KeywordNER.from_ontology(fs.ontology)`;
        `agentic` → `AgenticNER(fs.llm)`. Override with `fs.attach_ner(...)`
        before first access to install a custom NER.
        """
        if self._ner is None:
            self._ner = self._build_ner()
        return self._ner

    @property
    def linker(self) -> Linker:
        """Lazily-built linker. Default: `ThreeTierLinker(fs.ontology)`.

        Override with `fs.attach_linker(...)` to inject a custom linker or
        plug a dense embedder for tier 3.
        """
        if self._linker is None:
            self._linker = self._build_linker()
        return self._linker

    def attach_ner(self, ner: NER) -> None:
        self._ner = ner

    def attach_linker(self, linker: Linker) -> None:
        self._linker = linker

    def annotate(self) -> ArtifactMeta:
        """Run NER + linking + embedding over every chunk in `chunk_store`.

        Writes mentions, entity_links, foodon_ids, embedding, embedding_model,
        and enrichment_version back onto each chunk. Returns the artifact
        metadata so callers can persist it.
        """
        from foodscholar.annotate.runner import run

        return run(
            self.chunk_store,
            ner=self.ner,
            linker=self.linker,
            embedder=self.embedder,
            config=self.config,
        )

    def _build_ner(self) -> NER:
        """Build the NER chosen by `cfg.annotate.ner`.

        `agentic` uses the facade's LLM (`fs.llm`); with the default mock LLM
        it will extract nothing, so a real run needs `cfg.llm` configured.
        """
        if self.config.annotate.ner == "agentic":
            from foodscholar.annotate.agent_ner import AgenticNER

            return AgenticNER(self.llm)

        from foodscholar.annotate.ner import KeywordNER

        return KeywordNER.from_ontology(self.ontology)

    def _build_linker(self) -> Linker:
        from foodscholar.annotate.linker import ThreeTierLinker

        lc = self.config.annotate.linker

        # Dense tier — built only when a dense_model is configured.
        dense_embedder: Embedder | None = None
        if lc.dense_model:
            from foodscholar.annotate.embedder import SapBERTEmbedder

            dense_embedder = SapBERTEmbedder(lc.dense_model)

        # LLM-select tier — uses the facade's LLM, only when opted in.
        llm_client: LLMClient | None = self.llm if lc.llm_select else None

        return ThreeTierLinker(
            self.ontology,
            fuzzy_threshold=lc.lexical_threshold,
            dense_threshold=lc.dense_threshold,
            semantic_type_gate=lc.semantic_type_gate,
            dense_embedder=dense_embedder,
            dense_cache_path=str(lc.dense_cache_path) if lc.dense_cache_path else None,
            llm_client=llm_client,
            llm_select_threshold=lc.llm_select_threshold,
            llm_candidate_k=lc.llm_candidate_k,
        )

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
        prefix = tuple(cfg.prefix_filter) if cfg.prefix_filter is not None else None
        return FoodOnAPI(terms, prefix_filter=prefix)

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

    def build_layer_a(self) -> ArtifactMeta:
        """Build Layer A — the curated, multi-facet backbone from FoodOn."""
        from foodscholar.layer_a import build_layer_a

        return build_layer_a(
            self.chunk_store,
            self.graph_store,
            self.ontology,
            config=self.config.layer_a,
            full_config=self.config,
        )

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
