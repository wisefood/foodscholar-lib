"""The `FoodScholar` facade — single entry point for the entire library.

Three equivalent ways to construct, all release-ready:

    # Pure in-code config — no YAML on disk.
    fs = FoodScholar.from_config({
        "corpus": {"chunks_path": "data/chunks.csv"},
        "ontology": {"foodon_path": "data/foodon.owl"},
        "storage": {"chunk_store": {"backend": "memory"},
                    "graph_store": {"backend": "memory"}},
    })

    # YAML file.
    fs = FoodScholar.from_config("config.yaml")

    # An already-validated config object.
    fs = FoodScholar.from_config(FoodScholarConfig(...))

For notebooks and tests the zero-config form skips backends entirely:

    fs = FoodScholar.in_memory()

The facade owns the wiring; phase modules stay pure (no I/O, no global state).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from foodscholar import __version__
from foodscholar.config import FoodScholarConfig, resolve_config
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

ConfigSource = str | Path | dict[str, Any] | FoodScholarConfig

_DEFERRED_TEMPLATE = (
    "phase '{phase}' is not implemented yet in foodscholar v{version}. "
    "See BRIEF.md §12 for the implementation order."
)


def _deferred(phase: str) -> NotImplementedError:
    return NotImplementedError(_DEFERRED_TEMPLATE.format(phase=phase, version=__version__))


class _MockEmbedder:
    """Deterministic toy embedder for the in-memory facade."""

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

    Only used for Layer C card generation now that NER is GLiNER and the
    linker is purely dense. Returns an empty JSON object — enough to satisfy
    the protocol and let pipelines run without an LLM, but it produces no
    real card text. Tests that exercise LLM behavior inject their own client.
    """

    model_id = "mock-llm-v0"

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return "Mock answer citing [CHUNK]."

    def generate_json(
        self, prompt: str, schema: dict[str, object], max_tokens: int = 1024
    ) -> dict[str, object]:
        return {}


class FoodScholar:
    """User-facing facade for the library."""

    def __init__(
        self,
        config: ConfigSource,
        *,
        chunk_store: ChunkStore,
        graph_store: GraphStore,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        cfg = resolve_config(config)
        self.config = cfg
        self.chunk_store = chunk_store
        self.graph_store = graph_store
        self.embedder: Embedder = embedder or _MockEmbedder()
        self.llm: LLMClient = llm or _MockLLM()
        self.config_hash = config_hash(cfg)
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
        config: ConfigSource | None = None,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
    ) -> FoodScholar:
        """Zero-config facade backed by `InMemoryChunkStore` + `InMemoryGraphStore`.

        Intended for notebooks, tests, and quick experiments. Pass a `config`
        (dict, YAML path, or `FoodScholarConfig`) to override defaults.
        """
        configure_logging()
        cfg = resolve_config(config) if config is not None else _minimal_memory_config()
        return cls(
            config=cfg,
            chunk_store=InMemoryChunkStore(),
            graph_store=InMemoryGraphStore(),
            embedder=embedder,
            llm=llm,
        )

    @classmethod
    def from_config(
        cls,
        config: ConfigSource,
        *,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
    ) -> FoodScholar:
        """Construct from a YAML path, a Python dict, or a validated config.

        Builds whichever stores the config declares. `memory` works today;
        `elastic` and `neo4j` adapters lift the same constructor.
        """
        configure_logging()
        cfg = resolve_config(config)

        chunk_backend = cfg.storage.chunk_store.backend
        if chunk_backend == "memory":
            chunk_store: ChunkStore = InMemoryChunkStore()
        elif chunk_backend == "elastic":
            from foodscholar.storage.elastic import ElasticChunkStore

            cs = cfg.storage.chunk_store
            chunk_store = ElasticChunkStore(
                url=cs.url or "http://localhost:9200",
                index=cs.index or "foodscholar_chunks",
                api_key=cs.api_key,
                username=cs.username,
                password=cs.password,
            )
        else:
            raise ValueError(f"unknown chunk_store backend: {chunk_backend}")

        graph_backend = cfg.storage.graph_store.backend
        if graph_backend == "memory":
            graph_store: GraphStore = InMemoryGraphStore()
        elif graph_backend == "neo4j":
            from foodscholar.storage.neo4j import Neo4jGraphStore

            gs = cfg.storage.graph_store
            graph_store = Neo4jGraphStore(
                url=gs.url or "bolt://localhost:7687",
                user=gs.user or "neo4j",
                password=gs.password,  # None → driver looks up NEO4J_PASSWORD
            )
        else:
            raise ValueError(f"unknown graph_store backend: {graph_backend}")

        if llm is None and cfg.llm is not None:
            from foodscholar.llm import build_llm

            llm = build_llm(cfg.llm)

        # Build the chunk embedder from cfg.annotate unless the caller passed
        # one. SPECTER2 for abstracts, BGE-large for textbook/guide, routed by
        # source_type (BRIEF §2/§7). Falls back to the mock — with a loud
        # warning — when the [annotate] deps are missing.
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
            "nel_backend": self.config.annotate.linker.nel_backend,
            "prompt_version": self.config.layer_c.prompt_version,
        }

    def load_chunks(self, path: str | Path) -> int:
        """Read chunks from a parquet/jsonl/csv path and upsert into the chunk store."""
        from foodscholar.corpus import load_chunks

        chunks = load_chunks(path)
        self.chunk_store.upsert(chunks)
        self._log.info("corpus.loaded", n=len(chunks), config_hash=self.config_hash)
        return len(chunks)

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Upsert an explicit list of chunks (useful in tests and notebooks)."""
        self.chunk_store.upsert(chunks)

    def load_and_annotate(
        self,
        path: str | Path,
        *,
        snapshot_path: str | Path | None = None,
    ) -> ArtifactMeta | None:
        """Single-pass: load chunks → run GLiNER+HNSW annotate → optional snapshot.

        This is the release-ready entry point that mirrors the validated
        prototype's `main()` — one call per corpus file. If a parquet snapshot
        is configured (via `snapshot_path` here or `cfg.corpus.annotated_snapshot_path`)
        and already exists with non-zero size, the call short-circuits and
        returns None — matching the prototype's skip-if-output-exists idempotency.
        """
        from foodscholar.corpus import write_chunks_parquet

        target = Path(snapshot_path) if snapshot_path is not None else (
            self.config.corpus.annotated_snapshot_path
        )
        if target is not None and target.exists() and target.stat().st_size > 0:
            self._log.info("load_and_annotate.skip_existing", snapshot=str(target))
            return None

        self.load_chunks(path)
        meta = self.annotate()

        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            n = write_chunks_parquet(self.chunk_store.scan(), target)
            self._log.info(
                "load_and_annotate.snapshot_written",
                snapshot=str(target),
                n=n,
            )
        return meta

    # ------------------------------------------------------------------ ontology

    @property
    def ontology(self) -> FoodOnAPI:
        if self._ontology is None:
            self._ontology = self._load_ontology()
        return self._ontology

    def load_ontology(self, *, refresh: bool = False) -> FoodOnAPI:
        if refresh:
            self._ontology = None
        return self.ontology

    def attach_ontology(self, api: FoodOnAPI) -> None:
        self._ontology = api

    # ------------------------------------------------------------------ annotate

    @property
    def ner(self) -> NER:
        """Lazily-built NER. `cfg.annotate.ner = 'gliner'` (the only choice in v0.1).

        Override with `fs.attach_ner(...)` before first access to install a custom NER.
        """
        if self._ner is None:
            self._ner = self._build_ner()
        return self._ner

    @property
    def linker(self) -> Linker:
        """Lazily-built linker. Default: `HNSWLinker` over `HNSWNELIndex`.

        First access builds the FoodOn term index (or loads it from the cache
        path) — that's the expensive call; subsequent accesses are free.
        """
        if self._linker is None:
            self._linker = self._build_linker()
        return self._linker

    def attach_ner(self, ner: NER) -> None:
        self._ner = ner

    def attach_linker(self, linker: Linker) -> None:
        self._linker = linker

    def annotate(self) -> ArtifactMeta:
        """Run NER + linking + embedding over every chunk in `chunk_store`."""
        from foodscholar.annotate.runner import run

        return run(
            self.chunk_store,
            ner=self.ner,
            linker=self.linker,
            embedder=self.embedder,
            config=self.config,
        )

    def _build_ner(self) -> NER:
        from foodscholar.annotate.gliner_ner import GLinerNER

        gc = self.config.annotate.gliner
        return GLinerNER(
            model_id=gc.model_id,
            threshold=gc.threshold,
            flat_ner=gc.flat_ner,
            labels=gc.labels,
            batch_size=gc.batch_size,
            max_length=gc.max_length,
        )

    def _build_linker(self) -> Linker:
        from foodscholar.annotate.linker import HNSWLinker
        from foodscholar.annotate.nel_index import (
            ElasticNELIndex,
            HNSWNELIndex,
            NELIndex,
        )

        lc = self.config.annotate.linker
        nel_index: NELIndex
        if lc.nel_backend == "hnsw":
            nel_index = HNSWNELIndex(
                self.ontology,
                encoder=lc.nel_encoder,
                top_k=lc.nel_top_k,
                min_sim=lc.nel_min_sim,
                index_path=lc.nel_index_path,
                metadata_path=lc.nel_metadata_path,
            )
        elif lc.nel_backend == "elastic":
            nel_index = ElasticNELIndex(
                url=self.config.storage.chunk_store.url or "",
                index=lc.es_index or "",
            )
        else:
            raise ValueError(f"unknown nel_backend: {lc.nel_backend}")
        return HNSWLinker(nel_index, min_sim=lc.nel_min_sim)

    def _load_ontology(self) -> FoodOnAPI:
        from foodscholar.ontology import FoodOnAPI, load_ontology

        if self.config.ontology is None:
            raise RuntimeError(
                "no ontology section in config — set `ontology.foodon_path` "
                "in your config, or call fs.attach_ontology(api) directly."
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
        """Provision both backing stores declared by the config.

        Calls `chunk_store.init()` and `graph_store.init()` — both methods are
        in the storage protocols and are no-ops for the in-memory backends, so
        this works uniformly regardless of where the stores live.
        """
        self.chunk_store.init()
        self.graph_store.init()
        self._log.info(
            "init.done",
            chunk_store=self.config.storage.chunk_store.backend,
            graph_store=self.config.storage.graph_store.backend,
            config_hash=self.config_hash,
        )

    def build_layer_a(self) -> ArtifactMeta:
        from foodscholar.layer_a import build_layer_a

        return build_layer_a(
            self.chunk_store,
            self.graph_store,
            self.ontology,
            config=self.config.layer_a,
            full_config=self.config,
        )

    def attach(self) -> None:
        raise _deferred("attach")

    def build_layer_b(self) -> None:
        raise _deferred("build-layer-b")

    def build_layer_c(self) -> None:
        raise _deferred("build-layer-c")

    def build(self) -> None:
        self.annotate()
        self.build_layer_a()
        self.attach()
        self.build_layer_b()
        self.build_layer_c()

    def query(self, text: str) -> Answer:
        raise _deferred("query")


def _minimal_memory_config() -> FoodScholarConfig:
    """Smallest valid config for `FoodScholar.in_memory()` with no args."""
    return FoodScholarConfig.model_validate(
        {
            "corpus": {"chunks_path": "data/chunks.parquet"},
            "storage": {
                "chunk_store": {"backend": "memory"},
                "graph_store": {"backend": "memory"},
            },
        }
    )
