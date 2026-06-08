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

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foodscholar import __version__
from foodscholar.config import FoodScholarConfig, resolve_config
from foodscholar.graph_view import GraphView
from foodscholar.logging import configure_logging, get_logger
from foodscholar.storage.memory import (
    InMemoryCardStore,
    InMemoryChunkStore,
    InMemoryEntityStore,
    InMemoryGraphStore,
)
from foodscholar.storage.protocols import (
    NER,
    CardStore,
    ChunkStore,
    Embedder,
    EntityStore,
    GraphStore,
    Linker,
    LLMClient,
)
from foodscholar.versioning import config_hash

if TYPE_CHECKING:
    from foodscholar.evaluation.audit import AuditReport
    from foodscholar.evaluation.quality import QualityReport
    from foodscholar.io.artifacts import ArtifactMeta
    from foodscholar.io.chunk import Chunk
    from foodscholar.io.graph import Card
    from foodscholar.layer_a.semantic_consolidation import ConsolidationArtifact
    from foodscholar.ontology import FoodOnAPI
    from foodscholar.retrieval import Answer

ConfigSource = str | Path | dict[str, Any] | FoodScholarConfig

# Model ids that don't count as "real" embeddings — used by fs.embed() to
# decide whether a chunk needs encoding under `only_missing=True`. The list
# is small on purpose: anything outside it is treated as a real production
# embedder so we don't accidentally re-encode BGE vectors.
_MOCK_EMBEDDING_MODELS: frozenset[str] = frozenset(
    {"mock-embedder-v0", "hash-embedder-v0"}
)


def _is_real_embedding(model_id: str | None) -> bool:
    """True iff the chunk already carries a real (non-mock) embedding."""
    return bool(model_id) and model_id not in _MOCK_EMBEDDING_MODELS


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
        entity_store: EntityStore | None = None,
        card_store: CardStore | None = None,
    ) -> None:
        cfg = resolve_config(config)
        self.config = cfg
        self.chunk_store = chunk_store
        self.graph_store = graph_store
        # Vector store for Layer C cards. Defaults to in-memory so the facade
        # works without ES; production wiring builds an ElasticCardStore.
        self.card_store: CardStore = card_store or InMemoryCardStore()
        # Embedder is built lazily on first access — the production BGE-base
        # embedder costs ~440 MB of model weights to load, which we don't want
        # to pay just to print info().
        self._embedder: Embedder | None = embedder
        self.llm: LLMClient = llm or _MockLLM()
        # First-class entity store. Mirrors the chunk-store backend (memory ↔
        # in-memory, elastic ↔ Elastic entity index). Built eagerly because
        # the adapter ctor is cheap and we want fs.entities to work without
        # extra wiring.
        self.entity_store: EntityStore = entity_store or InMemoryEntityStore()
        self.config_hash = config_hash(cfg)
        self.graph = GraphView(chunk_store, graph_store)
        self.entities = _EntityView(self)
        # Visualization view — builds VizGraphs on demand. Renderers are
        # lazy-imported behind the [viz] extra so this import is free.
        from foodscholar.viz import VizView

        self.viz = VizView(self)
        self._ontology: FoodOnAPI | None = None
        self._ner: NER | None = None
        self._linker: Linker | None = None
        self._log = get_logger("foodscholar")

    @property
    def embedder(self) -> Embedder:
        """Lazily-built chunk embedder.

        For `memory` backends or when the `[annotate]` extra is missing, this
        is `_MockEmbedder()`. For production it is
        `HFEmbedder("BAAI/bge-base-en-v1.5")` (BRIEF §2 — single embedder
        across source types). First access pays the model-load cost;
        subsequent accesses are free.
        """
        if self._embedder is None:
            chunk_backend = self.config.storage.chunk_store.backend
            if chunk_backend == "memory":
                self._embedder = _MockEmbedder()
            else:
                built = self._build_embedder(self.config)
                self._embedder = built if built is not None else _MockEmbedder()
        return self._embedder

    @embedder.setter
    def embedder(self, value: Embedder) -> None:
        self._embedder = value

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
            entity_store=InMemoryEntityStore(),
            card_store=InMemoryCardStore(),
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
        entity_store: EntityStore
        if chunk_backend == "memory":
            chunk_store: ChunkStore = InMemoryChunkStore()
            entity_store = InMemoryEntityStore()
        elif chunk_backend == "elastic":
            from foodscholar.storage.elastic import ElasticChunkStore
            from foodscholar.storage.elastic_entities import ElasticEntityStore

            cs = cfg.storage.chunk_store
            chunk_store = ElasticChunkStore(
                url=cs.url or "http://localhost:9200",
                index=cs.index or "foodscholar_chunks",
                api_key=cs.api_key,
                username=cs.username,
                password=cs.password,
                bulk_size=cs.bulk_size,
            )
            # The entity index name mirrors the chunk index with an `_entities`
            # suffix so the two artifacts are always paired. Same bulk_size
            # — the paired indexes are a single tuning surface.
            entity_index = (cs.index or "foodscholar_chunks") + "_entities"
            entity_store = ElasticEntityStore(
                url=cs.url or "http://localhost:9200",
                index=entity_index,
                api_key=cs.api_key,
                username=cs.username,
                password=cs.password,
                bulk_size=cs.bulk_size,
            )
        else:
            raise ValueError(f"unknown chunk_store backend: {chunk_backend}")

        card_backend = cfg.storage.card_store.backend
        if card_backend == "memory":
            card_store: CardStore = InMemoryCardStore()
        elif card_backend == "elastic":
            from foodscholar.storage.elastic import ElasticCardStore

            cds = cfg.storage.card_store
            card_store = ElasticCardStore(
                url=cds.url or "http://localhost:9200",
                index=cds.index or "foodscholar_cards",
                api_key=cds.api_key,
                username=cds.username,
                password=cds.password,
                bulk_size=cds.bulk_size,
            )
        else:
            raise ValueError(f"unknown card_store backend: {card_backend}")

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

        # The chunk embedder is built lazily on first access (the `embedder`
        # property on the facade), so the ~440 MB BGE-base load does NOT
        # happen here just for fs.info() or fs.init(). An explicit `embedder=`
        # kwarg still wins — it skips the lazy build entirely.

        return cls(
            config=cfg,
            chunk_store=chunk_store,
            graph_store=graph_store,
            embedder=embedder,
            llm=llm,
            entity_store=entity_store,
            card_store=card_store,
        )

    @staticmethod
    def _build_embedder(cfg: FoodScholarConfig) -> Embedder | None:
        log = get_logger("foodscholar")
        try:
            from foodscholar.annotate.embedder import HFEmbedder

            return HFEmbedder(cfg.annotate.embedder)
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
        # `embedder` is lazy: never force it just to print info(). When
        # unbuilt, report which backend WILL be built on first use.
        if self._embedder is not None:
            embedder = self._embedder.model_id
        elif self.config.storage.chunk_store.backend == "memory":
            embedder = "lazy(mock)"
        else:
            embedder = f"lazy({self.config.annotate.embedder})"
        return {
            "foodscholar": __version__,
            "config_hash": self.config_hash,
            "chunk_store": self.config.storage.chunk_store.backend,
            "graph_store": self.config.storage.graph_store.backend,
            "embedder": embedder,
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
        ignore_source_types: set[str] | None = None,
    ) -> ArtifactMeta | None:
        """Single-pass: load chunks → run GLiNER+HNSW annotate → optional snapshot.

        This is the release-ready entry point that mirrors the validated
        prototype's `main()` — one call per corpus file. If a parquet snapshot
        is configured (via `snapshot_path` here or `cfg.corpus.annotated_snapshot_path`)
        and already exists with non-zero size, the call short-circuits and
        returns None — matching the prototype's skip-if-output-exists idempotency.

        ``ignore_source_types`` (defaults to ``cfg.corpus.ignore_source_types``)
        drops chunks whose ``source_type`` is in the set before annotation —
        useful to skip e.g. all ``abstract`` chunks when ingesting a
        guideline-only knowledge base.
        """
        from foodscholar.corpus import iter_chunks, write_chunks_parquet

        target = Path(snapshot_path) if snapshot_path is not None else (
            self.config.corpus.annotated_snapshot_path
        )
        if target is not None and target.exists() and target.stat().st_size > 0:
            self._log.info("load_and_annotate.skip_existing", snapshot=str(target))
            return None

        skip = self._resolve_ignored_source_types(ignore_source_types)
        if skip:
            kept = [c for c in iter_chunks(path) if c.source_type not in skip]
            n_skipped = self._count_skipped_chunks(path, skip)
            self.chunk_store.upsert(kept)
            self._log.info(
                "load_and_annotate.filtered",
                n_loaded=len(kept),
                n_skipped=n_skipped,
                ignore_source_types=sorted(skip),
            )
        else:
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

    def ingest(
        self,
        corpus_dir: str | Path,
        *,
        nel_dir: str | Path | None = None,
        snapshot_path: str | Path | None = None,
        ignore_source_types: set[str] | None = None,
    ) -> ArtifactMeta | None:
        """Ingest corpus + annotations into `fs.chunk_store`. Two modes:

        - ``nel_dir`` **supplied** → annotations come from pre-computed
          `(chunk_id, chunk_entities_ner, chunk_uri_nel)` CSVs (the prototype's
          output shape). Chunks are loaded from ``corpus_dir``, annotations
          attached by `chunk_id`, and everything upserted to `fs.chunk_store`.
          **No GLiNER, no HNSW, no chunk embedding** — fast, deterministic,
          works without the `[annotate]` extra installed. Call `fs.embed()`
          afterwards to fill in chunk vectors for kNN search.

        - ``nel_dir`` **omitted** → falls back to `load_and_annotate(corpus_dir)`
          which runs GLiNER + HNSW from scratch (this path does embed, since
          the runner has the BGE-base embedder already loaded).

        ``snapshot_path`` (or `cfg.corpus.annotated_snapshot_path`) writes a
        parquet snapshot of the annotated chunks after ingest. If the snapshot
        already exists and is non-empty the whole call short-circuits — the
        same idempotency guarantee as `load_and_annotate`.

        ``ignore_source_types`` (defaults to ``cfg.corpus.ignore_source_types``)
        drops chunks whose ``source_type`` is in the set before upsert. Their
        NEL rows are skipped too — nothing about them reaches the chunk store
        or shows up in ``fs.entities`` later. Pass e.g. ``{"abstract"}`` to
        ingest only textbook + guide chunks.
        """
        from foodscholar.corpus import (
            iter_chunks,
            load_nel_dir,
            write_chunks_parquet,
        )

        target = Path(snapshot_path) if snapshot_path is not None else (
            self.config.corpus.annotated_snapshot_path
        )
        if target is not None and target.exists() and target.stat().st_size > 0:
            self._log.info("ingest.skip_existing", snapshot=str(target))
            return None

        if nel_dir is None:
            return self.load_and_annotate(
                corpus_dir,
                snapshot_path=snapshot_path,
                ignore_source_types=ignore_source_types,
            )

        nel_map = load_nel_dir(nel_dir)
        skip = self._resolve_ignored_source_types(ignore_source_types)

        # Stream chunks, attach annotations, upsert in batches.
        #
        # No embedding happens here — that is a separate, opt-in step
        # (`fs.embed()`). Reasons:
        #   - The prototype's `nel_*.csv` files carry mentions + linked URIs
        #     only, not chunk vectors. There is nothing to attach beyond text.
        #   - The production chunk embedder (BGE-base) costs ~440 MB of model
        #     load + per-chunk encoding; making that mandatory at ingest forces
        #     users to wait for vector machinery they may not need (BM25 +
        #     filtered search work without it).
        #   - Iterating on annotations should be cheap. Keeping embed as its
        #     own step means re-ingesting after a NEL refresh doesn't redo
        #     the BGE-base pass.
        from foodscholar.annotate.runner import ENRICHMENT_VERSION
        from foodscholar.versioning import make_artifact_meta

        batch_size = max(1, self.config.annotate.batch_size)
        batch: list = []
        n_chunks = 0
        n_links = 0
        n_attached = 0
        n_skipped = 0

        def _flush(batch: list) -> None:
            nonlocal n_chunks, n_links, n_attached
            if not batch:
                return
            enriched = []
            for chunk in batch:
                ann = nel_map.get(chunk.chunk_id)
                if ann is None:
                    mentions, links, foodon_ids = [], [], []
                else:
                    mentions, links, foodon_ids = ann
                    n_attached += 1
                enriched.append(
                    chunk.model_copy(
                        update={
                            "mentions": mentions,
                            "entity_links": links,
                            "foodon_ids": foodon_ids,
                            "enrichment_version": ENRICHMENT_VERSION,
                        }
                    )
                )
                n_links += len(links)
            self.chunk_store.upsert(enriched)
            n_chunks += len(enriched)

        for chunk in iter_chunks(corpus_dir):
            if chunk.source_type in skip:
                n_skipped += 1
                continue
            batch.append(chunk)
            if len(batch) >= batch_size:
                _flush(batch)
                batch = []
        _flush(batch)

        meta = make_artifact_meta(
            phase="ingest",
            config=self.config,
            record_count=n_chunks,
        )
        self._log.info(
            "ingest.done",
            n_chunks=n_chunks,
            n_attached=n_attached,
            n_links=n_links,
            n_skipped=n_skipped,
            ignore_source_types=sorted(skip),
            artifact_id=meta.artifact_id,
            config_hash=meta.config_hash,
        )

        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            n = write_chunks_parquet(self.chunk_store.scan(), target)
            self._log.info("ingest.snapshot_written", snapshot=str(target), n=n)

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

    def embed(
        self,
        *,
        only_missing: bool = True,
        batch_size: int = 64,
    ) -> ArtifactMeta:
        """Fill in chunk-text embeddings for chunks already in `fs.chunk_store`.

        Walks the store, encodes each chunk with the configured embedder
        (BGE-base for production — BRIEF §2/§7), and writes back only the
        `embedding` + `embedding_model` fields via
        `chunk_store.update_embeddings_bulk`. Mentions, links, and other
        annotations are untouched.

        - `only_missing=True` (default): skip chunks whose `embedding_model`
          is already a real model id (anything that isn't the deterministic
          `mock-embedder-v0`). Re-runs are cheap.
        - `only_missing=False`: re-encode every chunk regardless. Useful
          after swapping the configured embedder.

        Builds the production embedder lazily on first call — that is the
        ~440 MB BGE-base load, paid once per process.
        """
        from foodscholar.versioning import make_artifact_meta

        embedder = self.embedder  # lazy build happens here on first call

        n_seen = 0
        n_embedded = 0
        n_skipped = 0
        pending_chunks: list = []

        def _flush() -> None:
            """Encode the pending batch and bulk-write the results.

            ONE encode call over the batch (the T4 amortization win) followed
            by ONE bulk write covering all results (the network round-trip win).
            """
            nonlocal n_embedded
            if not pending_chunks:
                return

            vecs = embedder.embed([c.text for c in pending_chunks])
            updates = [
                (c.chunk_id, v, embedder.model_id)
                for c, v in zip(pending_chunks, vecs, strict=True)
            ]

            self.chunk_store.update_embeddings_bulk(updates)
            n_embedded += len(updates)
            pending_chunks.clear()

        for batch in self.chunk_store.iter_chunks(batch_size=batch_size):
            for chunk in batch:
                n_seen += 1
                if only_missing and _is_real_embedding(chunk.embedding_model):
                    n_skipped += 1
                    continue
                pending_chunks.append(chunk)
                if len(pending_chunks) >= batch_size:
                    _flush()
        _flush()

        meta = make_artifact_meta(
            phase="embed",
            config=self.config,
            record_count=n_embedded,
        )
        self._log.info(
            "embed.done",
            n_seen=n_seen,
            n_embedded=n_embedded,
            n_skipped=n_skipped,
            embedder=embedder.model_id,
            artifact_id=meta.artifact_id,
            config_hash=meta.config_hash,
        )
        return meta

    def build_entities(
        self,
        *,
        cap_chunk_sample: int | None = None,
    ) -> ArtifactMeta:
        """Derive first-class `Entity` records from the chunks already in the
        store and write them to (a) `fs.entity_store`, (b) `fs.graph_store`
        as `(:Entity)` nodes with `(:Chunk)-[:MENTIONS]->(:Entity)` edges.

        Walks `fs.chunk_store.iter_chunks(...)`, dedupes `EntityLink`s by
        `ontology_id`, aggregates `(mention_count, chunk_count, chunk_ids,
        facet_hint, last_seen)`, and enriches with `(label, synonyms,
        ancestor_ids)` from `fs.ontology` when an ontology is configured AND
        the entity id is a FOODON id (other OBO prefixes ship with the
        most-frequent surface form as the label and no ancestors).

        Idempotent — re-running over an unchanged corpus produces the same
        Entity records; re-running after `fs.ingest` of new chunks updates
        counts and the chunk_ids sample.
        """
        from foodscholar.io.entity import ENTITY_CHUNK_SAMPLE_CAP, Entity
        from foodscholar.versioning import make_artifact_meta

        sample_cap = cap_chunk_sample if cap_chunk_sample is not None else ENTITY_CHUNK_SAMPLE_CAP
        ontology = self._ontology  # use only if already loaded — never force a load here

        aggregates: dict[str, dict[str, object]] = {}
        # per-entity: chunk_id -> (confidence, method) for the Neo4j edges
        chunk_edges: dict[str, dict[str, tuple[float, str]]] = {}

        n_chunks_seen = 0
        for batch in self.chunk_store.iter_chunks(batch_size=500):
            for chunk in batch:
                n_chunks_seen += 1
                for link in chunk.entity_links:
                    oid = link.ontology_id
                    if not oid:
                        continue
                    agg = aggregates.setdefault(
                        oid,
                        {
                            "mention_count": 0,
                            "chunk_ids": [],
                            "chunk_id_set": set(),
                            "surface_counts": {},  # surface form -> count, used for fallback label
                            "facet_counts": {},
                            "last_seen": chunk.created_at,
                        },
                    )
                    agg["mention_count"] = int(agg["mention_count"]) + 1
                    chunk_id_set: set[str] = agg["chunk_id_set"]  # type: ignore[assignment]
                    if chunk.chunk_id not in chunk_id_set:
                        chunk_id_set.add(chunk.chunk_id)
                        sample: list[str] = agg["chunk_ids"]  # type: ignore[assignment]
                        if len(sample) < sample_cap:
                            sample.append(chunk.chunk_id)
                    surface_counts: dict[str, int] = agg["surface_counts"]  # type: ignore[assignment]
                    surface = link.mention.text
                    surface_counts[surface] = surface_counts.get(surface, 0) + 1
                    facet_counts: dict[str, int] = agg["facet_counts"]  # type: ignore[assignment]
                    facet = _facet_hint_for_entity_type(link.mention.entity_type)
                    if facet is not None:
                        facet_counts[facet] = facet_counts.get(facet, 0) + 1
                    last: datetime = agg["last_seen"]  # type: ignore[assignment]
                    if chunk.created_at > last:
                        agg["last_seen"] = chunk.created_at

                    edges = chunk_edges.setdefault(oid, {})
                    # Highest-confidence link per (chunk, entity) wins — same
                    # entity may appear multiple times in one chunk.
                    prev = edges.get(chunk.chunk_id)
                    if prev is None or float(link.confidence) > prev[0]:
                        edges[chunk.chunk_id] = (float(link.confidence), link.method)

        entities: list[Entity] = []
        for oid, agg in aggregates.items():
            prefix = oid.split(":", 1)[0] if ":" in oid else ""
            label, synonyms, ancestors = _enrich_from_ontology(ontology, oid, prefix)
            if not label:
                # Fall back to the most-frequent surface form when ontology
                # lookup didn't yield a preferred label (typical for non-FoodOn
                # prefixes when the loader is FoodOn-scoped).
                surface_counts: dict[str, int] = agg["surface_counts"]  # type: ignore[assignment]
                label = max(surface_counts.items(), key=lambda x: x[1])[0] if surface_counts else oid
            facet_counts: dict[str, int] = agg["facet_counts"]  # type: ignore[assignment]
            facet_hint = (
                max(facet_counts.items(), key=lambda x: x[1])[0] if facet_counts else None
            )
            entities.append(
                Entity(
                    ontology_id=oid,
                    prefix=prefix,
                    label=label,
                    synonyms=tuple(synonyms),
                    ancestor_ids=tuple(ancestors),
                    facet_hint=facet_hint,  # type: ignore[arg-type]
                    mention_count=int(agg["mention_count"]),
                    chunk_count=len(agg["chunk_id_set"]),  # type: ignore[arg-type]
                    chunk_ids=tuple(agg["chunk_ids"]),  # type: ignore[arg-type]
                    last_seen=agg["last_seen"],  # type: ignore[arg-type]
                )
            )

        # Persist: entity store first (so search works even if graph write
        # later fails partway through), then Neo4j entity nodes + edges.
        self.entity_store.upsert(entities)
        self.graph_store.upsert_entities(entities)
        for oid, edges in chunk_edges.items():
            links = [(cid, conf, method) for cid, (conf, method) in edges.items()]
            self.graph_store.attach_chunks_to_entity(oid, links)

        meta = make_artifact_meta(
            phase="build_entities",
            config=self.config,
            record_count=len(entities),
        )
        self._log.info(
            "build_entities.done",
            n_chunks_seen=n_chunks_seen,
            n_entities=len(entities),
            n_edges=sum(len(e) for e in chunk_edges.values()),
            artifact_id=meta.artifact_id,
            config_hash=meta.config_hash,
        )
        return meta

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
        """Provision the backing stores declared by the config.

        Calls `chunk_store.init()`, `entity_store.init()`, `graph_store.init()`,
        and `card_store.init()` — all are in the storage protocols and are
        no-ops for the in-memory backends, so this works uniformly regardless
        of where the stores live.
        """
        self.chunk_store.init()
        self.entity_store.init()
        self.graph_store.init()
        self.card_store.init()
        self._log.info(
            "init.done",
            chunk_store=self.config.storage.chunk_store.backend,
            entity_store=type(self.entity_store).__name__,
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
            llm=self.llm,
        )

    def attach(self) -> ArtifactMeta:
        from foodscholar.layer_a import attach as _attach

        return _attach(
            self.chunk_store,
            self.graph_store,
            self.ontology,
            full_config=self.config,
        )

    def semantic_consolidate(
        self, *, facet: str = "foods", dry_run: bool = True
    ) -> ConsolidationArtifact:
        """Embed shelves, find near-duplicate pairs, and merge via an LLM judge.

        Runs *after* `fs.attach()` so the judge can ground each decision on
        real sample chunks (it reads `chunk.shelf_ids`, written by attach).

        - `dry_run=True` (default): read-only. Returns the
          `ConsolidationArtifact` — candidates, decisions, and what the
          pre-LLM filters dropped — for inspection. Nothing is persisted.
        - `dry_run=False`: applies confirmed merges (above
          `auto_merge_confidence`), re-persists the shelf set, and re-runs
          `fs.attach()` so the merged-away shelves' chunks re-home onto the
          surviving canonical shelf.

        Set `layer_a.semantic_consolidation.judge_enabled=False` for a
        zero-cost candidate preview (no LLM calls).
        """
        from foodscholar.layer_a.semantic_consolidation import consolidate

        cfg = self.config.layer_a.semantic_consolidation
        shelves = self.graph_store.list_shelves()
        facet_shelves = [s for s in shelves if s.facet == facet]

        merged, artifact = consolidate(
            facet_shelves,
            self.chunk_store,
            self.ontology,
            self.embedder,
            self.llm,
            cfg,
            self.config_hash,
            facet=facet,
            dry_run=dry_run,
        )

        if not dry_run and len(merged) != len(facet_shelves):
            keep = [s for s in shelves if s.facet != facet]
            all_shelves = sorted(
                keep + merged,
                key=lambda s: (s.facet, s.depth, s.label.lower(), s.shelf_id),
            )
            self.graph_store.clear_layer_a()
            self.graph_store.upsert_shelves(all_shelves)
            # Repair denormalized shelf_ids: merged losers' chunks re-home onto
            # the canonical via attach's see_also routing.
            self.attach()

        self._log.info(
            "semantic_consolidate.done",
            facet=facet,
            dry_run=dry_run,
            applied_groups=len(artifact.applied_groups),
            shelves_removed=artifact.shelves_removed,
            config_hash=self.config_hash,
        )
        return artifact

    def audit(self) -> AuditReport:
        """Run cross-store invariant checks and return a structured report.

        Read-only — never writes. Returns an `AuditReport` with five sections:
        inventory, coverage, cross-store consistency, attach integrity, and
        structural sanity. `report.passed` is True iff zero critical checks
        failed; `report.critical_failures` lists the broken invariants.
        Print `report` directly for a human-readable summary.
        """
        from foodscholar.evaluation.audit import audit as _audit

        return _audit(
            self.chunk_store,
            self.graph_store,
            config_hash=self.config_hash,
        )

    def quality_report(
        self,
        *,
        facet: str = "foods",
        top_n: int = 20,
        sample_size: int = 20,
        canonical_terms: tuple[str, ...] | None = None,
        seed: int = 0,
    ) -> QualityReport:
        """Produce a domain-expert quality report for one facet of Layer A.

        Pairs with `fs.audit()` but answers a different question: not "is the
        graph correctly built" (invariants) but "is the graph good"
        (semantic / coverage). Read-only; the output is a Pydantic
        `QualityReport` whose `__str__` is Markdown for direct notebook
        viewing.

        Sections:

        1. Top shelves at a glance — table sorted by chunk_count with
           direct/lifted split + 3 example chunk snippets per shelf.
        2. Hierarchy walkthrough — parent chains + sample descendants for
           the top 5 shelves, so the expert can sanity-check navigation.
        3. Suspicious shelves — conservative heuristic flags (EFSA-style
           code prefixes, "datum" labels, collapse-misses, zero-chunk
           survivors).
        4. Canonical vocabulary check — checklist of foods/nutrients/
           conditions an expert would expect, with status per term.
           Override the default list via `canonical_terms=`.
        5. Random chunk sample — `sample_size` randomly-chosen attached
           chunks with their text + attached shelf labels, formatted for
           hand audit per BRIEF §17. `seed` for reproducibility.
        """
        from foodscholar.evaluation.quality import (
            _DEFAULT_CANONICAL_TERMS,
        )
        from foodscholar.evaluation.quality import (
            quality_report as _quality_report,
        )

        return _quality_report(
            self.chunk_store,
            self.graph_store,
            self.ontology,
            config_hash=self.config_hash,
            facet=facet,  # type: ignore[arg-type]
            top_n=top_n,
            sample_size=sample_size,
            canonical_terms=canonical_terms or _DEFAULT_CANONICAL_TERMS,
            seed=seed,
        )

    def build_layer_b(
        self,
        *,
        facet: str = "foods",
        dry_run: bool = False,
    ):
        """Build Layer B themes for `facet`.

        For each shelf with `≥ cfg.layer_b.min_chunks_per_shelf` attached
        chunks whose embedded-fraction clears `cfg.layer_b.min_embedded_fraction`,
        runs the dual-pass pipeline (similarity + relatedness), merges
        candidates greedily, labels themes (c-TF-IDF + LLM polish when
        `cfg.layer_b.labeling.strategy == "llm"`), picks a per-pass-aware
        primary chunk, and persists:
          - `(:Theme)` nodes + `(:Shelf)-[:IN_SHELF...]->(:Theme)` edges
          - `(:Chunk)-[:THEME_OF {primary, weight}]->(:Theme)` edges
          - ES `theme_ids` denorm via `bulk_set_theme_ids` (preserves
            `shelf_ids`)

        Skips the synthetic facet root (`facet:foods` etc.) — that's the
        iteration-8 unclassified bucket.

        `dry_run=True` runs the full pipeline but skips all writes. Useful
        for `n_themes` estimates and audit-decision inspection without
        modifying the stores.

        Returns a `LayerBArtifact` summarizing the run (themed/skipped
        shelf counts, total themes, per-pass distribution, leiden seed,
        timestamps).

        See `layer_b_construction_brief.md` for the full architecture.
        """
        from foodscholar.layer_b.builder import build_layer_b as _build_layer_b

        return _build_layer_b(self, facet=facet, dry_run=dry_run)

    def build_quality_report(self, *, facet: str = "foods"):
        """Read-only WARN-level quality report for Layer B of `facet`.

        Pairs with `fs.audit()` (CRITICAL invariants) but answers "is the build
        *good* / well-tuned" rather than "is it correct". Reads shelves, themes,
        and attachments; mutates nothing. Returns a `LayerBQualityReport` whose
        `__str__` is Markdown for notebook viewing — structural stats (shelves,
        depth, fanout, support ratios), theme stats (coverage, source mix,
        duplicate/tiny/leakage counts), and a list of `LayerBWarning`s.

        Warning thresholds come from `cfg.layer_b.audit`. See
        `layer_b/quality.py` for the full metric + warning list.
        """
        from foodscholar.layer_b.quality import build_quality_report as _q

        return _q(self.chunk_store, self.graph_store, self.config.layer_b, facet=facet)

    def sweep_layer_b(
        self,
        *,
        facet: str = "foods",
        grid: dict | None = None,
    ):
        """Non-mutating tuning sweep over a grid of Layer B configs.

        Runs each config combination as a `dry_run` build with cheap keyword
        labels and `per_shelf` Pass 1, scores the resulting quality metrics, and
        returns a ranked `SweepResult` (best first). Nothing is persisted — apply
        the winning config (`result.best`) yourself and rebuild.

        `grid` maps dotted `layer_b` config paths (e.g.
        `"leiden.min_community_size"`) to candidate values; defaults to the full
        160-config Cartesian product (`sweep.DEFAULT_GRID`). Scoring weights are
        fixed and documented in `layer_b/sweep.py`.
        """
        from foodscholar.layer_b.sweep import sweep_layer_b as _sweep

        return _sweep(self, facet=facet, grid=grid)

    def build_layer_c(self, *, facet: str = "foods", dry_run: bool = False):
        """Build Layer C — one summary Card per Layer B theme of `facet`.

        Each theme's member chunks are compressed by a cheap extractive method
        (Stage 1, map-reduce when large), then refined by the LLM into a Card
        (Stage 2). `dry_run=True` runs both stages but skips persistence.
        Returns a `LayerCReport`.
        """
        from foodscholar.layer_c.builder import build_layer_c as _build_layer_c

        return _build_layer_c(self, facet=facet, dry_run=dry_run)

    def benchmark_layer_c(
        self,
        *,
        facet: str = "foods",
        themes: int = 5,
        out: str | None = None,
    ):
        """Read-only benchmark of all extractive methods over the largest
        `themes` themes of `facet`. Writes per-method JSON metrics; returns the
        results keyed by theme id. No LLM, no persistence."""
        from foodscholar.layer_c.benchmark import benchmark_facet as _bench

        return _bench(self, facet=facet, themes=themes, out=out)

    def export_graphml(
        self,
        output: str | Path,
        *,
        facet: str | None = "foods",
    ) -> Path:
        """Export the Layer A/B/C graph (shelves + themes + cards) to GraphML.

        Typed nodes (`shelf` / `theme` / `card`) with their attributes and edges
        (`parent_of` / `has_theme` / `has_card`), readable by Gephi, Cytoscape,
        yEd, etc. `facet=None` exports every facet. Returns the output `Path`.
        """
        from foodscholar.io.graphml import export_graphml as _export

        return _export(self.graph, output, facet=facet)

    def search_cards(self, text: str, *, k: int = 10) -> list[Card]:
        """Vector-search Layer C cards by `text`. Embeds the query with the
        chunk embedder, runs kNN over the card store, and returns the matching
        `Card`s nearest-first. (A thin retrieval helper; full `query()` with
        answer synthesis is still deferred.)"""
        query_vec = self.embedder.embed([text])[0]
        hits = self.card_store.knn_search_cards(query_vec, k=k)
        cards = {c.card_id: c for c in self.card_store.get_many([cid for cid, _ in hits])}
        return [cards[cid] for cid, _ in hits if cid in cards]

    def build(self) -> None:
        self.annotate()
        self.build_layer_a()
        self.attach()
        self.build_layer_b()
        self.build_layer_c()

    def query(self, text: str) -> Answer:
        raise _deferred("query")

    # ------------------------------------------------------------------ helpers

    def _resolve_ignored_source_types(
        self, override: set[str] | None
    ) -> frozenset[str]:
        """Per-call override wins; otherwise read `cfg.corpus.ignore_source_types`."""
        if override is not None:
            return frozenset(override)
        return frozenset(self.config.corpus.ignore_source_types)

    @staticmethod
    def _count_skipped_chunks(path: str | Path, skip: frozenset[str]) -> int:
        """Count chunks at `path` whose `source_type` is in `skip`. One extra
        scan over the corpus — cheap for CSV/JSONL and only used to log the
        skip count when `ignore_source_types` is active.
        """
        if not skip:
            return 0
        from foodscholar.corpus import iter_chunks

        return sum(1 for c in iter_chunks(path) if c.source_type in skip)


def _minimal_memory_config() -> FoodScholarConfig:
    """Smallest valid config for `FoodScholar.in_memory()` with no args."""
    return FoodScholarConfig.model_validate(
        {
            "corpus": {"chunks_path": "data/chunks.parquet"},
            "storage": {
                "chunk_store": {"backend": "memory"},
                "graph_store": {"backend": "memory"},
                "card_store": {"backend": "memory"},
            },
        }
    )


# ---------------------------------------------------------------- entity helpers

# Canonical mapping lives in foodscholar.layer_a.facet — `build_entities` calls
# `_facet_hint_for_entity_type` below.
from foodscholar.layer_a.facet import (  # noqa: E402
    facet_for_entity_type as _facet_hint_for_entity_type,
)


def _enrich_from_ontology(
    ontology: Any, ontology_id: str, prefix: str
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Look up label / synonyms / ancestors via `fs.ontology` if available.

    Returns ``("", (), ())`` when no enrichment is possible (no ontology
    attached, or the id is not in the loaded ontology — typical for non-FOODON
    prefixes when the loader is FoodOn-scoped).
    """
    if ontology is None or prefix != "FOODON":
        return "", (), ()
    term = ontology.get(ontology_id) if hasattr(ontology, "get") else None
    if term is None:
        return "", (), ()
    return term.label, tuple(term.synonyms), tuple(term.ancestor_ids)


# ---------------------------------------------------------------- entity view


class _EntityView:
    """Fluent read surface over `fs.entity_store`. Exposed as `fs.entities`.

    Read-only: writes go through `fs.build_entities()`, which mirrors them to
    both the entity store and the graph store atomically per entity.
    """

    def __init__(self, fs: FoodScholar) -> None:
        self._fs = fs

    def __len__(self) -> int:
        return len(self._fs.entity_store.scan())

    def list(self, *, prefix: str | None = None, k: int = 100) -> list:
        """Top-`k` entities. Sorted by `chunk_count` desc when `prefix` is set;
        unsorted otherwise (use `search` for ranked queries).
        """
        if prefix is not None:
            return self._fs.entity_store.list_by_prefix(prefix, k=k)
        return self._fs.entity_store.scan()[:k]

    def get(self, ontology_id: str):  # type: ignore[no-untyped-def]
        return self._fs.entity_store.get(ontology_id)

    def search(self, query: str, *, prefix: str | None = None, k: int = 10) -> list:
        """Lexical search over label + synonyms. `prefix` filters to one OBO source."""
        return self._fs.entity_store.search(query, prefix=prefix, k=k)

    def chunks_for(self, ontology_id: str, *, k: int = 50) -> list:
        """Return chunks that mention `ontology_id`.

        Uses the chunk store's `foodon_ids` `terms`-filter for FoodOn ids
        (fast). For other prefixes it falls back to walking the entity's
        inline `chunk_ids` sample (capped at 50 by default).
        """
        if ontology_id.startswith("FOODON:"):
            # The chunk store exposes its index name via `.index` (Elastic) or
            # nothing (InMemory). Fast path: terms filter on foodon_ids.
            try:
                # ES-specific shortcut — InMemory falls through to the sample.
                if hasattr(self._fs.chunk_store, "_es"):
                    es = self._fs.chunk_store._es  # type: ignore[attr-defined]
                    idx = self._fs.chunk_store.index  # type: ignore[attr-defined]
                    resp = es.search(
                        index=idx,
                        body={
                            "size": k,
                            "query": {"term": {"foodon_ids": ontology_id}},
                        },
                    )
                    from foodscholar.io.chunk import Chunk

                    return [Chunk.model_validate(h["_source"]) for h in resp["hits"]["hits"]]
            except Exception:  # pragma: no cover — defensive fallback
                pass
        ent = self._fs.entity_store.get(ontology_id)
        if ent is None:
            return []
        return self._fs.chunk_store.get_many(list(ent.chunk_ids))[:k]

    def build(self, *, cap_chunk_sample: int | None = None):
        """Convenience: same as `fs.build_entities(...)`. Returns ArtifactMeta."""
        return self._fs.build_entities(cap_chunk_sample=cap_chunk_sample)
