from collections.abc import Iterable
from typing import Literal, Protocol, runtime_checkable

from foodscholar.io.chunk import Chunk, ChunkId, EntityLink, Mention
from foodscholar.io.entity import Entity
from foodscholar.io.graph import Card, Shelf, ShelfId, Theme, ThemeId
from foodscholar.io.ontology import OntologyId


@runtime_checkable
class ChunkStore(Protocol):
    def init(self) -> None:
        """Provision the underlying store (index, schema, etc.). Idempotent.

        Local stores (e.g. `InMemoryChunkStore`) implement this as a no-op so
        that `fs.init()` works the same regardless of backend.
        """
        ...

    def upsert(self, chunks: Iterable[Chunk]) -> None: ...
    def get(self, chunk_id: ChunkId) -> Chunk | None: ...
    def get_many(self, chunk_ids: list[ChunkId]) -> list[Chunk]: ...
    def search(
        self,
        query: str,
        theme_ids: list[ThemeId] | None = None,
        shelf_ids: list[ShelfId] | None = None,
        k: int = 10,
        use_vector: bool = True,
        use_bm25: bool = True,
    ) -> list[Chunk]: ...
    def update_attachments(
        self,
        chunk_id: ChunkId,
        shelf_ids: list[ShelfId],
        theme_ids: list[ThemeId],
    ) -> None: ...
    def update_annotations(
        self,
        chunk_id: ChunkId,
        mentions: list[Mention],
        entity_links: list[EntityLink],
        foodon_ids: list[str],
        enrichment_version: str,
    ) -> None: ...
    def update_embedding(
        self,
        chunk_id: ChunkId,
        embedding: list[float],
        embedding_model: str,
    ) -> None:
        """Patch the chunk's `embedding` + `embedding_model` only.

        Used by `fs.embed()` so re-embedding doesn't rewrite the
        `mentions` / `entity_links` / `foodon_ids` payload — a single
        field-scoped update on the remote backends, a `model_copy` on the
        in-memory ones.
        """
        ...
    def scan(self) -> list[Chunk]: ...
    def iter_chunks(self, batch_size: int = 1000) -> Iterable[list[Chunk]]: ...


@runtime_checkable
class GraphStore(Protocol):
    def init(self) -> None:
        """Provision the underlying store (constraints, indexes). Idempotent."""
        ...

    def upsert_shelves(self, shelves: list[Shelf]) -> None: ...
    def clear_layer_a(self) -> None:
        """Delete every (:Shelf) node and any edges attached to it.

        Called by `build_layer_a` before re-upsert so stale shelves from a
        previous projection (with a different blacklist / threshold) don't
        survive as ghosts. Local stores clear their shelf dict; Neo4j runs
        `MATCH (s:Shelf) DETACH DELETE s`, which kills PARENT_OF, HAS_THEME,
        HAS_CHUNK, DESCRIBES edges in one shot.

        Idempotent — a no-op when no shelves exist.
        """
        ...

    def upsert_themes(self, themes: list[Theme]) -> None: ...
    def upsert_cards(self, cards: list[Card]) -> None: ...
    def attach_chunks_to_shelf(self, shelf_id: ShelfId, chunk_ids: list[ChunkId]) -> None: ...
    def attach_chunks_to_theme(self, theme_id: ThemeId, chunk_ids: list[ChunkId]) -> None: ...
    def get_shelf(self, shelf_id: ShelfId) -> Shelf | None: ...
    def get_themes_for_shelf(self, shelf_id: ShelfId) -> list[Theme]: ...
    def get_chunks_for_theme(self, theme_id: ThemeId) -> list[ChunkId]: ...
    def get_neighbors(self, shelf_id: ShelfId, hops: int = 1) -> list[ShelfId]: ...
    def get_card(
        self, target_id: str, target_type: Literal["shelf", "theme"]
    ) -> Card | None: ...
    def list_shelves(self) -> list[Shelf]: ...
    def list_themes(self) -> list[Theme]: ...

    # Entity graph (first-class linked entities)
    def upsert_entities(self, entities: list[Entity]) -> None: ...
    def attach_chunks_to_entity(
        self,
        ontology_id: OntologyId,
        chunk_links: list[tuple[ChunkId, float, str]],
    ) -> None:
        """Wire up `(:Chunk)-[:MENTIONS {confidence, method}]->(:Entity)` edges.

        `chunk_links` is a list of `(chunk_id, confidence, method)` tuples
        carrying the per-mention metadata. Implementations must be idempotent
        — re-running with the same links must not duplicate the edges.
        """
        ...


@runtime_checkable
class EntityStore(Protocol):
    """Dedicated, queryable store for first-class linked entities.

    Local stores implement `init()` as a no-op; the Elastic adapter creates a
    `foodscholar_entities` index alongside the chunk index.
    """

    def init(self) -> None: ...
    def upsert(self, entities: Iterable[Entity]) -> None: ...
    def get(self, ontology_id: OntologyId) -> Entity | None: ...
    def get_many(self, ontology_ids: list[OntologyId]) -> list[Entity]: ...
    def list_by_prefix(self, prefix: str, *, k: int = 100) -> list[Entity]: ...
    def search(self, query: str, *, prefix: str | None = None, k: int = 10) -> list[Entity]: ...
    def scan(self) -> list[Entity]: ...


@runtime_checkable
class Embedder(Protocol):
    model_id: str

    @property
    def dim(self) -> int: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class LLMClient(Protocol):
    model_id: str

    def generate(self, prompt: str, max_tokens: int = 1024) -> str: ...

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, object],
        max_tokens: int = 1024,
    ) -> dict[str, object]:
        """Return a JSON object conforming to `schema` (a JSON-schema dict).

        Uses the provider's native structured-output mode where available.
        Guarantees the result *parses* and matches the schema's shape — it does
        NOT guarantee the values are semantically correct (e.g. an LLM-reported
        character offset may be a valid integer yet wrong). Callers that need
        correct positions must verify them against the source themselves.
        """
        ...


@runtime_checkable
class NER(Protocol):
    """Span-level food entity recognizer.

    Implementations should be deterministic given fixed model weights so
    pipeline reruns produce stable results.
    """

    model_id: str

    def extract(self, text: str) -> list[Mention]: ...


@runtime_checkable
class Linker(Protocol):
    """Maps a `Mention` to a single ontology id, or None if no candidate clears the threshold."""

    linker_id: str

    def link(self, mention: Mention) -> EntityLink | None: ...
