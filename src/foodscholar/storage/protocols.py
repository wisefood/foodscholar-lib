from collections.abc import Iterable
from typing import Literal, Protocol, runtime_checkable

from foodscholar.io.chunk import Chunk, ChunkId, EntityLink, Mention
from foodscholar.io.graph import Card, Shelf, ShelfId, Theme, ThemeId


@runtime_checkable
class ChunkStore(Protocol):
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
    def scan(self) -> list[Chunk]: ...


@runtime_checkable
class GraphStore(Protocol):
    def upsert_shelves(self, shelves: list[Shelf]) -> None: ...
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
