"""Fluent read/write surface over the chunk + graph stores.

`fs.graph` returns a `GraphView`. From there:

    fs.graph.shelves()                       -> list[ShelfHandle]
    fs.graph.shelf("s-med").themes()         -> list[ThemeHandle]
    fs.graph.shelf("s-med").chunks()         -> list[Chunk]
    fs.graph.shelf("s-med").card()           -> CardHandle | None
    fs.graph.theme("t-olive").chunks()
    fs.graph.search("olive oil", shelf="s-med", k=5)

Mutation lives on the same object so users don't hunt for a second class:

    fs.graph.add_shelf(shelf_id="s-med", label="Mediterranean diet",
                       facet="dietary_patterns", depth=1)
    fs.graph.attach_chunks(shelf="s-med", chunks=["c1","c2"])

Handles wrap their Pydantic model rather than subclass it — the model stays
serializable and frozen-friendly; navigation methods live on the handle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from foodscholar.io.chunk import Chunk, ChunkId
from foodscholar.io.graph import (
    Card,
    CardId,
    EvidenceQuality,
    Facet,
    Shelf,
    ShelfId,
    Theme,
    ThemeId,
)

if TYPE_CHECKING:
    from foodscholar.storage.protocols import ChunkStore, GraphStore


class ShelfHandle:
    """Read-side handle around a `Shelf` plus navigation methods."""

    __slots__ = ("_shelf", "_view")

    def __init__(self, view: GraphView, shelf: Shelf) -> None:
        self._view = view
        self._shelf = shelf

    # passthrough attributes — keeps `handle.label` etc. working
    @property
    def shelf_id(self) -> ShelfId:
        return self._shelf.shelf_id

    @property
    def label(self) -> str:
        return self._shelf.label

    @property
    def facet(self) -> Facet:
        return self._shelf.facet

    @property
    def depth(self) -> int:
        return self._shelf.depth

    @property
    def foodon_id(self) -> str | None:
        return self._shelf.foodon_id

    @property
    def parent_shelf_id(self) -> ShelfId | None:
        return self._shelf.parent_shelf_id

    @property
    def chunk_count(self) -> int:
        return self._shelf.chunk_count

    @property
    def model(self) -> Shelf:
        """Return the underlying Pydantic model (for serialization)."""
        return self._shelf

    # navigation
    def parent(self) -> ShelfHandle | None:
        if self._shelf.parent_shelf_id is None:
            return None
        return self._view.shelf(self._shelf.parent_shelf_id)

    def children(self) -> list[ShelfHandle]:
        return [
            self._view._wrap_shelf(s)
            for s in self._view._all_shelves()
            if s.parent_shelf_id == self._shelf.shelf_id
        ]

    def neighbors(self, hops: int = 1) -> list[ShelfHandle]:
        return [
            self._view._wrap_shelf(self._view._get_shelf(sid))
            for sid in self._view._graph.get_neighbors(self._shelf.shelf_id, hops=hops)
            if self._view._get_shelf(sid) is not None
        ]

    def themes(self) -> list[ThemeHandle]:
        return [
            self._view._wrap_theme(t)
            for t in self._view._graph.get_themes_for_shelf(self._shelf.shelf_id)
        ]

    def chunks(self) -> list[Chunk]:
        return [
            c
            for c in self._view._all_chunks()
            if self._shelf.shelf_id in c.shelf_ids
        ]

    def card(self) -> CardHandle | None:
        c = self._view._graph.get_card(self._shelf.shelf_id, "shelf")
        return self._view._wrap_card(c) if c else None

    def __repr__(self) -> str:
        return f"ShelfHandle({self._shelf.shelf_id!r}, label={self._shelf.label!r})"


class ThemeHandle:
    __slots__ = ("_theme", "_view")

    def __init__(self, view: GraphView, theme: Theme) -> None:
        self._view = view
        self._theme = theme

    @property
    def theme_id(self) -> ThemeId:
        return self._theme.theme_id

    @property
    def label(self) -> str:
        return self._theme.label

    @property
    def shelf_ids(self) -> list[ShelfId]:
        return self._theme.shelf_ids

    @property
    def chunk_count(self) -> int:
        return self._theme.chunk_count

    @property
    def discovered_by(self) -> str:
        return self._theme.discovered_by

    @property
    def model(self) -> Theme:
        return self._theme

    def shelves(self) -> list[ShelfHandle]:
        return [self._view.shelf(sid) for sid in self._theme.shelf_ids if self._view._get_shelf(sid)]  # type: ignore[misc]

    def chunks(self) -> list[Chunk]:
        cids = self._view._graph.get_chunks_for_theme(self._theme.theme_id)
        return self._view._chunks.get_many(list(cids))

    def card(self) -> CardHandle | None:
        c = self._view._graph.get_card(self._theme.theme_id, "theme")
        return self._view._wrap_card(c) if c else None

    def __repr__(self) -> str:
        return f"ThemeHandle({self._theme.theme_id!r}, label={self._theme.label!r})"


class CardHandle:
    __slots__ = ("_card", "_view")

    def __init__(self, view: GraphView, card: Card) -> None:
        self._view = view
        self._card = card

    @property
    def card_id(self) -> CardId:
        return self._card.card_id

    @property
    def target_id(self) -> str:
        return self._card.target_id

    @property
    def target_type(self) -> Literal["shelf", "theme"]:
        return self._card.target_type

    @property
    def title(self) -> str:
        return self._card.title

    @property
    def summary(self) -> str:
        return self._card.summary

    @property
    def tip(self) -> str | None:
        return self._card.tip

    @property
    def evidence_quality(self) -> EvidenceQuality:
        return self._card.evidence_quality

    @property
    def cited_chunk_ids(self) -> list[ChunkId]:
        return self._card.cited_chunk_ids

    @property
    def model(self) -> Card:
        return self._card

    def target(self) -> ShelfHandle | ThemeHandle | None:
        if self._card.target_type == "shelf":
            return self._view.shelf(self._card.target_id)
        return self._view.theme(self._card.target_id)

    def cited_chunks(self) -> list[Chunk]:
        return self._view._chunks.get_many(self._card.cited_chunk_ids)

    def __repr__(self) -> str:
        return f"CardHandle({self._card.card_id!r}, {self._card.target_type}={self._card.target_id!r})"


class GraphView:
    """Fluent read/write surface around the chunk + graph stores.

    Exposed as `fs.graph` on the facade. Holds no state of its own — every
    call routes through the underlying stores, so the view stays in lockstep
    with mutations made directly on the stores or by the phase modules.
    """

    def __init__(self, chunk_store: ChunkStore, graph_store: GraphStore) -> None:
        self._chunks = chunk_store
        self._graph = graph_store

    # ---------------------------------------------------------------- lookups

    def shelf(self, shelf_id: ShelfId) -> ShelfHandle | None:
        s = self._get_shelf(shelf_id)
        return self._wrap_shelf(s) if s else None

    def theme(self, theme_id: ThemeId) -> ThemeHandle | None:
        t = self._get_theme(theme_id)
        return self._wrap_theme(t) if t else None

    def card(
        self, target_id: str, target_type: Literal["shelf", "theme"]
    ) -> CardHandle | None:
        c = self._graph.get_card(target_id, target_type)
        return self._wrap_card(c) if c else None

    def chunk(self, chunk_id: ChunkId) -> Chunk | None:
        return self._chunks.get(chunk_id)

    # ---------------------------------------------------------------- listings

    def shelves(self, *, facet: Facet | None = None) -> list[ShelfHandle]:
        out = [self._wrap_shelf(s) for s in self._all_shelves()]
        if facet is not None:
            out = [h for h in out if h.facet == facet]
        return out

    def themes(self) -> list[ThemeHandle]:
        return [self._wrap_theme(t) for t in self._all_themes()]

    def roots(self) -> list[ShelfHandle]:
        """Shelves with no parent — the top of every facet."""
        return [self._wrap_shelf(s) for s in self._all_shelves() if s.parent_shelf_id is None]

    # ---------------------------------------------------------------- search

    def search(
        self,
        query: str,
        *,
        shelf: ShelfId | None = None,
        theme: ThemeId | None = None,
        k: int = 10,
    ) -> list[Chunk]:
        """Hybrid retrieval over the chunk store, optionally scoped to a shelf or theme."""
        return self._chunks.search(
            query,
            shelf_ids=[shelf] if shelf else None,
            theme_ids=[theme] if theme else None,
            k=k,
        )

    # ---------------------------------------------------------------- mutation

    def add_shelf(
        self,
        shelf: Shelf | None = None,
        /,
        **kwargs: Any,
    ) -> ShelfHandle:
        """Insert (or upsert) a shelf. Pass a `Shelf` model or keyword args."""
        s = shelf if shelf is not None else Shelf(**kwargs)
        self._graph.upsert_shelves([s])
        return self._wrap_shelf(s)

    def add_shelves(self, shelves: list[Shelf]) -> list[ShelfHandle]:
        self._graph.upsert_shelves(shelves)
        return [self._wrap_shelf(s) for s in shelves]

    def add_theme(
        self,
        theme: Theme | None = None,
        /,
        **kwargs: Any,
    ) -> ThemeHandle:
        t = theme if theme is not None else Theme(**kwargs)
        self._graph.upsert_themes([t])
        return self._wrap_theme(t)

    def add_themes(self, themes: list[Theme]) -> list[ThemeHandle]:
        self._graph.upsert_themes(themes)
        return [self._wrap_theme(t) for t in themes]

    def add_card(
        self,
        card: Card | None = None,
        /,
        **kwargs: Any,
    ) -> CardHandle:
        c = card if card is not None else Card(**kwargs)
        self._graph.upsert_cards([c])
        return self._wrap_card(c)

    def add_cards(self, cards: list[Card]) -> list[CardHandle]:
        self._graph.upsert_cards(cards)
        return [self._wrap_card(c) for c in cards]

    def attach_chunks(
        self,
        chunks: list[ChunkId],
        *,
        shelf: ShelfId | None = None,
        theme: ThemeId | None = None,
    ) -> None:
        """Attach chunks to a shelf or theme.

        Updates both the graph edges AND the denormalized `shelf_ids`/`theme_ids`
        on the chunk records, so the chunk store can filter on them at search
        time. Idempotent — re-running merges sets, doesn't duplicate.
        """
        if shelf is None and theme is None:
            raise ValueError("attach_chunks requires `shelf=` or `theme=`")
        if shelf is not None and theme is not None:
            raise ValueError("attach_chunks takes either `shelf=` or `theme=`, not both")

        if shelf is not None:
            self._graph.attach_chunks_to_shelf(shelf, chunks)
        else:
            assert theme is not None
            self._graph.attach_chunks_to_theme(theme, chunks)

        for cid in chunks:
            c = self._chunks.get(cid)
            if c is None:
                continue
            new_shelves = (
                sorted(set(c.shelf_ids) | {shelf}) if shelf else list(c.shelf_ids)
            )
            new_themes = (
                sorted(set(c.theme_ids) | {theme}) if theme else list(c.theme_ids)
            )
            self._chunks.update_attachments(
                cid, shelf_ids=new_shelves, theme_ids=new_themes
            )

    # ---------------------------------------------------------------- summary

    def summary(self) -> dict[str, int]:
        """Cheap snapshot of graph size — useful for notebook sanity checks."""
        shelves = self._all_shelves()
        themes = self._all_themes()
        return {
            "shelves": len(shelves),
            "themes": len(themes),
            "roots": sum(1 for s in shelves if s.parent_shelf_id is None),
        }

    # ---------------------------------------------------------------- internals

    def _wrap_shelf(self, s: Shelf) -> ShelfHandle:
        return ShelfHandle(self, s)

    def _wrap_theme(self, t: Theme) -> ThemeHandle:
        return ThemeHandle(self, t)

    def _wrap_card(self, c: Card) -> CardHandle:
        return CardHandle(self, c)

    def _get_shelf(self, shelf_id: ShelfId) -> Shelf | None:
        return self._graph.get_shelf(shelf_id)

    def _get_theme(self, theme_id: ThemeId) -> Theme | None:
        for t in self._all_themes():
            if t.theme_id == theme_id:
                return t
        return None

    def _all_shelves(self) -> list[Shelf]:
        return self._graph.list_shelves()

    def _all_themes(self) -> list[Theme]:
        return self._graph.list_themes()

    def _all_chunks(self) -> list[Chunk]:
        return self._chunks.scan()
