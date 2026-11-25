"""Neighbour expansion over any chunk store: the replacement for overlap.

Given a chunk, return the chunks around it, in reading order:

- ``window``: the chunk and up to ``n`` neighbours either side;
- ``section``: every chunk of the chunk's parent section;
- ``page``: every chunk touching a page (the chunk's first page by default);
- ``figure``: every chunk referencing a figure or table asset.

It works on the adjacency keys :mod:`foodscholar.chunking` writes into
``source_metadata`` (``ordinal``, ``prev_chunk_id``/``next_chunk_id``,
``parent_id``, ``pages``, ``figure_ids``/``table_ids``), through one store
primitive, ``find_by_metadata``. Chunks from the legacy pipeline carry none of
these keys; ``window`` then falls back to nothing but the chunk itself, and
``page`` to the first-page key the legacy pipeline wrote.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from foodscholar.io.chunk import Chunk, ChunkId

ExpandMode = Literal["window", "section", "page", "figure"]


class _Store(Protocol):
    def get(self, chunk_id: ChunkId) -> Chunk | None: ...
    def get_many(self, chunk_ids: list[ChunkId]) -> list[Chunk]: ...
    def find_by_metadata(
        self, key: str, values: list[Any], *, source_doc_id: str | None = None, limit: int = 1000
    ) -> list[Chunk]: ...


def _ordinal(chunk: Chunk) -> int:
    value = chunk.source_metadata.get("ordinal")
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _ids(value: object) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


def _ordered(chunks: list[Chunk]) -> list[Chunk]:
    unique = {c.chunk_id: c for c in chunks}
    return sorted(unique.values(), key=lambda c: (_ordinal(c), c.chunk_id))


def expand(
    store: _Store,
    chunk_id: ChunkId,
    mode: ExpandMode = "window",
    n: int = 1,
    *,
    page: int | None = None,
    asset_id: str | None = None,
    limit: int = 1000,
) -> list[Chunk]:
    """Chunks around `chunk_id`, in reading order. Empty if the chunk is unknown."""
    chunk = store.get(chunk_id)
    if chunk is None:
        return []
    meta = chunk.source_metadata
    doc = chunk.source_doc_id
    if mode == "window":
        if n < 0:
            raise ValueError("n must be non-negative")
        if meta.get("ordinal") is not None:
            o = _ordinal(chunk)
            wanted = [o + d for d in range(-n, n + 1) if o + d >= 0]
            return _ordered(store.find_by_metadata("ordinal", wanted, source_doc_id=doc, limit=limit))
        return _follow_links(store, chunk, n)
    if mode == "section":
        parent = meta.get("parent_id")
        if not parent:
            return [chunk]
        return _ordered(store.find_by_metadata("parent_id", [parent], source_doc_id=doc, limit=limit))
    if mode == "page":
        target = page if page is not None else meta.get("page_start", meta.get("page_number"))
        if target is None:
            return [chunk]
        key = "pages" if "pages" in meta else "page_number"
        return _ordered(store.find_by_metadata(key, [target], source_doc_id=doc, limit=limit))
    if mode == "figure":
        ids = [asset_id] if asset_id else [*_ids(meta.get("figure_ids")), *_ids(meta.get("table_ids"))]
        if not ids:
            return [chunk]
        found = store.find_by_metadata("figure_ids", ids, source_doc_id=doc, limit=limit)
        found += store.find_by_metadata("table_ids", ids, source_doc_id=doc, limit=limit)
        return _ordered(found)
    raise ValueError(f"unknown expand mode {mode!r} (want window, section, page or figure)")


def _follow_links(store: _Store, chunk: Chunk, n: int) -> list[Chunk]:
    out = [chunk]
    for key in ("prev_chunk_id", "next_chunk_id"):
        cur = chunk
        for _ in range(n):
            nxt = cur.source_metadata.get(key)
            if not nxt:
                break
            got = store.get(str(nxt))
            if got is None:
                break
            out.append(got)
            cur = got
    return _ordered(out)


def metadata_matches(chunk: Chunk, key: str, values: set[str], source_doc_id: str | None) -> bool:
    """The in-memory semantics of ``find_by_metadata``: string equality, lists match any element."""
    if source_doc_id is not None and chunk.source_doc_id != source_doc_id:
        return False
    value = chunk.source_metadata.get(key)
    if value is None:
        return False
    if isinstance(value, list):
        return any(str(v) in values for v in value)
    return str(value) in values
