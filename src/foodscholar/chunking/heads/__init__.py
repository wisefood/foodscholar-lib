"""Heads: one per document genre, each turning a source file into IR.

A head is organised by *genre*, not by domain: a textbook and an annual report
are both ``long_document``. Heads register themselves here; the chunker looks
one up from the source type's profile. Built-in: the record head
(:mod:`foodscholar.chunking.heads.record`). Looking up a genre with no head says
which one is missing; an existing IR can always be assembled.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from foodscholar.chunking.config import ChunkingConfig
    from foodscholar.chunking.ir import DocumentIR
    from foodscholar.chunking.registry import SourceRecord


@runtime_checkable
class Head(Protocol):
    genre: str

    def extract(
        self,
        path: Path,
        *,
        record: SourceRecord | None,
        config: ChunkingConfig,
        source_type: str,
    ) -> list[DocumentIR]:
        """One source file → one or more documents (a JSON of abstracts holds many)."""
        ...


_FACTORIES: dict[str, Callable[[], Head]] = {}
_KNOWN = {
    "record": "the record head (journal abstracts)",
    "long_document": "the long-document head (textbooks)",
    "leaflet": "the leaflet head (dietary guides)",
}


def register_head(genre: str, factory: Callable[[], Head]) -> None:
    """Register (or replace) the head for a genre."""
    _FACTORIES[genre] = factory


def get_head(genre: str) -> Head:
    if genre in _FACTORIES:
        return _FACTORIES[genre]()
    name = _KNOWN.get(genre)
    hint = f" {name[0].upper()}{name[1:]} is not implemented yet." if name else ""
    raise NotImplementedError(
        f"no head is registered for genre {genre!r}.{hint} Until then, assemble an "
        f"existing IR with Chunker.assemble() or `foodscholar chunk run --ir`."
    )


def registered_genres() -> list[str]:
    return sorted(_FACTORIES)


def _register_builtin() -> None:
    from foodscholar.chunking.heads.record import RecordHead

    register_head("record", RecordHead)


_register_builtin()

