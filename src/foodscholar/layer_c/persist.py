"""Persist Layer C cards. Single write — the Card model carries
`target_id`/`target_type="theme"` so the graph store routes them. Mirrors the
additive contract of `layer_b/persist.py`."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foodscholar.io.graph import Card
    from foodscholar.storage.protocols import GraphStore


def persist_cards(cards: list["Card"], graph_store: "GraphStore") -> None:
    """Upsert theme cards into the graph store. No-op on empty input."""
    if not cards:
        return
    graph_store.upsert_cards(cards)
