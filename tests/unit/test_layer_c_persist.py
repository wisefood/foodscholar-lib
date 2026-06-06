"""Layer C persistence: cards -> graph_store.upsert_cards."""

from __future__ import annotations

from foodscholar.io.graph import Card
from foodscholar.layer_c.persist import persist_cards
from foodscholar.storage.memory import InMemoryGraphStore


def _card(tid: str) -> Card:
    return Card(
        card_id=f"card:theme:{tid}", target_id=tid, target_type="theme",
        title="t", summary="s", evidence_quality="high",
        cited_chunk_ids=["c1"], llm_model="m", prompt_version="v1",
    )


def test_persist_cards_writes_to_store() -> None:
    gs = InMemoryGraphStore()
    persist_cards([_card("t1"), _card("t2")], gs)
    assert gs.get_card("t1", "theme") is not None
    assert gs.get_card("t2", "theme") is not None


def test_persist_empty_is_noop() -> None:
    gs = InMemoryGraphStore()
    persist_cards([], gs)  # must not raise
    assert gs.get_card("t1", "theme") is None
