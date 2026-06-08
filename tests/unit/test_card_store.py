"""Card embeddings: Card model fields + InMemoryCardStore (upsert/get/knn)."""

from __future__ import annotations

from foodscholar.io.graph import Card
from foodscholar.storage.memory import InMemoryCardStore


def _card(cid: str, *, embedding=None) -> Card:
    return Card(
        card_id=cid, target_id=cid.replace("card:theme:", ""), target_type="theme",
        title="t", summary="s", evidence_quality="high",
        cited_chunk_ids=["c1"], llm_model="m", prompt_version="v1",
        embedding=embedding, embedding_model=("test" if embedding else None),
    )


def test_card_embedding_defaults_none() -> None:
    c = _card("card:theme:t1")
    assert c.embedding is None
    assert c.embedding_model is None


def test_card_carries_embedding() -> None:
    c = _card("card:theme:t1", embedding=[0.1, 0.2, 0.3])
    assert c.embedding == [0.1, 0.2, 0.3]
    assert c.embedding_model == "test"


def test_in_memory_card_store_upsert_get() -> None:
    cs = InMemoryCardStore()
    cs.init()
    cs.upsert([_card("card:theme:t1"), _card("card:theme:t2")])
    got = cs.get_many(["card:theme:t1", "card:theme:t2", "missing"])
    assert {c.card_id for c in got} == {"card:theme:t1", "card:theme:t2"}


def test_in_memory_card_store_knn_orders_by_similarity() -> None:
    cs = InMemoryCardStore()
    cs.upsert([
        _card("card:theme:near", embedding=[1.0, 0.0, 0.0]),
        _card("card:theme:mid", embedding=[0.7, 0.7, 0.0]),
        _card("card:theme:far", embedding=[0.0, 0.0, 1.0]),
        _card("card:theme:novec"),  # no embedding — excluded from knn
    ])
    hits = cs.knn_search_cards([1.0, 0.0, 0.0], k=3)
    ids = [cid for cid, _ in hits]
    assert ids[0] == "card:theme:near"
    assert "card:theme:novec" not in ids
    # scores are descending
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_in_memory_card_store_knn_exclude() -> None:
    cs = InMemoryCardStore()
    cs.upsert([
        _card("card:theme:a", embedding=[1.0, 0.0]),
        _card("card:theme:b", embedding=[0.9, 0.1]),
    ])
    hits = cs.knn_search_cards([1.0, 0.0], k=5, exclude_ids=["card:theme:a"])
    assert [cid for cid, _ in hits] == ["card:theme:b"]
