"""Canonical smoke test (BRIEF §11).

Walks the entire pipeline against in-memory stores using mocked NER,
embedder, and LLM. Not a substitute for phase-level unit tests once the
real implementations land, but proves the contracts hook together.
"""

from __future__ import annotations

from foodscholar.io.chunk import EntityLink, Mention
from foodscholar.io.graph import Card, Shelf, Theme
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore


def test_smoke_end_to_end(mini_chunks, mock_embedder, mock_llm):  # type: ignore[no-untyped-def]
    chunk_store = InMemoryChunkStore()
    graph_store = InMemoryGraphStore()

    # 1. corpus
    chunk_store.upsert(mini_chunks)
    assert chunk_store.get("c1") is not None

    # 2. annotate (mock NER + mock embedder)
    mention = Mention(
        text="olive oil", start=29, end=38, score=0.95, ner_model_version="mock-ner-v0"
    )
    link = EntityLink(
        mention=mention,
        ontology_id="FOODON:03309927",
        confidence=0.93,
        method="lexical_exact",
        linker_version="mock-linker-v0",
    )
    c1 = chunk_store.get("c1")
    assert c1 is not None
    annotated = c1.model_copy(
        update={
            "mentions": [mention],
            "entity_links": [link],
            "foodon_ids": ["FOODON:03309927"],
            "enrichment_version": "smoke-v1",
        }
    )
    chunk_store.upsert([annotated])
    assert chunk_store.get("c1").foodon_ids == ["FOODON:03309927"]

    # 3. Layer A — hand-curated mini ontology projected to 3 shelves
    shelves = [
        Shelf(shelf_id="s-foods", label="Foods", facet="foods", depth=0),
        Shelf(
            shelf_id="s-med",
            label="Mediterranean diet",
            facet="dietary_patterns",
            depth=1,
            parent_shelf_id="s-foods",
        ),
        Shelf(
            shelf_id="s-allergy",
            label="Food allergies",
            facet="allergies",
            depth=1,
        ),
    ]
    graph_store.upsert_shelves(shelves)

    # 4. attach chunks to shelves
    attachments = {
        "s-med": ["c1", "c2", "c5"],
        "s-allergy": ["c3"],
        "s-foods": ["c4"],
    }
    for shelf_id, cids in attachments.items():
        graph_store.attach_chunks_to_shelf(shelf_id, cids)
        for cid in cids:
            existing = chunk_store.get(cid)
            assert existing is not None
            new_shelf_ids = sorted(set(existing.shelf_ids) | {shelf_id})
            chunk_store.update_attachments(
                cid, shelf_ids=new_shelf_ids, theme_ids=list(existing.theme_ids)
            )

    # 5. Layer B — discover two themes attached to s-med
    themes = [
        Theme(
            theme_id="t-olive",
            label="Olive oil cardiovascular benefits",
            shelf_ids=["s-med"],
            discovered_by="leiden",
            discovery_version="smoke-v1",
        ),
        Theme(
            theme_id="t-plant",
            label="Plant-based metabolic markers",
            shelf_ids=["s-med"],
            discovered_by="leiden",
            discovery_version="smoke-v1",
        ),
    ]
    graph_store.upsert_themes(themes)
    graph_store.attach_chunks_to_theme("t-olive", ["c1"])
    graph_store.attach_chunks_to_theme("t-plant", ["c5"])
    # denormalize theme_ids onto chunks
    for theme_id, cids in [("t-olive", ["c1"]), ("t-plant", ["c5"])]:
        for cid in cids:
            existing = chunk_store.get(cid)
            assert existing is not None
            chunk_store.update_attachments(
                cid,
                shelf_ids=list(existing.shelf_ids),
                theme_ids=sorted(set(existing.theme_ids) | {theme_id}),
            )

    # 6. Layer C — two mock cards
    cards = [
        Card(
            card_id="card-s-med",
            target_id="s-med",
            target_type="shelf",
            title="Mediterranean diet",
            summary=mock_llm.generate("summarize s-med"),
            evidence_quality="high",
            cited_chunk_ids=["c1", "c2"],
            llm_model=mock_llm.model_id,
            prompt_version="v1",
        ),
        Card(
            card_id="card-t-olive",
            target_id="t-olive",
            target_type="theme",
            title="Olive oil and cardiovascular health",
            summary=mock_llm.generate("summarize t-olive"),
            evidence_quality="medium",
            cited_chunk_ids=["c1"],
            llm_model=mock_llm.model_id,
            prompt_version="v1",
        ),
    ]
    graph_store.upsert_cards(cards)

    # 7. end-to-end query: ask about cardiovascular and assert we retrieve cited chunks
    hits = chunk_store.search(
        "olive oil cardiovascular", theme_ids=["t-olive"], k=5
    )
    assert any(h.chunk_id == "c1" for h in hits)

    card = graph_store.get_card("t-olive", "theme")
    assert card is not None
    # every claim in a card must trace to a cited chunk — assert the citations exist
    for cid in card.cited_chunk_ids:
        chunk = chunk_store.get(cid)
        assert chunk is not None, f"missing cited chunk {cid}"
