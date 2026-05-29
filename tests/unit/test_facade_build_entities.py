"""Tests for `fs.build_entities()` and the `fs.entities` view."""

from __future__ import annotations

from foodscholar import FoodScholar
from foodscholar.io.chunk import Chunk, EntityLink, Mention


def _mention(text: str, *, entity_type: str = "food", start: int = 0) -> Mention:
    return Mention(
        text=text,
        start=start,
        end=start + len(text),
        score=1.0,
        ner_model_version="test",
        entity_type=entity_type,  # type: ignore[arg-type]
    )


def _link(mention: Mention, ontology_id: str, *, confidence: float = 0.9) -> EntityLink:
    return EntityLink(
        mention=mention,
        ontology_id=ontology_id,
        confidence=confidence,
        method="dense",
        linker_version="test",
    )


def _chunk(chunk_id: str, *, mentions: list[Mention], links: list[EntityLink]) -> Chunk:
    foodon_ids: list[str] = []
    for ln in links:
        if ln.ontology_id.startswith("FOODON:") and ln.ontology_id not in foodon_ids:
            foodon_ids.append(ln.ontology_id)
    return Chunk(
        chunk_id=chunk_id,
        text="dummy text",
        source_doc_id="d",
        source_type="abstract",
        section_type="abstract",
        mentions=mentions,
        entity_links=links,
        foodon_ids=foodon_ids,
    )


def test_build_entities_dedupes_across_chunks() -> None:
    fs = FoodScholar.in_memory()

    m_olive = _mention("olive oil", entity_type="food")
    m_olive2 = _mention("Olive Oil", entity_type="food")  # duplicate surface, case-different
    m_iron = _mention("iron", entity_type="micronutrient")

    chunks = [
        _chunk(
            "c1",
            mentions=[m_olive, m_iron],
            links=[_link(m_olive, "FOODON:03309927"), _link(m_iron, "CHEBI:18248")],
        ),
        _chunk(
            "c2",
            mentions=[m_olive2],
            links=[_link(m_olive2, "FOODON:03309927")],
        ),
    ]
    fs.upsert_chunks(chunks)
    meta = fs.build_entities()

    assert meta.phase == "build_entities"
    assert meta.record_count == 2

    olive = fs.entity_store.get("FOODON:03309927")
    assert olive is not None
    assert olive.prefix == "FOODON"
    assert olive.mention_count == 2
    assert olive.chunk_count == 2
    assert sorted(olive.chunk_ids) == ["c1", "c2"]
    assert olive.facet_hint == "foods"

    iron = fs.entity_store.get("CHEBI:18248")
    assert iron is not None
    assert iron.prefix == "CHEBI"
    assert iron.chunk_count == 1
    assert iron.facet_hint == "nutrients"


def test_build_entities_label_falls_back_to_most_frequent_surface() -> None:
    """For non-FOODON prefixes (no ontology enrichment), the most-common surface form wins."""
    fs = FoodScholar.in_memory()
    m1 = _mention("China", entity_type="other")
    m2 = _mention("China", entity_type="other")
    m3 = _mention("PRC", entity_type="other")
    fs.upsert_chunks(
        [
            _chunk("c1", mentions=[m1], links=[_link(m1, "GAZ:00002845")]),
            _chunk("c2", mentions=[m2, m3], links=[_link(m2, "GAZ:00002845"), _link(m3, "GAZ:00002845")]),
        ]
    )
    fs.build_entities()
    gaz = fs.entity_store.get("GAZ:00002845")
    assert gaz is not None
    assert gaz.label == "China"  # 2 occurrences vs PRC's 1
    assert gaz.mention_count == 3


def test_build_entities_writes_graph_edges() -> None:
    fs = FoodScholar.in_memory()
    m = _mention("olive oil", entity_type="food")
    fs.upsert_chunks([_chunk("c1", mentions=[m], links=[_link(m, "FOODON:03309927")])])
    fs.build_entities()
    # The in-memory graph store keeps an entity → {chunk_id: (conf, method)} map.
    bucket = fs.graph_store._entity_chunks["FOODON:03309927"]  # type: ignore[attr-defined]
    assert "c1" in bucket
    conf, method = bucket["c1"]
    assert method == "dense"
    assert conf == 0.9


def test_build_entities_keeps_highest_confidence_per_chunk() -> None:
    """If the same entity is linked twice in a chunk, the edge carries the max confidence."""
    fs = FoodScholar.in_memory()
    m1 = _mention("olive oil", entity_type="food", start=0)
    m2 = _mention("olive oil", entity_type="food", start=20)
    fs.upsert_chunks(
        [
            _chunk(
                "c1",
                mentions=[m1, m2],
                links=[
                    _link(m1, "FOODON:03309927", confidence=0.7),
                    _link(m2, "FOODON:03309927", confidence=0.95),
                ],
            )
        ]
    )
    fs.build_entities()
    conf, _ = fs.graph_store._entity_chunks["FOODON:03309927"]["c1"]  # type: ignore[attr-defined]
    assert conf == 0.95


def test_build_entities_idempotent() -> None:
    fs = FoodScholar.in_memory()
    m = _mention("olive oil", entity_type="food")
    fs.upsert_chunks([_chunk("c1", mentions=[m], links=[_link(m, "FOODON:03309927")])])
    fs.build_entities()
    first = fs.entity_store.get("FOODON:03309927")
    fs.build_entities()
    second = fs.entity_store.get("FOODON:03309927")
    assert first is not None and second is not None
    assert (first.mention_count, first.chunk_count) == (second.mention_count, second.chunk_count)


def test_build_entities_respects_sample_cap() -> None:
    fs = FoodScholar.in_memory()
    m = _mention("olive oil", entity_type="food")
    fs.upsert_chunks(
        [
            _chunk(f"c{i}", mentions=[m], links=[_link(m, "FOODON:03309927")])
            for i in range(20)
        ]
    )
    fs.build_entities(cap_chunk_sample=5)
    olive = fs.entity_store.get("FOODON:03309927")
    assert olive is not None
    assert olive.chunk_count == 20  # full count
    assert len(olive.chunk_ids) == 5  # but only a sample stored inline


# ------------------------------------------------------------- fs.entities view


def test_entities_namespace_lookup_and_search() -> None:
    fs = FoodScholar.in_memory()
    m_olive = _mention("olive oil", entity_type="food")
    m_iron = _mention("iron", entity_type="micronutrient")
    fs.upsert_chunks(
        [
            _chunk(
                "c1",
                mentions=[m_olive, m_iron],
                links=[
                    _link(m_olive, "FOODON:03309927"),
                    _link(m_iron, "CHEBI:18248"),
                ],
            )
        ]
    )
    fs.build_entities()

    assert fs.entities.get("FOODON:03309927") is not None
    assert len(fs.entities) == 2

    foodon_only = fs.entities.list(prefix="FOODON")
    assert [e.ontology_id for e in foodon_only] == ["FOODON:03309927"]

    [match] = fs.entities.search("olive")
    assert match.ontology_id == "FOODON:03309927"


def test_entities_chunks_for_foodon_uses_sample_for_in_memory() -> None:
    fs = FoodScholar.in_memory()
    m = _mention("olive oil", entity_type="food")
    fs.upsert_chunks([_chunk("c1", mentions=[m], links=[_link(m, "FOODON:03309927")])])
    fs.build_entities()
    chunks = fs.entities.chunks_for("FOODON:03309927")
    assert [c.chunk_id for c in chunks] == ["c1"]


def test_fs_init_includes_entity_store() -> None:
    """fs.init() now also calls entity_store.init() — verify both routes are touched."""
    calls: list[str] = []

    class _Rec:
        def init(self) -> None:
            calls.append(type(self).__name__)

        def __getattr__(self, name):  # pragma: no cover — fail loud on unexpected method
            raise AttributeError(name)

    fs = FoodScholar.in_memory()
    fs.chunk_store = _Rec()  # type: ignore[assignment]
    fs.entity_store = _Rec()  # type: ignore[assignment]
    fs.graph_store = _Rec()  # type: ignore[assignment]
    fs.init()
    assert calls.count("_Rec") == 3
