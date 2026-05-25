"""Pass 2 — relatedness graph from shared FoodOn entity IDs."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("igraph")

from foodscholar.config import RelatednessConfig
from foodscholar.io.chunk import Chunk, EntityLink, Mention
from foodscholar.layer_b.relatedness_graph import build_relatedness_graph


def _link(oid: str, conf: float = 0.95) -> EntityLink:
    m = Mention(text="x", start=0, end=1, score=conf, ner_model_version="v")
    return EntityLink(
        mention=m, ontology_id=oid, confidence=conf, method="dense", linker_version="v",
    )


def _chunk(cid: str, *, links: list[EntityLink]) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=f"text {cid}",
        source_doc_id="d",
        source_type="abstract",
        section_type="abstract",
        entity_links=links,
    )


def test_relatedness_graph_edge_when_two_share_two_entities() -> None:
    chunks = [
        _chunk("a", links=[_link("FOODON:1"), _link("FOODON:2")]),
        _chunk("b", links=[_link("FOODON:1"), _link("FOODON:2")]),
        _chunk("c", links=[_link("FOODON:1"), _link("FOODON:3")]),
    ]
    cfg = RelatednessConfig(
        tau_strict=0.5,
        min_shared_ids=2,
        max_doc_frequency=1.0,
        always_exclude_iris=[],
    )
    g = build_relatedness_graph(chunks, cfg)
    assert g.vcount() == 3
    assert g.ecount() == 1


def test_relatedness_graph_excludes_low_confidence_links() -> None:
    """Links below tau_strict don't participate in edge formation."""
    chunks = [
        _chunk("a", links=[_link("FOODON:1", 0.5), _link("FOODON:2", 0.5)]),
        _chunk("b", links=[_link("FOODON:1", 0.5), _link("FOODON:2", 0.5)]),
    ]
    cfg = RelatednessConfig(
        tau_strict=0.80,
        min_shared_ids=1,
        max_doc_frequency=1.0,
        always_exclude_iris=[],
    )
    g = build_relatedness_graph(chunks, cfg)
    assert g.ecount() == 0


def test_relatedness_graph_drops_ubiquitous_entities() -> None:
    """An entity appearing in > max_doc_frequency of chunks contributes no edges."""
    chunks = [
        _chunk("a", links=[_link("FOODON:U"), _link("FOODON:1"), _link("FOODON:2")]),
        _chunk("b", links=[_link("FOODON:U"), _link("FOODON:1"), _link("FOODON:2")]),
        _chunk("c", links=[_link("FOODON:U")]),
        _chunk("d", links=[_link("FOODON:U")]),
    ]
    # FOODON:U appears in 4/4 = 100% → dropped at max_doc_frequency=0.5
    cfg = RelatednessConfig(
        tau_strict=0.5,
        min_shared_ids=2,
        max_doc_frequency=0.5,
        always_exclude_iris=[],
    )
    g = build_relatedness_graph(chunks, cfg)
    # After dropping FOODON:U, a-b still share FOODON:1 + FOODON:2 ≥ 2 → 1 edge.
    assert g.ecount() == 1


def test_relatedness_graph_always_exclude_iris() -> None:
    chunks = [
        _chunk("a", links=[_link("FOODON:00001002"), _link("FOODON:X")]),
        _chunk("b", links=[_link("FOODON:00001002"), _link("FOODON:X")]),
    ]
    cfg = RelatednessConfig(
        tau_strict=0.5,
        min_shared_ids=2,
        max_doc_frequency=1.0,
        always_exclude_iris=["FOODON:00001002"],
    )
    g = build_relatedness_graph(chunks, cfg)
    # After excluding the umbrella class, only FOODON:X remains shared (1 < 2).
    assert g.ecount() == 0


def test_relatedness_graph_edge_weight_idf_style() -> None:
    """w(i,j) = Σ_e∈shared 1 / log(1 + doc_freq[e]).
    Rare entities weight more than common ones."""
    # 2 chunks share a rare entity (df=2) AND a common entity (df=10).
    chunks: list[Chunk] = []
    chunks.append(_chunk("a", links=[_link("RARE"), _link("COMMON")]))
    chunks.append(_chunk("b", links=[_link("RARE"), _link("COMMON")]))
    for i in range(8):
        chunks.append(_chunk(f"d{i}", links=[_link("COMMON")]))
    cfg = RelatednessConfig(
        tau_strict=0.5,
        min_shared_ids=2,
        max_doc_frequency=1.0,
        always_exclude_iris=[],
    )
    g = build_relatedness_graph(chunks, cfg)
    assert g.ecount() == 1
    expected = 1.0 / math.log(1 + 2) + 1.0 / math.log(1 + 10)
    assert g.es["weight"][0] == pytest.approx(expected, rel=1e-6)


def test_relatedness_graph_empty_returns_empty() -> None:
    cfg = RelatednessConfig()
    g = build_relatedness_graph([], cfg)
    assert g.vcount() == 0
    assert g.ecount() == 0
