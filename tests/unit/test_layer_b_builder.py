"""Layer B per-shelf builders (Pass 1, Pass 2, full per-shelf pipeline,
top-level orchestrator). Tests grow as Phases 1-4 land their respective
builders."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("igraph")
pytest.importorskip("leidenalg")

from foodscholar.config import LayerBConfig  # noqa: E402
from foodscholar.io.chunk import Chunk, EntityLink, Mention  # noqa: E402
from foodscholar.layer_b.builder import (  # noqa: E402
    build_shelf_relatedness_candidates,
    build_shelf_similarity_candidates,
)


def _chunk(cid: str, *, text: str = "x", vec=None, source_type: str = "abstract") -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=text,
        source_doc_id="d",
        source_type=source_type,  # type: ignore[arg-type]
        section_type="abstract",
        embedding=vec.tolist() if vec is not None else None,
        embedding_model="test" if vec is not None else None,
    )


# ----------------------------------------------------------------------------
# Pass 1 (similarity) per-shelf builder
# ----------------------------------------------------------------------------


def test_build_shelf_similarity_candidates_recovers_two_clusters() -> None:
    """Two well-separated embedding clusters → two similarity candidates."""
    rng = np.random.default_rng(42)
    chunks: list[Chunk] = []
    for i in range(6):
        v = np.zeros(8)
        v[0] = 1.0
        v += rng.normal(0, 0.01, 8)
        v /= np.linalg.norm(v)
        chunks.append(_chunk(f"a{i}", text="calcium bone density", vec=v))
    for i in range(6):
        v = np.zeros(8)
        v[1] = 1.0
        v += rng.normal(0, 0.01, 8)
        v /= np.linalg.norm(v)
        chunks.append(_chunk(f"b{i}", text="cholesterol cardiovascular", vec=v))

    cfg = LayerBConfig()
    cfg.leiden.min_community_size = 3
    cfg.similarity.knn_k = 4
    cfg.similarity.edge_threshold = 0.5

    candidates = build_shelf_similarity_candidates(chunks, cfg)
    assert len(candidates) == 2
    cluster_a = {f"a{i}" for i in range(6)}
    cluster_b = {f"b{i}" for i in range(6)}
    cand_a, cand_b = candidates
    if cand_a.chunk_ids.issubset(cluster_a):
        assert cand_b.chunk_ids.issubset(cluster_b)
    else:
        assert cand_a.chunk_ids.issubset(cluster_b)
        assert cand_b.chunk_ids.issubset(cluster_a)


def test_build_shelf_similarity_candidates_skips_chunks_without_embeddings() -> None:
    """Chunks with embedding=None must be excluded from the graph and absent
    from output candidates, not crash the builder."""
    chunk_with_vec = _chunk("a", vec=np.array([1.0, 0.0]))
    chunk_no_vec = _chunk("b")  # no embedding
    cfg = LayerBConfig()
    cfg.leiden.min_community_size = 1
    cfg.similarity.knn_k = 1
    candidates = build_shelf_similarity_candidates([chunk_with_vec, chunk_no_vec], cfg)
    for cand in candidates:
        assert "b" not in cand.chunk_ids


def test_build_shelf_similarity_candidates_emits_centroid() -> None:
    """Each candidate's centroid_embedding is the mean of its members'
    normalized vectors — needed by the primary picker for sim themes."""
    rng = np.random.default_rng(0)
    chunks = []
    for i in range(5):
        v = np.zeros(4)
        v[0] = 1.0
        v += rng.normal(0, 0.01, 4)
        v /= np.linalg.norm(v)
        chunks.append(_chunk(f"c{i}", vec=v))

    cfg = LayerBConfig()
    cfg.leiden.min_community_size = 2
    cfg.similarity.knn_k = 3
    cfg.similarity.edge_threshold = 0.5

    candidates = build_shelf_similarity_candidates(chunks, cfg)
    assert candidates
    for cand in candidates:
        assert cand.centroid_embedding is not None
        assert len(cand.centroid_embedding) == 4
        # All vectors point near +x → centroid should too.
        assert cand.centroid_embedding[0] > 0.9


def test_build_shelf_similarity_candidates_empty_returns_empty() -> None:
    cfg = LayerBConfig()
    assert build_shelf_similarity_candidates([], cfg) == []


# ----------------------------------------------------------------------------
# Pass 2 (relatedness) per-shelf builder
# ----------------------------------------------------------------------------


def _link(oid: str, conf: float = 0.95) -> EntityLink:
    m = Mention(text="x", start=0, end=1, score=conf, ner_model_version="v")
    return EntityLink(
        mention=m, ontology_id=oid, confidence=conf, method="dense", linker_version="v",
    )


def _entity_chunk(cid: str, links: list[EntityLink]) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=cid,
        source_doc_id="d",
        source_type="abstract",
        section_type="abstract",
        entity_links=links,
    )


def test_build_shelf_relatedness_candidates_groups_by_shared_entities() -> None:
    """Chunks a,b,c share FOODON:1+2; d,e,f share FOODON:3+4. Two
    entity-anchored relatedness candidates expected."""
    chunks: list[Chunk] = []
    for cid in ("a", "b", "c"):
        chunks.append(_entity_chunk(cid, [_link("FOODON:1"), _link("FOODON:2")]))
    for cid in ("d", "e", "f"):
        chunks.append(_entity_chunk(cid, [_link("FOODON:3"), _link("FOODON:4")]))

    cfg = LayerBConfig()
    cfg.leiden.min_community_size = 2
    cfg.relatedness.min_shared_ids = 2
    cfg.relatedness.max_doc_frequency = 1.0

    candidates = build_shelf_relatedness_candidates(chunks, cfg)
    assert len(candidates) == 2
    for cand in candidates:
        assert cand.pass_name == "relatedness"
        # The candidate's foodon_ids = union of high-conf links across members.
        assert len(cand.foodon_ids) >= 2


def test_build_shelf_relatedness_candidates_handles_empty_chunks() -> None:
    cfg = LayerBConfig()
    assert build_shelf_relatedness_candidates([], cfg) == []


def test_build_shelf_relatedness_candidates_handles_no_edges() -> None:
    """Chunks with no shared entities → no edges → no communities → []."""
    chunks = [
        _entity_chunk("a", [_link("FOODON:1"), _link("FOODON:2")]),
        _entity_chunk("b", [_link("FOODON:3"), _link("FOODON:4")]),
        _entity_chunk("c", [_link("FOODON:5"), _link("FOODON:6")]),
    ]
    cfg = LayerBConfig()
    cfg.leiden.min_community_size = 2
    cfg.relatedness.min_shared_ids = 2
    cfg.relatedness.max_doc_frequency = 1.0
    assert build_shelf_relatedness_candidates(chunks, cfg) == []
