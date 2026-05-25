"""Layer B per-shelf builders (Pass 1, Pass 2, full per-shelf pipeline,
top-level orchestrator). Tests grow as Phases 1-4 land their respective
builders."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("igraph")
pytest.importorskip("leidenalg")

from foodscholar.config import LayerBConfig  # noqa: E402
from foodscholar.io.chunk import Chunk  # noqa: E402
from foodscholar.layer_b.builder import build_shelf_similarity_candidates  # noqa: E402


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
