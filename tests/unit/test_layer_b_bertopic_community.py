"""run_bertopic: cluster chunk embeddings -> chunk-id groups (hdbscan|kmeans)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("bertopic")
pytest.importorskip("sklearn")

from foodscholar.config import BertopicConfig  # noqa: E402
from foodscholar.io.chunk import Chunk  # noqa: E402
from foodscholar.layer_b.bertopic_community import run_bertopic  # noqa: E402
from foodscholar.storage.memory import InMemoryChunkStore  # noqa: E402


def _chunk(cid: str, vec) -> Chunk:
    return Chunk(
        chunk_id=cid, text=f"text for {cid}", source_doc_id="d",
        source_type="abstract", section_type="abstract",
        embedding=list(vec), embedding_model="test",
    )


def _two_cluster_store(n_per: int = 12, dim: int = 16):
    """Two well-separated blobs in embedding space."""
    rng = np.random.RandomState(0)
    a = rng.normal(loc=+3.0, scale=0.1, size=(n_per, dim))
    b = rng.normal(loc=-3.0, scale=0.1, size=(n_per, dim))
    cs = InMemoryChunkStore()
    chunks = []
    for i in range(n_per):
        chunks.append(_chunk(f"a{i}", a[i]))
        chunks.append(_chunk(f"b{i}", b[i]))
    cs.upsert(chunks)
    return cs, [c.chunk_id for c in chunks]


def test_kmeans_returns_chunk_id_groups() -> None:
    cs, ids = _two_cluster_store()
    cfg = BertopicConfig(clusterer="kmeans", n_clusters=2, min_topic_size=2)
    groups = run_bertopic(ids, cs, cfg)
    assert groups, "expected at least one topic"
    # every group is a set of known chunk ids; no overlap; covered ids ⊆ input
    seen: set[str] = set()
    for g in groups:
        assert g.issubset(set(ids))
        assert not (g & seen)
        seen |= g


def test_min_topic_size_filters_small_topics() -> None:
    cs, ids = _two_cluster_store(n_per=12)
    cfg = BertopicConfig(clusterer="kmeans", n_clusters=2, min_topic_size=100)
    # min size 100 > any topic → everything filtered out
    assert run_bertopic(ids, cs, cfg) == []


def test_empty_input_returns_empty() -> None:
    cs = InMemoryChunkStore()
    assert run_bertopic([], cs, BertopicConfig()) == []


def test_hdbscan_runs_and_groups_are_disjoint() -> None:
    cs, ids = _two_cluster_store(n_per=20)
    cfg = BertopicConfig(clusterer="hdbscan", min_topic_size=5)
    groups = run_bertopic(ids, cs, cfg)
    # HDBSCAN may merge/own-bucket; assert structural invariants only
    seen: set[str] = set()
    for g in groups:
        assert g.issubset(set(ids))
        assert not (g & seen)
        seen |= g
