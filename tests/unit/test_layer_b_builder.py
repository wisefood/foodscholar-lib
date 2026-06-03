"""Layer B builders: relatedness (per-shelf), global similarity, and
top-level orchestrator. Tests cover Phases 1-4."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("igraph")
pytest.importorskip("leidenalg")

from foodscholar.config import FoodScholarConfig, LayerBConfig  # noqa: E402
from foodscholar.io.chunk import Chunk, EntityLink, Mention  # noqa: E402
from foodscholar.layer_b.builder import (  # noqa: E402
    build_global_similarity_candidates,
    build_shelf_relatedness_candidates,
)
from foodscholar.storage.memory import InMemoryChunkStore  # noqa: E402


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


# ----------------------------------------------------------------------------
# build_global_similarity_candidates (Task 7)
# ----------------------------------------------------------------------------


def _global_store(cluster_vecs: list[tuple[str, list[float]]]) -> tuple[InMemoryChunkStore, list[str]]:
    """Helper: build an InMemoryChunkStore from (chunk_id, vec) pairs."""
    store = InMemoryChunkStore()
    chunks = [
        Chunk(
            chunk_id=cid,
            text=f"text {cid}",
            source_doc_id="d",
            source_type="abstract",
            section_type="other",
            embedding=vec,
            embedding_model="m",
        )
        for cid, vec in cluster_vecs
    ]
    store.upsert(chunks)
    chunk_ids = [cid for cid, _ in cluster_vecs]
    return store, chunk_ids


def test_build_global_similarity_candidates_returns_themecandidate_records() -> None:
    """6 chunks in 2 well-separated clusters → at least 1 ThemeCandidate with
    pass_name='global_similarity', centroid_embedding set, foodon_ids empty."""
    rng = np.random.default_rng(42)

    cluster_vecs: list[tuple[str, list[float]]] = []
    for i in range(3):
        v = np.zeros(8)
        v[0] = 1.0
        v += rng.normal(0, 0.01, 8)
        v /= np.linalg.norm(v)
        cluster_vecs.append((f"a{i}", v.tolist()))
    for i in range(3):
        v = np.zeros(8)
        v[1] = 1.0
        v += rng.normal(0, 0.01, 8)
        v /= np.linalg.norm(v)
        cluster_vecs.append((f"b{i}", v.tolist()))

    store, chunk_ids = _global_store(cluster_vecs)

    cfg = LayerBConfig()
    cfg.leiden.min_community_size = 2
    cfg.similarity.knn_k = 3
    cfg.similarity.edge_threshold = 0.5
    cfg.similarity.require_mutual = False

    candidates = build_global_similarity_candidates(chunk_ids, store, cfg)

    assert len(candidates) >= 1
    for cand in candidates:
        assert cand.pass_name == "global_similarity"
        assert cand.centroid_embedding is not None
        assert len(cand.centroid_embedding) == 8
        assert cand.foodon_ids == set()


def test_build_global_similarity_candidates_returns_empty_when_no_embeddings() -> None:
    """Chunks with embedding=None → graph has no edges → no communities → []."""
    store = InMemoryChunkStore()
    store.upsert([
        Chunk(
            chunk_id="x",
            text="no vec",
            source_doc_id="d",
            source_type="abstract",
            section_type="other",
            embedding=None,
        )
    ])
    cfg = LayerBConfig()
    candidates = build_global_similarity_candidates(["x"], store, cfg)
    assert candidates == []


# ----------------------------------------------------------------------------
# build_layer_b — hybrid global/per-shelf orchestrator (Task 9)
# ----------------------------------------------------------------------------


@pytest.fixture
def cross_shelf_fs():
    """FoodScholar in-memory fixture with 6 chunks split across two shelves.

    Chunks c0-c2 → shelf:fat; c3-c5 → shelf:meat. All 6 share a tight
    embedding cluster (near [1, 0, 0]) so the global similarity pass should
    group them into one cross-shelf community. The per-shelf relatedness
    pass has no entity links so produces nothing — the outcome is a single
    global_similarity theme spanning both shelves.
    """
    from foodscholar import FoodScholar
    from foodscholar.io.graph import Shelf
    from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore

    chunk_store = InMemoryChunkStore()
    graph_store = InMemoryGraphStore()

    # Two shelves under 'foods'.
    graph_store.upsert_shelves([
        Shelf(shelf_id="shelf:fat", label="fat", facet="foods", depth=1, chunk_count=3),
        Shelf(shelf_id="shelf:meat", label="meat", facet="foods", depth=1, chunk_count=3),
    ])

    # 6 chunks — all near [1, 0, 0] so global similarity groups them together.
    def jitter(i: int) -> list[float]:
        v = [1.0 - 0.01 * i, 0.01 * i, 0.0]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v]

    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            text=f"chunk {i} nutrition food",
            source_doc_id="d",
            source_type="textbook",
            section_type="other",
            embedding=jitter(i),
            embedding_model="m",
        )
        for i in range(6)
    ]
    chunk_store.upsert(chunks)

    # Attach c0-c2 to shelf:fat and c3-c5 to shelf:meat using the singular API.
    graph_store.attach_chunks_to_shelf("shelf:fat", [(f"c{i}", []) for i in range(3)])
    graph_store.attach_chunks_to_shelf("shelf:meat", [(f"c{i}", []) for i in range(3, 6)])

    cfg = FoodScholarConfig.model_validate({
        "corpus": {"chunks_path": "data/chunks.parquet"},
        "storage": {
            "chunk_store": {"backend": "memory"},
            "graph_store": {"backend": "memory"},
        },
        "layer_b": {
            "min_chunks_per_shelf": 2,
            "min_embedded_fraction": 0.0,
            "similarity": {"knn_k": 5, "edge_threshold": 0.5},
            "leiden": {"min_community_size": 2},
        },
    })

    return FoodScholar(cfg, chunk_store=chunk_store, graph_store=graph_store)


def test_build_layer_b_emits_cross_shelf_themes_when_global_finds_them(
    cross_shelf_fs,
) -> None:
    """Global similarity pass discovers 6 tightly-clustered chunks split
    across two shelves → the resulting Theme has both shelves in shelf_ids.

    The old per-shelf orchestrator would only see shelf-local clusters and
    could never emit a theme spanning both shelves. This test documents the
    new v0.2 contract.
    """
    fs = cross_shelf_fs
    artifact = fs.build_layer_b(facet="foods", dry_run=False)

    themes = fs.graph_store.list_themes()
    assert themes, "expected at least one theme from the global similarity pass"

    cross_shelf = [t for t in themes if len(t.shelf_ids) >= 2]
    assert cross_shelf, (
        f"expected ≥1 cross-shelf theme; got themes={[(t.label, t.shelf_ids) for t in themes]}"
    )
    # Every theme must reference at least one shelf.
    assert all(len(t.shelf_ids) >= 1 for t in themes)
    # discovery_version must be the new v0.2.
    for t in themes:
        assert t.discovery_version == "v0.2", f"unexpected version on {t.theme_id}: {t.discovery_version}"
    # Artifact reflects the global pass.
    assert artifact.n_themes_total >= 1
    assert "global_similarity" in artifact.n_themes_by_pass
