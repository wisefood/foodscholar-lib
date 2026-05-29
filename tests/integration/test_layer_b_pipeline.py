"""Mini-corpus end-to-end test for fs.build_layer_b().

Two shelves x 8 chunks (4 sim cluster A + 4 sim cluster B per shelf).
Each cluster also shares 2 FoodOn entities so the relatedness pass picks
them up too. We expect: ≥ 2 themed shelves, ≥ 4 themes total (2 per
shelf), and the cross-store parity audit to pass.

Uses LayerBConfig.labeling.strategy='keyword' to avoid LLM dependency
in CI; the keyword labeling path is deterministic.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("igraph")
pytest.importorskip("leidenalg")
pytest.importorskip("sklearn")

from foodscholar import FoodScholar  # noqa: E402
from foodscholar.io.chunk import Chunk, EntityLink, Mention  # noqa: E402
from foodscholar.io.graph import Shelf  # noqa: E402


def _link(oid: str, conf: float = 0.95) -> EntityLink:
    m = Mention(text="x", start=0, end=1, score=conf, ner_model_version="v")
    return EntityLink(
        mention=m, ontology_id=oid, confidence=conf, method="dense", linker_version="v",
    )


def _chunk(
    cid: str,
    *,
    text: str,
    vec,  # type: ignore[no-untyped-def]
    links: list[EntityLink],
    shelf_ids: list[str],
) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=text,
        source_doc_id="d",
        source_type="abstract",
        section_type="abstract",
        embedding=vec.tolist(),
        embedding_model="test-bge",
        entity_links=links,
        shelf_ids=shelf_ids,
    )


def test_build_layer_b_runs_end_to_end_on_in_memory_stores() -> None:
    """Full Layer B pipeline against in-memory stores. Assertions:
      - artifact.n_shelves_themed == 2 (both eligible shelves themed)
      - artifact.n_themes_total >= 4 (at least 2 themes per shelf)
      - audit_layer_b passes (parity = 1.0, no dangling, no empty themes)
    """
    fs = FoodScholar.in_memory()

    rng = np.random.default_rng(42)
    chunks: list[Chunk] = []

    # Shelf S1 — 4 calcium chunks + 4 cholesterol chunks
    for i in range(4):
        v = np.zeros(8)
        v[0] = 1.0
        v += rng.normal(0, 0.01, 8)
        v /= np.linalg.norm(v)
        chunks.append(
            _chunk(
                f"s1a{i}",
                text=f"calcium bone density study {i}",
                vec=v,
                links=[_link("FOODON:CALCIUM"), _link("FOODON:BONE")],
                shelf_ids=["s1"],
            )
        )
    for i in range(4):
        v = np.zeros(8)
        v[1] = 1.0
        v += rng.normal(0, 0.01, 8)
        v /= np.linalg.norm(v)
        chunks.append(
            _chunk(
                f"s1b{i}",
                text=f"cholesterol cardiovascular risk {i}",
                vec=v,
                links=[_link("FOODON:CHOLESTEROL"), _link("FOODON:HEART")],
                shelf_ids=["s1"],
            )
        )
    # Shelf S2 — 4 vitamin D + 4 iron
    for i in range(4):
        v = np.zeros(8)
        v[2] = 1.0
        v += rng.normal(0, 0.01, 8)
        v /= np.linalg.norm(v)
        chunks.append(
            _chunk(
                f"s2a{i}",
                text=f"vitamin D supplementation {i}",
                vec=v,
                links=[_link("FOODON:VITAMIND"), _link("FOODON:SUPPLEMENT")],
                shelf_ids=["s2"],
            )
        )
    for i in range(4):
        v = np.zeros(8)
        v[3] = 1.0
        v += rng.normal(0, 0.01, 8)
        v /= np.linalg.norm(v)
        chunks.append(
            _chunk(
                f"s2b{i}",
                text=f"iron deficiency anemia {i}",
                vec=v,
                links=[_link("FOODON:IRON"), _link("FOODON:ANEMIA")],
                shelf_ids=["s2"],
            )
        )

    fs.upsert_chunks(chunks)
    fs.graph_store.upsert_shelves(
        [
            Shelf(shelf_id="s1", label="s1", facet="foods", depth=1, chunk_count=8),
            Shelf(shelf_id="s2", label="s2", facet="foods", depth=1, chunk_count=8),
        ]
    )
    # Wire shelf↔chunk attachments via the graph store (mirrors what fs.attach()
    # would have done). The in-memory graph store stores them in _shelf_chunks.
    for cid in [f"s1a{i}" for i in range(4)] + [f"s1b{i}" for i in range(4)]:
        fs.graph_store.attach_chunks_to_shelf("s1", [(cid, [])])
    for cid in [f"s2a{i}" for i in range(4)] + [f"s2b{i}" for i in range(4)]:
        fs.graph_store.attach_chunks_to_shelf("s2", [(cid, [])])

    # Loosen knobs for the small fixture
    fs.config.layer_b.min_chunks_per_shelf = 4
    fs.config.layer_b.leiden.min_community_size = 2
    fs.config.layer_b.similarity.knn_k = 2
    fs.config.layer_b.similarity.edge_threshold = 0.5
    fs.config.layer_b.relatedness.min_shared_ids = 2
    fs.config.layer_b.relatedness.max_doc_frequency = 1.0
    fs.config.layer_b.merge.dedupe_threshold = 0.3
    fs.config.layer_b.labeling.strategy = "keyword"  # no LLM dep in CI

    artifact = fs.build_layer_b(facet="foods")

    # Fetch themes and assert shelf reachability
    themes = fs.graph_store.list_themes()
    cross_shelf_themes = [t for t in themes if len(t.shelf_ids) >= 2]
    print(f"cross-shelf themes: {len(cross_shelf_themes)} of {len(themes)}")
    assert all(len(t.shelf_ids) >= 1 for t in themes), \
        "every theme must be reachable from at least one shelf"

    # Two shelves clusterable, both themed.
    assert artifact.n_shelves_themed == 2, f"got {artifact}"
    # >= 2 themes per shelf x 2 shelves = >= 4 themes total.
    assert artifact.n_themes_total >= 4, f"got {artifact}"

    # Cross-store parity
    from foodscholar.layer_b.audit import audit_layer_b

    report = audit_layer_b(fs.chunk_store, fs.graph_store)
    assert report.passed, f"audit failed: {report}"


def test_build_layer_b_dry_run_persists_nothing() -> None:
    """dry_run=True returns the artifact but writes no themes."""
    fs = FoodScholar.in_memory()
    rng = np.random.default_rng(0)
    chunks: list[Chunk] = []
    for i in range(6):
        v = np.zeros(4)
        v[0] = 1.0
        v += rng.normal(0, 0.01, 4)
        v /= np.linalg.norm(v)
        chunks.append(
            _chunk(
                f"c{i}",
                text=f"calcium intake {i}",
                vec=v,
                links=[_link("FOODON:1"), _link("FOODON:2")],
                shelf_ids=["s1"],
            )
        )
    fs.upsert_chunks(chunks)
    fs.graph_store.upsert_shelves(
        [Shelf(shelf_id="s1", label="s1", facet="foods", depth=1, chunk_count=6)]
    )
    for cid in [f"c{i}" for i in range(6)]:
        fs.graph_store.attach_chunks_to_shelf("s1", [(cid, [])])

    fs.config.layer_b.min_chunks_per_shelf = 4
    fs.config.layer_b.leiden.min_community_size = 2
    fs.config.layer_b.similarity.knn_k = 2
    fs.config.layer_b.similarity.edge_threshold = 0.5
    fs.config.layer_b.relatedness.min_shared_ids = 2
    fs.config.layer_b.relatedness.max_doc_frequency = 1.0
    fs.config.layer_b.labeling.strategy = "keyword"

    artifact = fs.build_layer_b(facet="foods", dry_run=True)
    # Builder ran (theme counts reported), but nothing landed in the stores
    assert artifact.n_shelves_themed >= 0
    assert fs.graph_store.list_themes() == []
    for chunk in fs.chunk_store.scan():
        assert chunk.theme_ids == []
