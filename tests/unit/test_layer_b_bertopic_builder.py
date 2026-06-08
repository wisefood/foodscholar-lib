"""Builder dispatch: cfg.algorithm='bertopic' routes Pass 1 through run_bertopic;
subtree scope expands a shelf's chunks to include descendants."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from foodscholar.config import LayerBConfig  # noqa: E402
from foodscholar.io.chunk import Chunk  # noqa: E402
from foodscholar.layer_b.builder import build_shelf_bertopic_candidates  # noqa: E402
from foodscholar.storage.memory import InMemoryChunkStore  # noqa: E402


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid, text=f"t {cid}", source_doc_id="d",
        source_type="abstract", section_type="abstract",
        embedding=[1.0, 0.0, 0.0], embedding_model="test",
    )


def test_bertopic_candidates_from_groups(monkeypatch) -> None:
    cs = InMemoryChunkStore()
    cs.upsert([_chunk("c1"), _chunk("c2"), _chunk("c3")])
    cfg = LayerBConfig(algorithm="bertopic")

    # stub run_bertopic to return one group of two chunk ids
    monkeypatch.setattr(
        "foodscholar.layer_b.builder.run_bertopic",
        lambda ids, store, bcfg: [{"c1", "c2"}],
    )
    cands = build_shelf_bertopic_candidates(["c1", "c2", "c3"], cs, cfg)
    assert len(cands) == 1
    c = cands[0]
    assert c.chunk_ids == {"c1", "c2"}
    assert c.discovered_by == "bertopic"
    assert c.pass_name == "global_similarity"
    assert c.centroid_embedding is not None


def test_bertopic_candidates_empty_when_no_groups(monkeypatch) -> None:
    cs = InMemoryChunkStore()
    cs.upsert([_chunk("c1")])
    monkeypatch.setattr(
        "foodscholar.layer_b.builder.run_bertopic",
        lambda ids, store, bcfg: [],
    )
    assert build_shelf_bertopic_candidates(["c1"], cs, LayerBConfig()) == []


def test_build_layer_b_dispatches_to_bertopic(monkeypatch) -> None:
    """End-to-end: cfg.algorithm='bertopic' makes per-shelf Pass 1 use
    run_bertopic; the persisted theme is discovered_by='bertopic'."""
    from foodscholar import FoodScholar, FoodScholarConfig
    from foodscholar.io.graph import Shelf
    from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore

    chunk_store = InMemoryChunkStore()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_shelves([
        Shelf(shelf_id="shelf:dairy", label="dairy", facet="foods", depth=1,
              chunk_count=4),
    ])
    chunk_store.upsert([_chunk(f"c{i}") for i in range(4)])
    graph_store.attach_chunks_to_shelf("shelf:dairy", [(f"c{i}", []) for i in range(4)])

    cfg = FoodScholarConfig.model_validate({
        "corpus": {"chunks_path": "data/chunks.parquet"},
        "storage": {"chunk_store": {"backend": "memory"},
                    "graph_store": {"backend": "memory"}},
        "layer_b": {
            "algorithm": "bertopic",
            "pass1_mode": "per_shelf",
            "min_chunks_per_shelf": 2,
            "min_embedded_fraction": 0.0,
            "bertopic": {"min_topic_size": 2},
        },
    })
    fs = FoodScholar(cfg, chunk_store=chunk_store, graph_store=graph_store)

    # stub BERTopic clustering: one topic of 3 chunks
    monkeypatch.setattr(
        "foodscholar.layer_b.builder.run_bertopic",
        lambda ids, store, bcfg: [{"c0", "c1", "c2"}],
    )
    fs.build_layer_b(facet="foods", dry_run=False)

    themes = fs.graph_store.list_themes()
    assert themes, "expected a bertopic theme"
    assert any(t.discovered_by == "bertopic" for t in themes)
