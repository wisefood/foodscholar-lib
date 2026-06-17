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


def test_bertopic_mode_skips_pass2_relatedness(monkeypatch) -> None:
    """In bertopic mode Pass 2 (relatedness) is NOT run — the two passes are
    orthogonal and never merge, so relatedness just bolts on noise. Only
    BERTopic (global_similarity) themes should be persisted; no 'relatedness'
    and no 'merged' themes, even when chunks carry strong entity links that
    WOULD form relatedness communities in leiden mode."""
    from foodscholar import FoodScholar, FoodScholarConfig
    from foodscholar.io.chunk import EntityLink, Mention
    from foodscholar.io.graph import Shelf
    from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore

    def _link(oid: str) -> EntityLink:
        m = Mention(text="x", start=0, end=1, score=0.95, ner_model_version="v")
        return EntityLink(mention=m, ontology_id=oid, confidence=0.95,
                          method="dense", linker_version="v")

    def _chunk_with_entities(cid: str) -> Chunk:
        # Two shared FoodOn ids across all chunks → a relatedness community
        # would form if Pass 2 ran.
        return Chunk(
            chunk_id=cid, text=f"t {cid}", source_doc_id="d",
            source_type="abstract", section_type="abstract",
            embedding=[1.0, 0.0, 0.0], embedding_model="test",
            entity_links=[_link("FOODON:1"), _link("FOODON:2")],
        )

    chunk_store = InMemoryChunkStore()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_shelves([
        Shelf(shelf_id="shelf:dairy", label="dairy", facet="foods", depth=1,
              chunk_count=4),
    ])
    chunk_store.upsert([_chunk_with_entities(f"c{i}") for i in range(4)])
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
            "leiden": {"min_community_size": 2},
            "relatedness": {"min_shared_ids": 2, "max_doc_frequency": 1.0},
        },
    })
    fs = FoodScholar(cfg, chunk_store=chunk_store, graph_store=graph_store)

    monkeypatch.setattr(
        "foodscholar.layer_b.builder.run_bertopic",
        lambda ids, store, bcfg: [{"c0", "c1", "c2"}],
    )
    art = fs.build_layer_b(facet="foods", dry_run=False)

    passes = {t.discovery_pass for t in fs.graph_store.list_themes()}
    assert "relatedness" not in passes, "Pass 2 ran in bertopic mode"
    assert "merged" not in passes, "merge fired in bertopic mode"
    assert passes == {"global_similarity"}, f"unexpected passes: {passes}"
    assert "relatedness" not in art.n_themes_by_pass


def test_bertopic_ignores_global_pass1_mode_and_never_runs_leiden(monkeypatch) -> None:
    """bertopic + pass1_mode='global' must NOT fall through to Leiden's global
    branch (the old silent bug). It runs BERTopic per-shelf and never calls any
    Leiden candidate builder."""
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
            "pass1_mode": "global",  # the previously-silent bad combo
            "min_chunks_per_shelf": 2,
            "min_embedded_fraction": 0.0,
            "bertopic": {"min_topic_size": 2},
        },
    })
    fs = FoodScholar(cfg, chunk_store=chunk_store, graph_store=graph_store)

    # Trip-wire: Leiden's similarity builder must never be called in bertopic mode.
    def _boom(*a, **k):
        raise AssertionError("Leiden build_global_similarity_candidates ran in bertopic mode")

    monkeypatch.setattr(
        "foodscholar.layer_b.builder.build_global_similarity_candidates", _boom
    )
    monkeypatch.setattr(
        "foodscholar.layer_b.builder.run_bertopic",
        lambda ids, store, bcfg: [{"c0", "c1", "c2"}],
    )
    fs.build_layer_b(facet="foods", dry_run=False)

    themes = fs.graph_store.list_themes()
    assert themes and all(t.discovered_by == "bertopic" for t in themes)


def test_leiden_subtree_scope_unions_descendant_chunks(monkeypatch) -> None:
    """scope='subtree' for LEIDEN feeds a parent shelf its own chunks PLUS every
    descendant shelf's chunks into Pass 1 (the knob used to be bertopic-only)."""
    from foodscholar import FoodScholar, FoodScholarConfig
    from foodscholar.io.graph import Shelf
    from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore

    chunk_store = InMemoryChunkStore()
    graph_store = InMemoryGraphStore()
    # parent (fruit) with 1 direct chunk + child (apple) with 3 direct chunks.
    graph_store.upsert_shelves([
        Shelf(shelf_id="shelf:fruit", label="fruit", facet="foods", depth=1),
        Shelf(shelf_id="shelf:apple", label="apple", facet="foods", depth=2,
              parent_shelf_id="shelf:fruit"),
    ])
    chunk_store.upsert([_chunk(f"c{i}") for i in range(4)])
    graph_store.attach_chunks_to_shelf("shelf:fruit", [("c0", [])])
    graph_store.attach_chunks_to_shelf("shelf:apple", [(f"c{i}", []) for i in (1, 2, 3)])

    cfg = FoodScholarConfig.model_validate({
        "corpus": {"chunks_path": "data/chunks.parquet"},
        "storage": {"chunk_store": {"backend": "memory"},
                    "graph_store": {"backend": "memory"}},
        "layer_b": {
            "algorithm": "leiden",
            "pass1_mode": "per_shelf",
            "scope": "subtree",
            "min_chunks_per_shelf": 3,   # fruit has only 1 direct → needs subtree to qualify
            "min_embedded_fraction": 0.0,
        },
    })
    fs = FoodScholar(cfg, chunk_store=chunk_store, graph_store=graph_store)

    seen_scoped: dict[str, set] = {}

    def _capture(chunk_ids, chunk_store, cfg):  # noqa: ANN001
        # record what the fruit shelf was fed; return no candidates (we only
        # assert on the scoping, not on clustering output).
        seen_scoped[tuple(sorted(chunk_ids))] = set(chunk_ids)
        return []

    monkeypatch.setattr(
        "foodscholar.layer_b.builder.build_global_similarity_candidates", _capture
    )
    fs.build_layer_b(facet="foods", dry_run=True)

    # The fruit shelf must have been fed all 4 chunks (its own + apple's), so a
    # 4-chunk scoped set was seen — impossible under scope='direct' (fruit=1).
    assert any(len(s) == 4 for s in seen_scoped.values()), (
        f"subtree scope did not union descendants; saw {seen_scoped}"
    )
