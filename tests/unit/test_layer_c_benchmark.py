"""Layer C benchmark: run all methods over a theme -> MethodResult list."""

from __future__ import annotations

import pytest

from foodscholar.config import FoodScholarConfig, LayerCConfig
from foodscholar.io.chunk import Chunk
from foodscholar.io.graph import Shelf, Theme
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore

pytest.importorskip("sumy")

from foodscholar.layer_c.benchmark import benchmark_theme


def _fs():
    cs = InMemoryChunkStore()
    gs = InMemoryGraphStore()
    docs = [
        "Oats are a whole grain rich in soluble fiber called beta glucan.",
        "Beta glucan in oats can lower cholesterol and improve heart health.",
        "Rice is a staple cereal grain eaten across the world.",
        "Wheat flour is milled from wheat and used to bake bread.",
    ]
    cs.upsert([Chunk(chunk_id=f"c{i}", text=t, source_doc_id="d",
                     source_type="abstract", section_type="abstract")
               for i, t in enumerate(docs)])
    gs.upsert_shelves([Shelf(shelf_id="s1", label="cereal", facet="foods", depth=1)])
    gs.upsert_themes([Theme(theme_id="t1", label="Cereal grains", shelf_ids=["s1"],
                            chunk_count=4, discovered_by="leiden", discovery_version="v",
                            facet="foods", discovery_pass="merged",
                            keyword_terms=["oat", "rice"])])
    # signature is (chunk_id, theme_id, primary, weight)
    gs.attach_chunks_to_themes_bulk([(f"c{i}", "t1", i == 0, 1.0) for i in range(4)])

    class _FS:
        pass

    fs = _FS()
    fs.chunk_store = cs
    fs.graph_store = gs
    from foodscholar.graph_view import GraphView
    fs.graph = GraphView(cs, gs)
    fs.config = FoodScholarConfig(corpus={"chunks_path": "x"})
    fs.config.layer_c = LayerCConfig(stage1_sentences=2)
    return fs


def test_benchmark_theme_returns_all_methods() -> None:
    fs = _fs()
    results = benchmark_theme(fs, "t1")
    methods = {r.method for r in results}
    assert methods == {"lexrank", "lsa", "luhn", "textrank", "nltk_freq"}
    for r in results:
        assert r.input_chunks == 4
        assert r.input_chars > 0
        assert r.summary_length_chars == len(r.summary)
        assert r.execution_time_ms >= 0
