"""GraphML export: shelves + themes + cards → a typed GraphML graph."""

from __future__ import annotations

import pytest

pytest.importorskip("networkx")
import networkx as nx

from foodscholar import FoodScholar
from foodscholar.io.graph import Card, Shelf, Theme


def _fs_with_graph() -> FoodScholar:
    fs = FoodScholar.in_memory()
    fs.graph_store.upsert_shelves([
        Shelf(shelf_id="s:root", label="foods", facet="foods", depth=0, chunk_count=10),
        Shelf(shelf_id="s:dairy", label="dairy", facet="foods", depth=1,
              parent_shelf_id="s:root", chunk_count=6),
    ])
    fs.graph_store.upsert_themes([
        Theme(theme_id="t:milk", label="milk calcium", shelf_ids=["s:dairy"],
              chunk_count=4, discovered_by="bertopic", discovery_version="v",
              facet="foods", discovery_pass="global_similarity",
              keyword_terms=["milk", "calcium"]),
    ])
    fs.graph_store.upsert_cards([
        Card(card_id="c:milk", target_id="t:milk", target_type="theme",
             title="Milk & calcium", summary="Milk has calcium.",
             evidence_quality="high", cited_chunk_ids=["ch1", "ch2"],
             llm_model="llama-3.1-8b-instant", prompt_version="v1"),
    ])
    return fs


def test_export_graphml_writes_file(tmp_path) -> None:
    fs = _fs_with_graph()
    out = tmp_path / "graph.graphml"
    path = fs.export_graphml(out, facet="foods")
    assert path == out
    assert out.exists()
    g = nx.read_graphml(out)
    # 2 shelves + 1 theme + 1 card
    assert g.number_of_nodes() == 4
    assert g.nodes["s:dairy"]["node_type"] == "shelf"
    assert g.nodes["t:milk"]["node_type"] == "theme"
    assert g.nodes["c:milk"]["node_type"] == "card"


def test_export_graphml_edges(tmp_path) -> None:
    fs = _fs_with_graph()
    out = tmp_path / "g.graphml"
    fs.export_graphml(out, facet="foods")
    g = nx.read_graphml(out)
    assert g.has_edge("s:root", "s:dairy")            # parent_of
    assert g["s:root"]["s:dairy"]["edge_type"] == "parent_of"
    assert g.has_edge("s:dairy", "t:milk")            # has_theme
    assert g["s:dairy"]["t:milk"]["edge_type"] == "has_theme"
    assert g.has_edge("t:milk", "c:milk")             # has_card
    assert g["t:milk"]["c:milk"]["edge_type"] == "has_card"


def test_export_graphml_scalar_attrs_only(tmp_path) -> None:
    # GraphML can't hold list attrs; list fields must be flattened to strings.
    fs = _fs_with_graph()
    out = tmp_path / "g.graphml"
    fs.export_graphml(out, facet="foods")
    g = nx.read_graphml(out)
    for _, data in g.nodes(data=True):
        for v in data.values():
            assert isinstance(v, (str, int, float, bool))
    # theme keyword_terms flattened, label preserved
    assert "milk" in g.nodes["t:milk"]["keyword_terms"]
    assert g.nodes["t:milk"]["chunk_count"] == 4


def test_export_graphml_returns_str_path_for_str_input(tmp_path) -> None:
    fs = _fs_with_graph()
    p = str(tmp_path / "s.graphml")
    out = fs.export_graphml(p, facet="foods")
    assert str(out) == p
