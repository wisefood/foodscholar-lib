"""Per-node BERTopic explorer: walks shelves (any/all facets), fits topics per
node, surfaces the Layer C card, and renders collapsible HTML."""

from __future__ import annotations

from foodscholar import FoodScholar
from foodscholar.io.chunk import Chunk
from foodscholar.io.graph import Card, Shelf, Theme
from foodscholar.viz.bertopic_explorer import build_pernode_explorer


def _chunk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, text=f"text {cid}", source_doc_id="d",
                 source_type="abstract", section_type="abstract",
                 embedding=[1.0, 0.0, 0.0], embedding_model="m")


def _fs_two_facets() -> FoodScholar:
    fs = FoodScholar.in_memory()
    gs, cs = fs.graph_store, fs.chunk_store
    gs.upsert_shelves([
        Shelf(shelf_id="s:dairy", label="dairy", facet="foods", depth=1, chunk_count=4),
        Shelf(shelf_id="s:milk", label="milk", facet="foods", depth=2,
              parent_shelf_id="s:dairy", chunk_count=2),
        Shelf(shelf_id="s:fiber", label="fiber", facet="nutrients", depth=1,
              chunk_count=2),
    ])
    cs.upsert([_chunk(f"c{i}") for i in range(6)])
    gs.attach_chunks_to_shelf("s:dairy", [("c0", []), ("c1", [])])
    gs.attach_chunks_to_shelf("s:milk", [("c2", []), ("c3", [])])
    gs.attach_chunks_to_shelf("s:fiber", [("c4", []), ("c5", [])])
    # a theme + card on the dairy shelf (so the explorer can surface a card)
    gs.upsert_themes([Theme(theme_id="t:dairy", label="milk calcium",
                            shelf_ids=["s:dairy"], chunk_count=2,
                            discovered_by="bertopic", discovery_version="v",
                            facet="foods", discovery_pass="global_similarity",
                            keyword_terms=["milk", "calcium"])])
    gs.upsert_cards([Card(card_id="c:dairy", target_id="t:dairy", target_type="theme",
                          title="Milk & calcium", summary="Milk has calcium.",
                          evidence_quality="high", cited_chunk_ids=["c0"],
                          llm_model="m", prompt_version="v1")])
    return fs


def _stub_topics(monkeypatch) -> None:
    # one topic containing all of a node's chunks (deterministic, no real bertopic)
    monkeypatch.setattr(
        "foodscholar.viz.bertopic_explorer.run_bertopic",
        lambda ids, store, cfg: [set(ids)] if ids else [],
    )


def test_explorer_walks_single_facet(monkeypatch) -> None:
    _stub_topics(monkeypatch)
    exp = build_pernode_explorer(_fs_two_facets(), facet="foods", min_chunks=1)
    labels = {n["label"] for root in exp.roots for n in _flatten(root)}
    assert "dairy" in labels and "milk" in labels
    assert "fiber" not in labels  # nutrients facet excluded


def test_explorer_all_facets(monkeypatch) -> None:
    _stub_topics(monkeypatch)
    exp = build_pernode_explorer(_fs_two_facets(), facet=None, min_chunks=1)
    facets = {root["facet"] for root in exp.roots}
    assert "foods" in facets and "nutrients" in facets


def test_node_surfaces_topics_and_card(monkeypatch) -> None:
    _stub_topics(monkeypatch)
    exp = build_pernode_explorer(_fs_two_facets(), facet="foods", min_chunks=1)
    dairy = next(n for root in exp.roots for n in _flatten(root)
                 if n["shelf_id"] == "s:dairy")
    assert dairy["topics"], "expected at least one BERTopic topic"
    assert dairy["card"] is not None
    assert dairy["card"]["title"] == "Milk & calcium"
    assert dairy["card"]["evidence_quality"] == "high"


def test_render_writes_html(monkeypatch, tmp_path) -> None:
    _stub_topics(monkeypatch)
    exp = build_pernode_explorer(_fs_two_facets(), facet=None, min_chunks=1)
    out = tmp_path / "explorer.html"
    path = exp.render(output=out)
    assert path == out
    html = out.read_text()
    assert "<!DOCTYPE html>" in html
    assert "dairy" in html and "fiber" in html        # both facets present
    assert "Milk &amp; calcium" in html or "Milk & calcium" in html  # card shown


def test_viz_method_returns_explorer(monkeypatch, tmp_path) -> None:
    _stub_topics(monkeypatch)
    fs = _fs_two_facets()
    exp = fs.viz.bertopic_pernode(facet="foods", min_chunks=1)
    out = exp.render(output=tmp_path / "e.html")
    assert out.exists()


def _flatten(node: dict) -> list[dict]:
    out = [node]
    for c in node.get("children", []):
        out.extend(_flatten(c))
    return out
