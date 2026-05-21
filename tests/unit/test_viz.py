"""Tests for the viz module.

The builder is exercised against in-memory stores; the renderers we test
without their heavy deps are:

  - **Cytoscape** — pure stdlib, fully testable here.
  - **The dispatcher** — verify unknown backend raises, lazy-import errors
    bubble up with a helpful message.

The Pyvis / Graphviz / Matplotlib renderers themselves are lazy-imported and
their packages aren't in the default test env (they live in the `[viz]`
extra); we cover the ImportError path explicitly.
"""

from __future__ import annotations

import json

import pytest

from foodscholar import FoodScholar
from foodscholar.io.chunk import Chunk, EntityLink, Mention
from foodscholar.io.ontology import OntologyTerm
from foodscholar.ontology import FoodOnAPI
from foodscholar.viz import VizEdge, VizGraph, VizNode
from foodscholar.viz import builder as vb
from foodscholar.viz.view import RenderableGraph

# ----------------------------------------------------------------- model


def test_vizgraph_basic_construction() -> None:
    g = VizGraph(
        title="t",
        nodes=[VizNode(id="a", label="A", kind="entity")],
        edges=[],
        level="L1",
    )
    assert len(g) == 1
    assert g.nodes[0].id == "a"


def test_vizgraph_neighbors() -> None:
    g = VizGraph(
        title="t",
        nodes=[VizNode(id="a", label="A", kind="entity"),
               VizNode(id="b", label="B", kind="chunk"),
               VizNode(id="c", label="C", kind="entity")],
        edges=[VizEdge(source="b", target="a", kind="mentions"),
               VizEdge(source="b", target="c", kind="mentions")],
        level="L1",
    )
    assert sorted(g.neighbors("a")) == ["b"]
    assert sorted(g.neighbors("b")) == ["a", "c"]


# --------------------------------------------------------------- builders


def _mention(text: str, *, entity_type: str = "food") -> Mention:
    return Mention(
        text=text, start=0, end=len(text), score=1.0,
        ner_model_version="t", entity_type=entity_type,  # type: ignore[arg-type]
    )


def _link(mention: Mention, oid: str) -> EntityLink:
    return EntityLink(
        mention=mention, ontology_id=oid, confidence=0.9,
        method="dense", linker_version="t",
    )


def _chunk(chunk_id: str, links: list[EntityLink]) -> Chunk:
    foodon_ids: list[str] = []
    for ln in links:
        if ln.ontology_id.startswith("FOODON:") and ln.ontology_id not in foodon_ids:
            foodon_ids.append(ln.ontology_id)
    return Chunk(
        chunk_id=chunk_id,
        text=f"text about {chunk_id}",
        source_doc_id="d", source_type="abstract", section_type="abstract",
        mentions=[ln.mention for ln in links],
        entity_links=links,
        foodon_ids=foodon_ids,
    )


@pytest.fixture
def fs_with_entities() -> FoodScholar:
    fs = FoodScholar.in_memory()
    m_olive = _mention("olive oil")
    m_iron = _mention("iron", entity_type="micronutrient")
    m_uk = _mention("UK", entity_type="Country")
    chunks = [
        _chunk("c1", [
            _link(m_olive, "FOODON:03309927"),
            _link(m_iron, "CHEBI:18248"),
            _link(m_uk, "GAZ:00002637"),
        ]),
        _chunk("c2", [
            _link(m_olive, "FOODON:03309927"),
            _link(m_iron, "CHEBI:18248"),
        ]),
        _chunk("c3", [_link(_mention("apple"), "FOODON:00001141")]),
    ]
    fs.upsert_chunks(chunks)
    fs.build_entities()
    return fs


def test_entity_histogram_unfiltered(fs_with_entities: FoodScholar) -> None:
    g = vb.entity_histogram(fs_with_entities, k=10)
    assert g.level == "L0"
    assert g.edges == []
    ids = {n.id for n in g.nodes}
    assert "FOODON:03309927" in ids
    assert "CHEBI:18248" in ids


def test_entity_histogram_prefix_filter(fs_with_entities: FoodScholar) -> None:
    g = vb.entity_histogram(fs_with_entities, prefix="FOODON", k=10)
    assert all(n.attrs["prefix"] == "FOODON" for n in g.nodes)
    assert all(n.kind == "entity" for n in g.nodes)


def test_entity_neighborhood_includes_anchor_and_chunks(fs_with_entities: FoodScholar) -> None:
    g = vb.entity_neighborhood(fs_with_entities, "FOODON:03309927")
    assert g.level == "L1"
    # Anchor is present and flagged.
    [anchor] = [n for n in g.nodes if n.id == "FOODON:03309927"]
    assert anchor.attrs["is_anchor"] is True
    # c1 + c2 mention olive oil — both should be chunk nodes.
    chunk_ids = {n.id for n in g.nodes if n.kind == "chunk"}
    assert chunk_ids == {"c1", "c2"}
    # Edges from chunks point at the anchor.
    chunk_to_anchor = [
        e for e in g.edges
        if e.target == "FOODON:03309927" and e.kind == "mentions"
    ]
    assert {e.source for e in chunk_to_anchor} == {"c1", "c2"}


def test_entity_neighborhood_collects_co_entities(fs_with_entities: FoodScholar) -> None:
    g = vb.entity_neighborhood(fs_with_entities, "FOODON:03309927")
    # iron is co-mentioned with olive oil in c1 + c2 → should appear.
    co_ids = {n.id for n in g.nodes if n.kind == "entity" and not n.attrs.get("is_anchor")}
    assert "CHEBI:18248" in co_ids
    # UK appears only with olive oil in c1 — still surfaces (max_co_entities is plenty).
    assert "GAZ:00002637" in co_ids


def test_entity_neighborhood_missing_id_returns_placeholder(
    fs_with_entities: FoodScholar,
) -> None:
    g = vb.entity_neighborhood(fs_with_entities, "FOODON:DOES_NOT_EXIST")
    assert g.attrs["missing"] is True
    assert len(g.nodes) == 1
    assert g.edges == []


def test_shelf_view_empty_state_when_no_layer_a() -> None:
    fs = FoodScholar.in_memory()
    g = vb.shelf_view(fs, "s-anything")
    assert g.attrs["empty_state"] is True
    assert g.level == "L2"


def test_backbone_empty_state_when_no_shelves() -> None:
    fs = FoodScholar.in_memory()
    g = vb.backbone(fs)
    assert g.attrs["empty_state"] is True
    assert g.level == "L3"


def test_ontology_subtree_walks_ancestors_and_descendants() -> None:
    # 3-level tiny ontology: root → mid → leaf, with `mid` as anchor.
    terms = [
        OntologyTerm(id="T:root", label="root", parent_ids=(), ancestor_ids=()),
        OntologyTerm(id="T:mid", label="mid", parent_ids=("T:root",),
                     ancestor_ids=("T:root",)),
        OntologyTerm(id="T:leaf", label="leaf", parent_ids=("T:mid",),
                     ancestor_ids=("T:mid", "T:root")),
    ]
    api = FoodOnAPI(terms, prefix_filter=None)
    g = vb.ontology_subtree(api, "T:mid")
    assert g.level == "L4"
    ids = {n.id for n in g.nodes}
    assert ids == {"T:root", "T:mid", "T:leaf"}
    assert any(e.source == "T:mid" and e.target == "T:root" for e in g.edges)
    assert any(e.source == "T:leaf" and e.target == "T:mid" for e in g.edges)


def test_ontology_subtree_missing_term() -> None:
    api = FoodOnAPI([OntologyTerm(id="T:1", label="x")], prefix_filter=None)
    g = vb.ontology_subtree(api, "T:not-here")
    assert g.attrs["missing"] is True


# -------------------------------------------------------- facade view


def test_fs_viz_namespace_returns_renderable_graphs(
    fs_with_entities: FoodScholar,
) -> None:
    rg = fs_with_entities.viz.entity_neighborhood("FOODON:03309927")
    assert isinstance(rg, RenderableGraph)
    assert rg.level == "L1"
    assert len(rg) > 1


def test_fs_viz_repr_is_informative(fs_with_entities: FoodScholar) -> None:
    rg = fs_with_entities.viz.entity_histogram(k=3)
    assert "RenderableGraph" in repr(rg)
    assert "level=L0" in repr(rg)


# ---------------------------------------------------- cytoscape renderer


def test_cytoscape_renders_self_contained_html(fs_with_entities: FoodScholar) -> None:
    rg = fs_with_entities.viz.entity_neighborhood("FOODON:03309927")
    html = rg.render("cytoscape")
    assert isinstance(html, str)
    assert "<html" in html.lower()
    assert "cytoscape" in html.lower()
    # The page uses the built-in `cose` layout (no external cose-bilkent
    # dependency, which previously left the canvas blank).
    assert "name: 'cose'" in html
    assert "cose-bilkent" not in html
    # Embedded elements should be valid JSON inside the page.
    start = html.find("const elements = ") + len("const elements = ")
    end = html.find(";\n", start)
    payload = html[start:end]
    elements = json.loads(payload)
    assert any(el["data"].get("source") for el in elements)  # at least one edge


def test_cytoscape_writes_html_to_disk(
    fs_with_entities: FoodScholar, tmp_path
) -> None:
    out = tmp_path / "g.html"
    written = fs_with_entities.viz.entity_neighborhood(
        "FOODON:03309927"
    ).render("cytoscape", output=out)
    assert written == out
    assert out.exists()
    assert "<html" in out.read_text().lower()


def test_renderable_graph_rejects_unknown_backend(
    fs_with_entities: FoodScholar,
) -> None:
    rg = fs_with_entities.viz.entity_histogram(k=3)
    with pytest.raises(ValueError, match="unknown viz backend"):
        rg.render("rainbow")  # type: ignore[arg-type]


def test_pyvis_renderer_import_error_when_extra_missing(monkeypatch) -> None:
    """If `pyvis` isn't installed, the lazy import should raise a helpful message."""
    import builtins

    real_import = builtins.__import__

    def fail(name, *a, **kw):
        if name.startswith("pyvis"):
            raise ImportError("simulated no-pyvis env")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fail)
    from foodscholar.viz.renderers import pyvis_renderer

    # Reset the module's cached class to force re-import inside __init__.
    with pytest.raises(ImportError, match=r"foodscholar\[viz\]"):
        pyvis_renderer.PyvisRenderer()
