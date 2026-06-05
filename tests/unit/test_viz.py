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
import re

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
    # No external cose-bilkent dependency (that was the blank-canvas bug).
    assert "cose-bilkent" not in html
    # Embedded elements should be valid JSON inside the page.
    start = html.find("const elements = ") + len("const elements = ")
    end = html.find(";\n", start)
    payload = html[start:end]
    elements = json.loads(payload)
    assert any(el["data"].get("source") for el in elements)  # at least one edge


def test_cytoscape_layout_is_concentric_for_l1(fs_with_entities: FoodScholar) -> None:
    """L1 entity neighborhood → concentric layout, with ranks stamped on nodes."""
    html = fs_with_entities.viz.entity_neighborhood("FOODON:03309927").render("cytoscape")
    layout = _extract_layout(html)
    assert layout["name"] == "concentric"
    # Anchor entity, chunks, co-entities → ranks 3, 2, 1.
    elements = _extract_elements(html)
    ranks_by_kind: dict[str, set[int]] = {}
    for el in elements:
        d = el["data"]
        if "concentric_rank" in d:
            ranks_by_kind.setdefault(d["kind"], set()).add(d["concentric_rank"])
    # Anchor entity (is_anchor=True) → 3; non-anchor entities → 1; chunks → 2.
    anchor_ranks = {d["concentric_rank"] for el in elements
                    if el["data"].get("anchor") for d in [el["data"]]}
    assert anchor_ranks == {3}
    assert ranks_by_kind.get("chunk") == {2}


def test_cytoscape_layout_per_level() -> None:
    """Each level picks the right layout family."""
    from foodscholar.viz.renderers.cytoscape_renderer import _layout_for_level

    assert _layout_for_level("L0")["name"] == "grid"
    assert _layout_for_level("L1")["name"] == "concentric"
    assert _layout_for_level("L2")["name"] == "cose"
    assert _layout_for_level("L3")["name"] == "breadthfirst"
    assert _layout_for_level("L4")["name"] == "breadthfirst"
    # `avoidOverlap` is true everywhere it's supported.
    for level in ("L0", "L1", "L3", "L4"):
        assert _layout_for_level(level).get("avoidOverlap") is True


def _extract_layout(html: str) -> dict:
    """Pull `const layoutConfig = {...};` from the page."""
    import re

    m = re.search(r"const layoutConfig = (\{.*?\});", html, re.DOTALL)
    assert m, "couldn't find layoutConfig in the rendered HTML"
    return json.loads(m.group(1))


def _extract_elements(html: str) -> list:
    import re

    m = re.search(r"const elements = (\[.*?\]);\nconst layoutConfig", html, re.DOTALL)
    assert m, "couldn't find elements in the rendered HTML"
    return json.loads(m.group(1))


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


# ------------------------------------------------------- layer_a_tree + tree


def _populate_tree_fs() -> FoodScholar:
    """Two shelves (parent 'dairy' eligible, child 'cow_milk' eligible) + a
    sub-threshold sibling 'rare' — plus one theme of each origin on cow_milk."""
    fs = FoodScholar.in_memory()
    fs.config.layer_b.min_chunks_per_shelf = 50
    fs.graph.add_shelf(shelf_id="dairy", label="dairy", facet="foods", depth=1,
                       chunk_count=3120, support_direct=900, support_lifted=2220)
    fs.graph.add_shelf(shelf_id="cow_milk", label="cow milk", facet="foods", depth=2,
                       parent_shelf_id="dairy", foodon_id="FOODON:1",
                       chunk_count=820, support_direct=540, support_lifted=280)
    fs.graph.add_shelf(shelf_id="rare", label="rare food", facet="foods", depth=2,
                       parent_shelf_id="dairy", chunk_count=12)
    for tid, label, pass_ in [
        ("t-merged", "calcium & bone health", "merged"),
        ("t-sim", "lactose intolerance", "global_similarity"),
        ("t-rel", "fermentation", "relatedness"),
    ]:
        fs.graph.add_theme(theme_id=tid, label=label, shelf_ids=["cow_milk"],
                           discovered_by="leiden", discovery_version="v0.2",
                           facet="foods", discovery_pass=pass_, chunk_count=100,
                           keyword_terms=["k1", "k2"])
    return fs


def test_layer_a_tree_nodes_edges_and_buckets() -> None:
    fs = _populate_tree_fs()
    g = vb.layer_a_tree(fs, "foods")

    ids = {n.id for n in g.nodes}
    assert ids == {"dairy", "cow_milk", "rare"}
    edges = {(e.source, e.target) for e in g.edges if e.kind == "parent_of"}
    assert edges == {("dairy", "cow_milk"), ("dairy", "rare")}

    cow = next(n for n in g.nodes if n.id == "cow_milk")
    assert cow.attrs["eligible"] is True
    assert cow.attrs["chunk_count"] == 820
    assert cow.attrs["support_direct"] == 540
    assert [t["label"] for t in cow.attrs["themes"]["merged"]] == ["calcium & bone health"]
    assert [t["label"] for t in cow.attrs["themes"]["global_similarity"]] == ["lactose intolerance"]
    assert [t["label"] for t in cow.attrs["themes"]["relatedness"]] == ["fermentation"]
    assert cow.attrs["themes"]["merged"][0]["keyword_terms"] == ["k1", "k2"]

    rare = next(n for n in g.nodes if n.id == "rare")
    assert rare.attrs["eligible"] is False
    assert rare.attrs["themes"] == {"merged": [], "global_similarity": [], "relatedness": []}

    assert g.attrs["n_shelves"] == 3
    assert g.attrs["n_eligible"] == 2
    assert g.attrs["n_themes"] == 3


def _populate_tree_fs_with_chunks() -> FoodScholar:
    """`_populate_tree_fs` plus three chunks attached to the cow_milk shelf, so
    the per-shelf Terms / Entities / Sources tabs have data to surface."""
    from foodscholar.io.chunk import Chunk

    fs = _populate_tree_fs()
    fs.upsert_chunks([
        Chunk(chunk_id="c1", text="calcium and bone health in milk and dairy products",
              source_doc_id="doc-A", source_type="textbook", section_type="textbook",
              year=2019, shelf_ids=["cow_milk"], foodon_ids=["FOODON:1", "FOODON:2"]),
        Chunk(chunk_id="c2", text="lactose intolerance and fermentation of milk",
              source_doc_id="doc-A", source_type="textbook", section_type="textbook",
              year=2019, shelf_ids=["cow_milk"], foodon_ids=["FOODON:1"]),
        Chunk(chunk_id="c3", text="bone density and calcium absorption from dairy",
              source_doc_id="doc-B", source_type="guide", section_type="guideline",
              shelf_ids=["cow_milk"], foodon_ids=["FOODON:2"]),
    ])
    return fs


def test_layer_a_tree_terms_entities_sources() -> None:
    fs = _populate_tree_fs_with_chunks()
    g = vb.layer_a_tree(fs, "foods")
    cow = next(n for n in g.nodes if n.id == "cow_milk")

    terms = {t["term"] for t in cow.attrs["terms"]}
    assert "calcium" in terms and "bone" in terms
    assert "and" not in terms and "in" not in terms  # stopwords filtered
    assert all({"term", "count"} <= set(t) for t in cow.attrs["terms"])

    ents = {e["id"]: e for e in cow.attrs["entities"]}
    assert ents["FOODON:1"]["count"] == 2 and ents["FOODON:2"]["count"] == 2
    assert all("label" in e for e in cow.attrs["entities"])

    srcs = {s["doc_id"]: s for s in cow.attrs["sources"]}
    assert srcs["doc-A"]["count"] == 2 and srcs["doc-B"]["count"] == 1
    assert srcs["doc-A"]["source_type"] == "textbook"

    # A shelf with no directly-attached chunks carries empty detail lists.
    dairy = next(n for n in g.nodes if n.id == "dairy")
    assert dairy.attrs["terms"] == []
    assert dairy.attrs["entities"] == []
    assert dairy.attrs["sources"] == []


def test_tree_renderer_has_detail_tabs() -> None:
    fs = _populate_tree_fs_with_chunks()
    g = vb.layer_a_tree(fs, "foods")
    from foodscholar.viz.renderers.tree_renderer import TreeRenderer

    html = TreeRenderer().render(g)
    for tab in ("Topics", "Terms", "Entities", "Sources"):
        assert tab in html
    assert "http://" not in html and "https://" not in html  # still self-contained
    m = re.search(r"const TREE_DATA = (\{.*?\});", html, re.DOTALL)
    data = json.loads(m.group(1))
    cow = _find_node(data["roots"], "cow_milk")
    assert cow["terms"] and cow["entities"] and cow["sources"]


def _find_node(nodes, node_id):
    for n in nodes:
        if n["id"] == node_id:
            return n
        hit = _find_node(n.get("children", []), node_id)
        if hit:
            return hit
    return None


def test_layer_a_tree_empty_state_when_no_shelves() -> None:
    fs = FoodScholar.in_memory()
    g = vb.layer_a_tree(fs, "foods")
    assert len(g.nodes) == 0
    assert g.attrs["n_shelves"] == 0


def test_tree_renderer_emits_self_contained_html() -> None:
    fs = _populate_tree_fs()
    g = vb.layer_a_tree(fs, "foods")
    from foodscholar.viz.renderers.tree_renderer import TreeRenderer

    html = TreeRenderer().render(g)
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert "http://" not in html and "https://" not in html  # no external deps

    m = re.search(r"const TREE_DATA = (\{.*?\});", html, re.DOTALL)
    assert m, "embedded TREE_DATA not found"
    data = json.loads(m.group(1))
    roots = data["roots"]
    assert [r["id"] for r in roots] == ["dairy"]
    child_ids = {c["id"] for c in roots[0]["children"]}
    assert child_ids == {"cow_milk", "rare"}
    for origin in ("merged", "global_similarity", "relatedness"):
        assert origin in html


def test_tree_renderer_writes_file(tmp_path) -> None:
    fs = _populate_tree_fs()
    g = vb.layer_a_tree(fs, "foods")
    from foodscholar.viz.renderers.tree_renderer import TreeRenderer

    out = tmp_path / "tree.html"
    returned = TreeRenderer().render(g, output=out)
    assert returned == out
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_fs_viz_layer_a_tree_render_via_facade() -> None:
    fs = _populate_tree_fs()
    rg = fs.viz.layer_a_tree("foods")
    assert isinstance(rg, RenderableGraph)
    html = rg.render("tree")
    assert "<!DOCTYPE html>" in html
    assert "cow milk" in html
