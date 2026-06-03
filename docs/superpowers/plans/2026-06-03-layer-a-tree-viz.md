# Layer A interactive tree + per-shelf Pass 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an interactive, self-contained HTML tree of the Layer A `foods` shelf hierarchy where clicking a shelf reveals its Layer B themes grouped by origin (merged / global_similarity / relatedness), and re-run Layer B in per-shelf Pass-1 mode so a real `merged` bucket exists.

**Architecture:** Extend the existing `viz` framework (`builder → VizGraph → renderer`). A new `builder.layer_a_tree` produces a `VizGraph` of shelf nodes (themes-by-origin in `attrs`, `parent_of` edges). A new `TreeRenderer` (backend name `"tree"`) serializes that to a nested-tree JSON baked into a vanilla-JS two-pane HTML file. A facade method `fs.viz.layer_a_tree(facet)` matches the repo idiom. The per-shelf Pass-1 rebuild is a runtime config toggle + `fs.build_layer_b`.

**Tech Stack:** Python 3, Pydantic v2, pytest, vanilla HTML/CSS/JS (no JS deps), Neo4j (live data), the foodscholar `viz` package.

---

## File structure

- **Create** `src/foodscholar/viz/renderers/tree_renderer.py` — `TreeRenderer(Renderer)`, `name="tree"`. Flat `VizGraph` → nested tree JSON → self-contained HTML. ~180 lines incl. template.
- **Modify** `src/foodscholar/viz/builder.py` — add `layer_a_tree(fs, facet="foods") -> VizGraph`.
- **Modify** `src/foodscholar/viz/renderers/__init__.py` — add lazy `tree()` constructor.
- **Modify** `src/foodscholar/viz/view.py` — add `"tree"` to `RendererName`, the render factory, and a `VizView.layer_a_tree(facet)` facade.
- **Modify** `tests/unit/test_viz.py` — tests for builder + renderer.
- **Create** `scripts/build_layer_a_tree.py` — runtime orchestration: toggle `pass1_mode="per_shelf"`, `fs.build_layer_b("foods")`, render to `data/viz/layer_a_tree_foods.html`. (Run manually; needs live stores.)

Reference models (already exist — do not redefine):
- `Shelf` fields: `shelf_id, label, display_label, facet, depth, foodon_id, parent_shelf_id, chunk_count, support_direct, support_lifted, see_also`.
- `Theme` fields: `theme_id, label, shelf_ids, chunk_count, discovered_by, discovery_version, facet, discovery_pass, keyword_terms, foodon_id_signature, ...`.
- Handle API: `fs.graph.shelves(facet=...)` → `list[ShelfHandle]`; `sh.model` (a `Shelf`), `sh.themes()` → `list[ThemeHandle]`; `th.model` (a `Theme`).
- `VizNode(id, label, kind, weight, facet, attrs)`, `VizEdge(source, target, kind, weight, attrs)`, `VizGraph(title, nodes, edges, level, attrs)`.

---

## Task 1: `builder.layer_a_tree`

**Files:**
- Modify: `src/foodscholar/viz/builder.py`
- Test: `tests/unit/test_viz.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_viz.py`:

```python
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
    # parent_of edges: dairy->cow_milk, dairy->rare
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


def test_layer_a_tree_empty_state_when_no_shelves() -> None:
    fs = FoodScholar.in_memory()
    g = vb.layer_a_tree(fs, "foods")
    assert len(g.nodes) == 0
    assert g.attrs["n_shelves"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_viz.py::test_layer_a_tree_nodes_edges_and_buckets -v`
Expected: FAIL with `AttributeError: module 'foodscholar.viz.builder' has no attribute 'layer_a_tree'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/foodscholar/viz/builder.py` (near `backbone`). Confirm the top of the file imports `VizEdge, VizGraph, VizNode` from `foodscholar.viz.model`; add them if missing.

```python
def layer_a_tree(fs: FoodScholar, facet: str = "foods") -> VizGraph:
    """Full Layer A shelf tree for a facet, with each shelf's Layer B themes
    grouped by `discovery_pass` in node attrs. Sub-threshold shelves (below
    `min_chunks_per_shelf`) are kept but flagged `eligible=False` and carry no
    themes. One `parent_of` edge per `parent_shelf_id`.
    """
    min_chunks = fs.config.layer_b.min_chunks_per_shelf
    shelves = fs.graph.shelves(facet=facet)

    nodes: list[VizNode] = []
    edges: list[VizEdge] = []
    n_eligible = 0
    n_themes = 0

    for sh in shelves:
        s = sh.model
        buckets: dict[str, list[dict[str, object]]] = {
            "merged": [], "global_similarity": [], "relatedness": [],
        }
        for th in sh.themes():
            t = th.model
            bucket = buckets.get(t.discovery_pass)
            if bucket is None:  # unknown pass — skip defensively
                continue
            bucket.append({
                "theme_id": t.theme_id,
                "label": t.label,
                "chunk_count": t.chunk_count,
                "keyword_terms": list(t.keyword_terms),
                "discovery_pass": t.discovery_pass,
            })
            n_themes += 1

        eligible = s.chunk_count >= min_chunks
        if eligible:
            n_eligible += 1

        nodes.append(VizNode(
            id=s.shelf_id,
            label=s.display_label or s.label,
            kind="shelf",
            weight=float(s.chunk_count),
            facet=facet,
            attrs={
                "chunk_count": s.chunk_count,
                "support_direct": s.support_direct,
                "support_lifted": s.support_lifted,
                "depth": s.depth,
                "foodon_id": s.foodon_id,
                "eligible": eligible,
                "themes": buckets,
            },
        ))
        if s.parent_shelf_id is not None:
            edges.append(VizEdge(
                source=s.parent_shelf_id, target=s.shelf_id, kind="parent_of",
            ))

    return VizGraph(
        title=f"Layer A tree — {facet}",
        nodes=nodes,
        edges=edges,
        level="L3",
        attrs={
            "facet": facet,
            "min_chunks_per_shelf": min_chunks,
            "n_shelves": len(nodes),
            "n_eligible": n_eligible,
            "n_themes": n_themes,
        },
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_viz.py::test_layer_a_tree_nodes_edges_and_buckets tests/unit/test_viz.py::test_layer_a_tree_empty_state_when_no_shelves -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/viz/builder.py tests/unit/test_viz.py
git commit -m "feat(viz): layer_a_tree builder — shelf tree with themes-by-origin"
```

---

## Task 2: `TreeRenderer` (backend `"tree"`)

**Files:**
- Create: `src/foodscholar/viz/renderers/tree_renderer.py`
- Modify: `src/foodscholar/viz/renderers/__init__.py`
- Modify: `src/foodscholar/viz/view.py`
- Test: `tests/unit/test_viz.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_viz.py` (reuses `_populate_tree_fs` from Task 1):

```python
def test_tree_renderer_emits_self_contained_html() -> None:
    fs = _populate_tree_fs()
    g = vb.layer_a_tree(fs, "foods")
    from foodscholar.viz.renderers.tree_renderer import TreeRenderer

    html = TreeRenderer().render(g)
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert "http://" not in html and "https://" not in html  # no external deps
    # embedded JSON tree round-trips and nests cow_milk + rare under dairy
    import re
    m = re.search(r"const TREE_DATA = (\{.*?\});", html, re.DOTALL)
    assert m, "embedded TREE_DATA not found"
    data = json.loads(m.group(1))
    roots = data["roots"]
    assert [r["id"] for r in roots] == ["dairy"]
    child_ids = {c["id"] for c in roots[0]["children"]}
    assert child_ids == {"cow_milk", "rare"}
    # all three origin classes appear in the markup/data
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
    html = fs.viz.layer_a_tree("foods").render("tree")
    assert "<!DOCTYPE html>" in html
    assert "cow milk" in html


def test_render_unknown_backend_still_raises(fs_with_entities: FoodScholar) -> None:
    with pytest.raises(ValueError, match="unknown viz backend"):
        fs_with_entities.viz.backbone().render("nope")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_viz.py::test_tree_renderer_emits_self_contained_html -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'foodscholar.viz.renderers.tree_renderer'`.

- [ ] **Step 3a: Create the renderer**

Create `src/foodscholar/viz/renderers/tree_renderer.py`:

```python
"""TreeRenderer: a `VizGraph` of shelf nodes (`parent_of` edges) → a
self-contained two-pane HTML tree. Left = collapsible shelf hierarchy with
counts; right = the clicked shelf's themes grouped by discovery origin.

No external assets: all data is baked in as JSON and the JS is vanilla.
Honors the base contract — return the HTML string when `output is None`,
else write the file and return its `Path`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from foodscholar.viz.model import VizGraph
from foodscholar.viz.renderers.base import Renderer

ORIGIN_COLORS = {
    "merged": "#16A34A",            # green — both passes agreed
    "global_similarity": "#2563EB",  # blue — embedding-only
    "relatedness": "#D97706",       # amber — entity-only
}
ORIGIN_LABELS = {
    "merged": "Merged",
    "global_similarity": "Similarity",
    "relatedness": "Relatedness",
}


class TreeRenderer(Renderer):
    name = "tree"

    def render(self, graph: VizGraph, *, output: str | Path | None = None) -> Any:
        tree = _to_tree(graph)
        html = _HTML_TEMPLATE.format(
            title=_esc(graph.title),
            tree_json=json.dumps(tree),
            origin_colors_json=json.dumps(ORIGIN_COLORS),
            origin_labels_json=json.dumps(ORIGIN_LABELS),
        )
        if output is None:
            return html
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return out


def _to_tree(graph: VizGraph) -> dict[str, Any]:
    """Reshape flat nodes/`parent_of` edges into nested roots+children."""
    by_id: dict[str, dict[str, Any]] = {}
    for n in graph.nodes:
        by_id[n.id] = {
            "id": n.id,
            "label": n.label,
            "chunk_count": n.attrs.get("chunk_count", 0),
            "support_direct": n.attrs.get("support_direct", 0),
            "support_lifted": n.attrs.get("support_lifted", 0),
            "depth": n.attrs.get("depth", 0),
            "foodon_id": n.attrs.get("foodon_id"),
            "eligible": n.attrs.get("eligible", False),
            "themes": n.attrs.get("themes", {}),
            "children": [],
        }
    child_ids: set[str] = set()
    for e in graph.edges:
        if e.kind != "parent_of":
            continue
        parent, child = by_id.get(e.source), by_id.get(e.target)
        if parent is not None and child is not None:
            parent["children"].append(child)
            child_ids.add(e.target)
    roots = [node for nid, node in by_id.items() if nid not in child_ids]
    # stable order: by chunk_count desc then id
    def _sort(nodes: list[dict[str, Any]]) -> None:
        nodes.sort(key=lambda d: (-d["chunk_count"], d["id"]))
        for d in nodes:
            _sort(d["children"])
    _sort(roots)
    return {"meta": graph.attrs, "roots": roots}


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font: 14px/1.45 system-ui, sans-serif; color: #111; }}
  header {{ padding: 10px 16px; background: #0f172a; color: #fff; }}
  header b {{ font-size: 15px; }} header .counts {{ opacity: .8; margin-left: 10px; }}
  .wrap {{ display: flex; height: calc(100vh - 44px); }}
  .tree {{ width: 46%; overflow: auto; border-right: 1px solid #e5e7eb; padding: 8px 0; }}
  .detail {{ flex: 1; overflow: auto; padding: 16px 20px; }}
  .row {{ display: flex; align-items: center; gap: 6px; padding: 2px 8px;
          cursor: pointer; white-space: nowrap; }}
  .row:hover {{ background: #f1f5f9; }} .row.sel {{ background: #dbeafe; }}
  .row.sub {{ color: #9ca3af; }}
  .caret {{ width: 12px; display: inline-block; color: #64748b; }}
  .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
  .cc {{ color: #475569; font-variant-numeric: tabular-nums; }}
  .tn {{ color: #1e3a8a; font-variant-numeric: tabular-nums; }}
  .kids {{ margin-left: 14px; }} .hidden {{ display: none; }}
  h2 {{ margin: 0 0 2px; }} .sub-meta {{ color: #64748b; margin-bottom: 14px; }}
  .origin {{ margin: 14px 0 6px; font-weight: 600; }}
  .theme {{ padding: 6px 10px; border-left: 3px solid #ccc; margin: 4px 0; background: #f8fafc; }}
  .theme .kw {{ color: #64748b; font-size: 12px; }}
  .filters button {{ margin-right: 4px; border: 1px solid #cbd5e1; background: #fff;
                     border-radius: 4px; padding: 2px 8px; cursor: pointer; }}
  .filters button.on {{ background: #0f172a; color: #fff; }}
</style></head><body>
<header><b id="h-title"></b><span class="counts" id="h-counts"></span></header>
<div class="wrap">
  <div class="tree" id="tree"></div>
  <div class="detail" id="detail"><p style="color:#64748b">Select a shelf on the left.</p></div>
</div>
<script>
const TREE_DATA = {tree_json};
const ORIGIN_COLORS = {origin_colors_json};
const ORIGIN_LABELS = {origin_labels_json};
let activeFilter = "all";

function esc(s) {{ return String(s == null ? "" : s)
  .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }}

function themeCount(node) {{
  const t = node.themes || {{}};
  return (t.merged||[]).length + (t.global_similarity||[]).length + (t.relatedness||[]).length;
}}

function renderTree() {{
  const m = TREE_DATA.meta || {{}};
  document.getElementById("h-title").textContent = "Layer A — " + (m.facet || "");
  document.getElementById("h-counts").textContent =
    (m.n_shelves||0) + " shelves · " + (m.n_eligible||0) + " eligible · " +
    (m.n_themes||0) + " themes";
  const root = document.getElementById("tree");
  root.innerHTML = "";
  TREE_DATA.roots.forEach(n => root.appendChild(rowFor(n, true)));
}}

function rowFor(node, open) {{
  const box = document.createElement("div");
  const row = document.createElement("div");
  row.className = "row" + (node.eligible ? "" : " sub");
  const hasKids = node.children && node.children.length;
  const tc = themeCount(node);
  const caret = hasKids ? (open ? "▾" : "▸") : " ";
  row.innerHTML =
    '<span class="caret">' + caret + '</span>' +
    '<span class="dot" style="background:' +
      (node.eligible ? "#16A34A" : "#cbd5e1") + '"></span>' +
    '<span class="lbl">' + esc(node.label) + '</span>' +
    '<span class="cc">' + node.chunk_count + '</span>' +
    (tc ? '<span class="tn">[' + tc + ']</span>' : '');
  let kids = null;
  if (hasKids) {{
    kids = document.createElement("div");
    kids.className = "kids" + (open ? "" : " hidden");
    node.children.forEach(c => kids.appendChild(rowFor(c, false)));
  }}
  row.onclick = (ev) => {{
    ev.stopPropagation();
    if (hasKids && ev.target.classList.contains("caret")) {{
      kids.classList.toggle("hidden");
      ev.target.textContent = kids.classList.contains("hidden") ? "▸" : "▾";
      return;
    }}
    document.querySelectorAll(".row.sel").forEach(r => r.classList.remove("sel"));
    row.classList.add("sel");
    showDetail(node);
  }};
  box.appendChild(row);
  if (kids) box.appendChild(kids);
  return box;
}}

function showDetail(node) {{
  const d = document.getElementById("detail");
  if (!node.eligible) {{
    d.innerHTML = "<h2>" + esc(node.label) + "</h2>" +
      '<div class="sub-meta">' + (node.foodon_id ? esc(node.foodon_id) + " · " : "") +
      "depth " + node.depth + " · " + node.chunk_count + " chunks</div>" +
      "<p style='color:#9ca3af'>Below the chunk threshold — no themes.</p>";
    return;
  }}
  let html = "<h2>" + esc(node.label) + "</h2>" +
    '<div class="sub-meta">' + (node.foodon_id ? esc(node.foodon_id) + " · " : "") +
    "depth " + node.depth + " · " + node.chunk_count + " chunks · direct " +
    node.support_direct + " / lifted " + node.support_lifted + "</div>";
  html += '<div class="filters">';
  ["all", "merged", "global_similarity", "relatedness"].forEach(f => {{
    html += '<button class="' + (activeFilter === f ? "on" : "") +
      '" onclick="setFilter(\\'' + f + '\\')">' +
      (f === "all" ? "all" : ORIGIN_LABELS[f]) + "</button>";
  }});
  html += "</div>";
  ["merged", "global_similarity", "relatedness"].forEach(origin => {{
    if (activeFilter !== "all" && activeFilter !== origin) return;
    const list = (node.themes || {{}})[origin] || [];
    html += '<div class="origin" style="color:' + ORIGIN_COLORS[origin] + '">' +
      ORIGIN_LABELS[origin].toUpperCase() + " (" + list.length + ")</div>";
    list.forEach(t => {{
      html += '<div class="theme" style="border-left-color:' + ORIGIN_COLORS[origin] + '">' +
        esc(t.label) + ' <span class="cc">' + t.chunk_count + "ch</span>" +
        (t.keyword_terms && t.keyword_terms.length ?
          '<div class="kw">' + esc(t.keyword_terms.join(", ")) + "</div>" : "") + "</div>";
    }});
  }});
  d.innerHTML = html;
  d.dataset.nodeId = node.id;
}}

let _lastNode = null;
const _origShowDetail = showDetail;
showDetail = function(n) {{ _lastNode = n; _origShowDetail(n); }};
function setFilter(f) {{ activeFilter = f; if (_lastNode) _origShowDetail(_lastNode); }}

renderTree();
</script></body></html>"""
```

- [ ] **Step 3b: Register the `tree` backend constructor**

Add to `src/foodscholar/viz/renderers/__init__.py`:

```python
def tree(**kwargs) -> Renderer:  # type: ignore[no-untyped-def]
    """Lazy constructor for `TreeRenderer`. Self-contained HTML, no deps."""
    from foodscholar.viz.renderers.tree_renderer import TreeRenderer

    return TreeRenderer()
```

- [ ] **Step 3c: Wire backend + facade in `view.py`**

In `src/foodscholar/viz/view.py`:

1. Extend the `RendererName` literal (line 26):

```python
RendererName = Literal["pyvis", "cytoscape", "graphviz", "matplotlib", "tree"]
```

2. Add `"tree"` to the `factory` dict inside `RenderableGraph.render` (after the `"matplotlib"` entry):

```python
            "matplotlib": _renderers.matplotlib,
            "tree": _renderers.tree,
```

3. Add the facade method to `VizView` (next to `backbone`):

```python
    def layer_a_tree(self, facet: str = "foods") -> RenderableGraph:
        """Full Layer A shelf tree for a facet, themes grouped by origin.
        Best rendered with the `"tree"` backend."""
        return RenderableGraph(builder.layer_a_tree(self._fs, facet))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_viz.py -k "tree or unknown_backend" -v`
Expected: PASS (`test_tree_renderer_emits_self_contained_html`, `test_tree_renderer_writes_file`, `test_fs_viz_layer_a_tree_render_via_facade`, `test_render_unknown_backend_still_raises`).

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/viz/renderers/tree_renderer.py src/foodscholar/viz/renderers/__init__.py src/foodscholar/viz/view.py tests/unit/test_viz.py
git commit -m "feat(viz): tree renderer + fs.viz.layer_a_tree facade"
```

---

## Task 3: Full suite + lint gate

**Files:** none (verification only).

- [ ] **Step 1: Run the full unit suite**

Run: `pytest tests/unit -q`
Expected: PASS (all green, including the four new tests).

- [ ] **Step 2: Lint**

Run: `ruff check src/foodscholar/viz tests/unit/test_viz.py`
Expected: no errors. Fix any reported issues, re-run until clean.

- [ ] **Step 3: Commit (only if lint changes were made)**

```bash
git add -A && git commit -m "style(viz): ruff clean for layer_a_tree"
```

---

## Task 4: Per-shelf Pass-1 rebuild + emit the artifact

**Files:**
- Create: `scripts/build_layer_a_tree.py`

> This task wires the runtime orchestration. **Running it requires live Neo4j + Elasticsearch with embeddings** (the same stores `build_graph.ipynb` uses) and takes ~10–15 min + LLM cost, so the actual execution is a manual step for the user — not something the plan executor runs. The executor only writes the script and confirms it imports cleanly.

- [ ] **Step 1: Create the script**

Create `scripts/build_layer_a_tree.py`:

```python
"""Re-run Layer B for `foods` in per-shelf Pass-1 mode, then render the
interactive Layer A tree to data/viz/layer_a_tree_foods.html.

Per-shelf Pass 1 yields shelf-scoped similarity candidates that can align with
the per-shelf relatedness pass, so a real `merged` origin bucket appears
(global mode produced zero). Run from the repo root with live stores up:

    python scripts/build_layer_a_tree.py
"""

from __future__ import annotations

from pathlib import Path

from foodscholar import FoodScholar


def main() -> None:
    fs = FoodScholar.from_config()  # same wiring build_graph.ipynb uses

    # Switch Pass 1 to per-shelf and rebuild Layer B for foods (replaces themes).
    fs.config.layer_b.pass1_mode = "per_shelf"
    artifact = fs.build_layer_b(facet="foods", dry_run=False)
    print("Layer B rebuilt:", artifact)

    by_pass = {"merged": 0, "global_similarity": 0, "relatedness": 0}
    for t in fs.graph_store.list_themes():
        by_pass[t.discovery_pass] = by_pass.get(t.discovery_pass, 0) + 1
    print("themes by pass:", by_pass)

    out = Path("data/viz/layer_a_tree_foods.html")
    path = fs.viz.layer_a_tree("foods").render("tree", output=out)
    print("wrote", path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm it imports without running the live build**

Run: `python -c "import ast; ast.parse(open('scripts/build_layer_a_tree.py').read()); print('parse-ok')"`
Expected: `parse-ok`.

(Do **not** run `python scripts/build_layer_a_tree.py` in the executor — it needs live stores. Leave that for the user.)

- [ ] **Step 3: Commit**

```bash
git add scripts/build_layer_a_tree.py
git commit -m "feat(layer_b): per-shelf Pass-1 rebuild + Layer A tree artifact script"
```

- [ ] **Step 4: Hand back to the user**

Tell the user to run, with Neo4j + Elasticsearch up and embeddings available:

```bash
python scripts/build_layer_a_tree.py
```

Then open `data/viz/layer_a_tree_foods.html` in a browser. Confirm the `themes by pass` printout shows a non-zero `merged` count, and that clicking shelves shows the three origin sections.

---

## Self-review notes

- **Spec coverage:** per-shelf Pass-1 rebuild (Task 4) ✓; full tree of all shelves incl. sub-threshold flagged (Task 1, `eligible`) ✓; click→themes-by-origin (Task 2 renderer) ✓; standalone `.html` to `data/viz/` (Task 2 `output=`, Task 4 path) ✓; idiomatic `fs.viz.X().render()` facade (Task 2 step 3c) ✓; builder + renderer unit tests + full suite gate (Tasks 1–3) ✓.
- **Type consistency:** `layer_a_tree` name used identically in builder, facade, and script. `attrs["themes"]` shape `{merged, global_similarity, relatedness}` defined in Task 1 and consumed unchanged in Task 2's `_to_tree`/JS. `TreeRenderer.name == "tree"` matches the `RendererName` literal and factory key.
- **No placeholders:** every code/command step is concrete.
