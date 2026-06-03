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
    "merged": "#16A34A",             # green — both passes agreed
    "global_similarity": "#2563EB",  # blue — embedding-only
    "relatedness": "#D97706",        # amber — entity-only
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
        html = (
            _HTML_TEMPLATE
            .replace("__TITLE__", _esc(graph.title))
            .replace("__TREE_JSON__", json.dumps(tree))
            .replace("__ORIGIN_COLORS__", json.dumps(ORIGIN_COLORS))
            .replace("__ORIGIN_LABELS__", json.dumps(ORIGIN_LABELS))
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

    def _sort(nodes: list[dict[str, Any]]) -> None:
        nodes.sort(key=lambda d: (-d["chunk_count"], d["id"]))
        for d in nodes:
            _sort(d["children"])

    _sort(roots)
    return {"meta": graph.attrs, "roots": roots}


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.45 system-ui, sans-serif; color: #111; }
  header { padding: 10px 16px; background: #0f172a; color: #fff; }
  header b { font-size: 15px; } header .counts { opacity: .8; margin-left: 10px; }
  .wrap { display: flex; height: calc(100vh - 44px); }
  .tree { width: 46%; overflow: auto; border-right: 1px solid #e5e7eb; padding: 8px 0; }
  .detail { flex: 1; overflow: auto; padding: 16px 20px; }
  .row { display: flex; align-items: center; gap: 6px; padding: 2px 8px;
         cursor: pointer; white-space: nowrap; }
  .row:hover { background: #f1f5f9; } .row.sel { background: #dbeafe; }
  .row.sub { color: #9ca3af; }
  .caret { width: 12px; display: inline-block; color: #64748b; }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .cc { color: #475569; font-variant-numeric: tabular-nums; }
  .tn { color: #1e3a8a; font-variant-numeric: tabular-nums; }
  .kids { margin-left: 14px; } .hidden { display: none; }
  h2 { margin: 0 0 2px; } .sub-meta { color: #64748b; margin-bottom: 14px; }
  .origin { margin: 14px 0 6px; font-weight: 600; }
  .theme { padding: 6px 10px; border-left: 3px solid #ccc; margin: 4px 0; background: #f8fafc; }
  .theme .kw { color: #64748b; font-size: 12px; }
  .filters { margin: 6px 0 10px; }
  .filters button { margin-right: 4px; border: 1px solid #cbd5e1; background: #fff;
                    border-radius: 4px; padding: 2px 8px; cursor: pointer; }
  .filters button.on { background: #0f172a; color: #fff; }
</style></head><body>
<header><b id="h-title"></b><span class="counts" id="h-counts"></span></header>
<div class="wrap">
  <div class="tree" id="tree"></div>
  <div class="detail" id="detail"><p style="color:#64748b">Select a shelf on the left.</p></div>
</div>
<script>
const TREE_DATA = __TREE_JSON__;
const ORIGIN_COLORS = __ORIGIN_COLORS__;
const ORIGIN_LABELS = __ORIGIN_LABELS__;
let activeFilter = "all";
let lastNode = null;

function esc(s) { return String(s == null ? "" : s)
  .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

function themeCount(node) {
  const t = node.themes || {};
  return (t.merged||[]).length + (t.global_similarity||[]).length + (t.relatedness||[]).length;
}

function renderTree() {
  const m = TREE_DATA.meta || {};
  document.getElementById("h-title").textContent = "Layer A — " + (m.facet || "");
  document.getElementById("h-counts").textContent =
    (m.n_shelves||0) + " shelves · " + (m.n_eligible||0) + " eligible · " +
    (m.n_themes||0) + " themes";
  const root = document.getElementById("tree");
  root.innerHTML = "";
  TREE_DATA.roots.forEach(n => root.appendChild(rowFor(n, true)));
}

function rowFor(node, open) {
  const box = document.createElement("div");
  const row = document.createElement("div");
  row.className = "row" + (node.eligible ? "" : " sub");
  const hasKids = node.children && node.children.length;
  const tc = themeCount(node);
  const caret = hasKids ? (open ? "▾" : "▸") : " ";
  row.innerHTML =
    '<span class="caret">' + caret + '</span>' +
    '<span class="dot" style="background:' +
      (node.eligible ? "#16A34A" : "#cbd5e1") + '"></span>' +
    '<span class="lbl">' + esc(node.label) + '</span>' +
    '<span class="cc">' + node.chunk_count + '</span>' +
    (tc ? '<span class="tn">[' + tc + ']</span>' : '');
  let kids = null;
  if (hasKids) {
    kids = document.createElement("div");
    kids.className = "kids" + (open ? "" : " hidden");
    node.children.forEach(c => kids.appendChild(rowFor(c, false)));
  }
  row.onclick = (ev) => {
    ev.stopPropagation();
    if (hasKids && ev.target.classList.contains("caret")) {
      kids.classList.toggle("hidden");
      ev.target.textContent = kids.classList.contains("hidden") ? "▸" : "▾";
      return;
    }
    document.querySelectorAll(".row.sel").forEach(r => r.classList.remove("sel"));
    row.classList.add("sel");
    showDetail(node);
  };
  box.appendChild(row);
  if (kids) box.appendChild(kids);
  return box;
}

function showDetail(node) {
  lastNode = node;
  const d = document.getElementById("detail");
  const head = "<h2>" + esc(node.label) + "</h2>" +
    '<div class="sub-meta">' + (node.foodon_id ? esc(node.foodon_id) + " · " : "") +
    "depth " + node.depth + " · " + node.chunk_count + " chunks";
  if (!node.eligible) {
    d.innerHTML = head + "</div><p style='color:#9ca3af'>Below the chunk threshold — no themes.</p>";
    return;
  }
  let html = head + " · direct " + node.support_direct + " / lifted " +
    node.support_lifted + "</div>";
  html += '<div class="filters">';
  ["all", "merged", "global_similarity", "relatedness"].forEach(f => {
    html += '<button class="' + (activeFilter === f ? "on" : "") +
      '" onclick="setFilter(\\'' + f + '\\')">' +
      (f === "all" ? "all" : ORIGIN_LABELS[f]) + "</button>";
  });
  html += "</div>";
  ["merged", "global_similarity", "relatedness"].forEach(origin => {
    if (activeFilter !== "all" && activeFilter !== origin) return;
    const list = (node.themes || {})[origin] || [];
    html += '<div class="origin" style="color:' + ORIGIN_COLORS[origin] + '">' +
      ORIGIN_LABELS[origin].toUpperCase() + " (" + list.length + ")</div>";
    list.forEach(t => {
      html += '<div class="theme" style="border-left-color:' + ORIGIN_COLORS[origin] + '">' +
        esc(t.label) + ' <span class="cc">' + t.chunk_count + "ch</span>" +
        (t.keyword_terms && t.keyword_terms.length ?
          '<div class="kw">' + esc(t.keyword_terms.join(", ")) + "</div>" : "") + "</div>";
    });
  });
  d.innerHTML = html;
}

function setFilter(f) { activeFilter = f; if (lastNode) showDetail(lastNode); }

renderTree();
</script></body></html>"""
