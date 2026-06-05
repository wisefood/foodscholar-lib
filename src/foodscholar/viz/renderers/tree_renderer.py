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

ORIGIN_COLORS = {  # WiseFood palette: brand-green / brand-purple / terracotta
    "merged": "#646d1a",             # brandg (olive green) — both passes agreed
    "global_similarity": "#733c95",  # brandp (purple) — embedding-only
    "relatedness": "#b5663f",        # terracotta — entity-only
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
            "terms": n.attrs.get("terms", []),
            "entities": n.attrs.get("entities", []),
            "sources": n.attrs.get("sources", []),
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
  /* --- WiseFood brand tokens (wisefood-ui app/assets/css/main.css) --------- */
  :root {
    --brand: #d53355; --brandg: #a6b52b; --terracotta: #d98a6b;
    --earth-1: #f6efe6; --earth-2: #cad5b2;
    --bg: #ffffff; --fg: #1f1a15; --muted: #7a6657; --line: #e8e1d4;
    --panel: #f6efe6; --panel-2: #efe7d8;             /* earth-1 surfaces */
    --header: #173f35; --header-fg: #f6efe6;          /* deep forest green */
    --accent: #646d1a; --accent-soft: #edf0d5; --sel: #f7d6dd;  /* olive + brand-soft select */
    --chip-bg: #edf0d5; --chip-fg: #424811;           /* brandg tints */
    --radius: 12px;
    --font-body: "Quicksand", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    --font-display: ui-rounded, "Quicksand", "SF Pro Rounded", system-ui, sans-serif;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.45 var(--font-body); color: var(--fg); background: var(--bg); }
  header { padding: 12px 18px; background: var(--header); color: var(--header-fg);
           display: flex; align-items: baseline; gap: 12px; }
  header b { font-size: 15px; letter-spacing: .2px; font-family: var(--font-display); }
  header .counts { opacity: .82; font-size: 12.5px; }
  .wrap { display: flex; height: calc(100vh - 48px); }
  .tree { width: 44%; overflow: auto; border-right: 1px solid var(--line);
          padding: 10px 0; background: var(--panel); }
  .detail { flex: 1; overflow: auto; padding: 18px 22px; }
  .row { display: flex; align-items: center; gap: 7px; padding: 3px 10px;
         cursor: pointer; white-space: nowrap; border-radius: 6px; margin: 0 6px; }
  .row:hover { background: var(--accent-soft); } .row.sel { background: var(--sel); }
  .row.sub { color: var(--muted); }
  .caret { width: 12px; display: inline-block; color: var(--muted); }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .cc { color: var(--muted); font-variant-numeric: tabular-nums; }
  .tn { color: var(--accent); font-variant-numeric: tabular-nums; font-weight: 600; }
  .kids { margin-left: 16px; } .hidden { display: none; }
  h2 { margin: 0 0 2px; font-size: 19px; font-family: var(--font-display); }
  .sub-meta { color: var(--muted); margin-bottom: 12px; }
  .tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--line); margin-bottom: 14px; }
  .tab { border: 0; background: transparent; color: var(--muted); cursor: pointer;
         padding: 8px 12px; font-size: 13.5px; border-bottom: 2px solid transparent; }
  .tab:hover { color: var(--fg); }
  .tab.on { color: var(--brand); border-bottom-color: var(--brand); font-weight: 600; }
  .filters { margin: 0 0 10px; }
  .filters button { margin-right: 5px; border: 1px solid var(--line); background: var(--bg);
                    border-radius: 999px; padding: 3px 11px; cursor: pointer; color: var(--muted); }
  .filters button.on { background: var(--header); color: var(--header-fg); border-color: var(--header); }
  .origin { margin: 14px 0 6px; font-weight: 600; }
  .theme { padding: 7px 11px; border-left: 3px solid var(--line); margin: 5px 0;
           background: var(--panel); border-radius: 0 var(--radius) var(--radius) 0; }
  .theme .kw { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { background: var(--chip-bg); color: var(--chip-fg); border-radius: 999px;
          padding: 3px 10px; font-size: 12.5px; }
  .chip b { color: var(--accent); font-variant-numeric: tabular-nums; }
  .list { display: flex; flex-direction: column; gap: 2px; }
  .item { display: flex; align-items: baseline; gap: 8px; padding: 5px 8px; border-radius: 6px; }
  .item:hover { background: var(--panel-2); }
  .item .n { margin-left: auto; color: var(--accent); font-variant-numeric: tabular-nums; font-weight: 600; }
  .item .meta { color: var(--muted); font-size: 12px; }
  .empty { color: var(--muted); font-style: italic; }
</style></head><body>
<header><b id="h-title"></b><span class="counts" id="h-counts"></span></header>
<div class="wrap">
  <div class="tree" id="tree"></div>
  <div class="detail" id="detail"><p class="empty">Select a shelf on the left.</p></div>
</div>
<script>
const TREE_DATA = __TREE_JSON__;
const ORIGIN_COLORS = __ORIGIN_COLORS__;
const ORIGIN_LABELS = __ORIGIN_LABELS__;
const TABS = [["topics","Topics"],["terms","Terms"],["entities","Entities"],["sources","Sources"]];
let activeFilter = "all";
let activeTab = "topics";
let lastNode = null;

function esc(s) { return String(s == null ? "" : s)
  .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

function themeCount(node) {
  const t = node.themes || {};
  return (t.merged||[]).length + (t.global_similarity||[]).length + (t.relatedness||[]).length;
}
function tabCount(node, k) {
  if (k === "topics") return themeCount(node);
  if (k === "terms") return (node.terms||[]).length;
  if (k === "entities") return (node.entities||[]).length;
  if (k === "sources") return (node.sources||[]).length;
  return 0;
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
      (node.eligible ? "var(--accent)" : "var(--line)") + '"></span>' +
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

function topicsBody(node) {
  if (!node.eligible)
    return "<p class='empty'>Below the chunk threshold — no themes discovered.</p>";
  let html = '<div class="filters">';
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
  return html;
}

function termsBody(node) {
  const terms = node.terms || [];
  if (!terms.length) return "<p class='empty'>No terms for this shelf.</p>";
  let html = '<div class="chips">';
  terms.forEach(t => {
    html += '<span class="chip">' + esc(t.term) + ' <b>' + t.count + "</b></span>";
  });
  return html + "</div>";
}

function entitiesBody(node) {
  const ents = node.entities || [];
  if (!ents.length) return "<p class='empty'>No FoodOn entities for this shelf.</p>";
  let html = '<div class="list">';
  ents.forEach(e => {
    html += '<div class="item"><span>' + esc(e.label) +
      '</span> <span class="meta">' + esc(e.id) + '</span>' +
      '<span class="n">' + e.count + "</span></div>";
  });
  return html + "</div>";
}

function sourcesBody(node) {
  const srcs = node.sources || [];
  if (!srcs.length) return "<p class='empty'>No source documents for this shelf.</p>";
  let html = '<div class="list">';
  srcs.forEach(s => {
    const meta = [s.source_type, s.year].filter(Boolean).join(" · ");
    html += '<div class="item"><span>' + esc(s.doc_id) + '</span>' +
      (meta ? ' <span class="meta">' + esc(meta) + "</span>" : "") +
      '<span class="n">' + s.count + "</span></div>";
  });
  return html + "</div>";
}

function bodyFor(node) {
  if (activeTab === "terms") return termsBody(node);
  if (activeTab === "entities") return entitiesBody(node);
  if (activeTab === "sources") return sourcesBody(node);
  return topicsBody(node);
}

function showDetail(node) {
  lastNode = node;
  const d = document.getElementById("detail");
  let html = "<h2>" + esc(node.label) + "</h2>" +
    '<div class="sub-meta">' + (node.foodon_id ? esc(node.foodon_id) + " · " : "") +
    "depth " + node.depth + " · " + node.chunk_count + " chunks · direct " +
    node.support_direct + " / lifted " + node.support_lifted + "</div>";
  html += '<div class="tabs">';
  TABS.forEach(([k, lab]) => {
    const n = tabCount(node, k);
    html += '<button class="tab ' + (activeTab === k ? "on" : "") +
      '" onclick="setTab(\\'' + k + '\\')">' + lab + " (" + n + ")</button>";
  });
  html += "</div>" + bodyFor(node);
  d.innerHTML = html;
}

function setFilter(f) { activeFilter = f; if (lastNode) showDetail(lastNode); }
function setTab(k) { activeTab = k; if (lastNode) showDetail(lastNode); }

renderTree();
</script></body></html>"""
