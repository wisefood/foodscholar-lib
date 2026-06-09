"""Per-node BERTopic explorer — a persistent version of the head-to-head
notebook's §15 view.

Walks the shelf tree for one facet (or every facet when ``facet=None``), fits
BERTopic per node over that node's chunks (reusing `layer_b.run_bertopic`), and
surfaces both the flat topics and the Layer C card generated for the node's
theme. `build_pernode_explorer(...)` returns a `PernodeExplorer` whose
`.render(output=...)` writes a self-contained collapsible HTML (WiseFood theme).

Operates on **shelves**, so it covers every facet and ontology prefix — not just
`foods` / `FOODON:`. `bertopic` is lazy-imported (via `run_bertopic`).
"""

from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foodscholar.layer_b.bertopic_community import run_bertopic

if TYPE_CHECKING:
    from foodscholar.facade import FoodScholar

_FACETS = ["foods", "health", "sustainability", "dietary_patterns",
           "allergies", "nutrients"]


def _card_payload(theme_handle) -> dict[str, Any] | None:
    card = theme_handle.card()
    if card is None:
        return None
    m = card.model
    return {
        "title": m.title,
        "summary": m.summary,
        "evidence_quality": m.evidence_quality,
        "tip": m.tip,
    }


class PernodeExplorer:
    """The fitted per-node tree(s) + HTML renderer."""

    def __init__(self, roots: list[dict[str, Any]], *, title: str) -> None:
        self.roots = roots
        self.title = title

    def render(self, *, output: str | Path | None = None) -> str | Path:
        """Render the collapsible two-pane HTML. Writes to `output` (returned as
        a `Path`) or returns the HTML string when `output` is None."""
        page = _render_html(self.roots, self.title)
        if output is None:
            return page
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page, encoding="utf-8")
        return out


def build_pernode_explorer(
    fs: FoodScholar,
    *,
    facet: str | None = "foods",
    scope: str | None = None,
    min_chunks: int | None = None,
) -> PernodeExplorer:
    """Fit BERTopic per shelf node and collect topics + cards into a tree.

    - `facet`: one facet, or `None` for every facet (each a top-level branch).
    - `scope`: `"direct"` | `"subtree"`; defaults to `config.layer_b.bertopic.scope`.
    - `min_chunks`: skip nodes with fewer chunks; defaults to the bertopic
      `min_topic_size`.
    """
    bcfg = fs.config.layer_b.bertopic
    use_subtree = (scope or bcfg.scope) == "subtree"
    floor = bcfg.min_topic_size if min_chunks is None else min_chunks

    facets = _FACETS if facet is None else [facet]

    # parent -> children, per shelf id, across the chosen facets
    all_shelves = [s for f in facets for s in fs.graph.shelves(facet=f)]
    by_id = {s.shelf_id: s for s in all_shelves}
    children: dict[str, list[str]] = {}
    roots_by_facet: dict[str, list[str]] = {}
    for s in all_shelves:
        pid = s.parent_shelf_id
        if pid and pid in by_id:
            children.setdefault(pid, []).append(s.shelf_id)
        else:
            roots_by_facet.setdefault(s.model.facet, []).append(s.shelf_id)

    # shelf -> directly-attached chunk ids, from the graph store's attachment
    # map (authoritative; chunk `shelf_ids` denorm may be unset in some stores).
    own_chunks: dict[str, list[str]] = {}
    for cid, sids in fs.graph_store.list_chunk_shelf_attachments().items():
        for sid in sids:
            if sid in by_id:
                own_chunks.setdefault(sid, []).append(cid)

    def _own_ids(sid: str) -> list[str]:
        return own_chunks.get(sid, [])

    def _subtree_ids(sid: str) -> list[str]:
        seen: set[str] = set()
        stack = [sid]
        while stack:
            cur = stack.pop()
            seen.update(_own_ids(cur))
            stack.extend(children.get(cur, ()))
        return sorted(seen)

    def _fit(sid: str) -> dict[str, Any]:
        sh = by_id[sid]
        ids = _subtree_ids(sid) if use_subtree else _own_ids(sid)
        node: dict[str, Any] = {
            "shelf_id": sid, "label": sh.label, "facet": sh.model.facet,
            "n_chunks": len(ids), "topics": [], "card": None,
            "children": [_fit(c) for c in children.get(sid, ())],
        }
        # card: from the shelf's first theme that has one
        for th in sh.themes():
            cp = _card_payload(th)
            if cp is not None:
                node["card"] = cp
                break
        # topics via BERTopic (skip too-small nodes)
        if len(ids) >= floor:
            groups = run_bertopic(ids, fs.chunk_store, bcfg)
            node["topics"] = [{"size": len(g)} for g in groups]
        return node

    roots: list[dict[str, Any]] = []
    if facet is None:
        for f in facets:
            kids = [_fit(sid) for sid in roots_by_facet.get(f, [])]
            roots.append({"shelf_id": f"facet:{f}", "label": f, "facet": f,
                          "n_chunks": sum(k["n_chunks"] for k in kids),
                          "topics": [], "card": None, "children": kids})
    else:
        roots = [_fit(sid) for sid in roots_by_facet.get(facet, [])]

    title = f"BERTopic per-node — {facet or 'all facets'}"
    return PernodeExplorer(roots, title=title)


# ---------------------------------------------------------------- rendering


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def _render_html(roots: list[dict[str, Any]], title: str) -> str:
    data = json.dumps(roots)
    return _TEMPLATE.replace("__TITLE__", _esc(title)).replace("__DATA__", data)


_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
:root{--brand:#d53355;--accent:#646d1a;--header:#173f35;--header-fg:#f6efe6;
--panel:#f6efe6;--line:#e8e1d4;--muted:#7a6657;--fg:#1f1a15;--sel:#f7d6dd;
--accent-soft:#edf0d5;--radius:12px;--font:"Quicksand",ui-sans-serif,system-ui,sans-serif;}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 var(--font);color:var(--fg)}
header{background:var(--header);color:var(--header-fg);padding:12px 18px;display:flex;gap:12px;align-items:baseline}
.wrap{display:flex;height:calc(100vh - 48px)}
.tree{width:42%;overflow:auto;border-right:1px solid var(--line);padding:10px 0;background:var(--panel)}
.detail{flex:1;overflow:auto;padding:18px 22px}
.row{display:flex;align-items:center;gap:7px;padding:3px 10px;cursor:pointer;white-space:nowrap;border-radius:6px;margin:0 6px}
.row:hover{background:var(--accent-soft)}.row.sel{background:var(--sel)}
.caret{width:12px;display:inline-block;color:var(--muted)}
.cc{color:var(--muted);font-variant-numeric:tabular-nums}
.tn{color:var(--accent);font-weight:600}
.kids{margin-left:16px}.hidden{display:none}
h2{margin:0 0 2px;font-size:19px}.sub{color:var(--muted);margin-bottom:12px}
.sec{margin:14px 0 6px;font-weight:600;color:var(--accent)}
.topic{padding:6px 11px;border-left:3px solid var(--accent);margin:5px 0;background:var(--panel);border-radius:0 var(--radius) var(--radius) 0}
.card{padding:10px 13px;border-left:3px solid var(--brand);margin:8px 0;background:#fff;border:1px solid var(--line);border-radius:var(--radius)}
.card .q{font-size:11px;color:var(--brand);font-weight:700;text-transform:uppercase}
.empty{color:var(--muted);font-style:italic}
</style></head><body>
<header><b>__TITLE__</b></header>
<div class="wrap"><div class="tree" id="tree"></div>
<div class="detail" id="detail"><p class="empty">Select a node on the left.</p></div></div>
<script>
const ROOTS = __DATA__;
function esc(s){return String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
function rowFor(node,open){
  const box=document.createElement("div"),row=document.createElement("div");
  row.className="row";const kids=node.children&&node.children.length;
  const tc=(node.topics||[]).length;
  row.innerHTML='<span class="caret">'+(kids?(open?"▾":"▸"):" ")+'</span>'+
    '<span>'+esc(node.label)+'</span>'+'<span class="cc">'+node.n_chunks+'</span>'+
    (tc?'<span class="tn">['+tc+']</span>':'')+(node.card?' ◆':'');
  let kc=null;
  if(kids){kc=document.createElement("div");kc.className="kids"+(open?"":" hidden");
    node.children.forEach(c=>kc.appendChild(rowFor(c,false)));}
  row.onclick=(e)=>{e.stopPropagation();
    if(kids&&e.target.classList.contains("caret")){kc.classList.toggle("hidden");
      e.target.textContent=kc.classList.contains("hidden")?"▸":"▾";return;}
    document.querySelectorAll(".row.sel").forEach(r=>r.classList.remove("sel"));
    row.classList.add("sel");showDetail(node);};
  box.appendChild(row);if(kc)box.appendChild(kc);return box;
}
function showDetail(node){
  const d=document.getElementById("detail");
  let h="<h2>"+esc(node.label)+"</h2><div class='sub'>facet "+esc(node.facet)+
    " · "+node.n_chunks+" chunks · "+(node.topics||[]).length+" topics</div>";
  if(node.card){const c=node.card;
    h+="<div class='card'><div class='q'>Layer C card · "+esc(c.evidence_quality)+"</div>"+
       "<b>"+esc(c.title)+"</b><div>"+esc(c.summary)+"</div>"+
       (c.tip?"<div class='cc'>tip: "+esc(c.tip)+"</div>":"")+"</div>";}
  h+="<div class='sec'>BERTopic topics ("+(node.topics||[]).length+")</div>";
  if((node.topics||[]).length){
    node.topics.forEach((t,i)=>{h+="<div class='topic'>topic "+(i+1)+
      " <span class='cc'>"+t.size+" chunks</span></div>";});
  } else { h+="<p class='empty'>No topics (node below the chunk floor or no fit).</p>"; }
  d.innerHTML=h;
}
const root=document.getElementById("tree");
ROOTS.forEach(n=>root.appendChild(rowFor(n,true)));
</script></body></html>"""
