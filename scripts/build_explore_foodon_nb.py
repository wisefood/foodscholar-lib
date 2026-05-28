"""Assemble notebooks/explore_foodon.ipynb from source cells.

Kept as a script (not hand-edited JSON) so the notebook source stays
reviewable and regenerable. Run with the foodscholar env:

    /mnt/miniconda3/envs/foodscholar/bin/python scripts/build_explore_foodon_nb.py

Then execute headless with nbclient (see the bottom of this file / the
notebook's own run instructions).
"""

from __future__ import annotations

import nbformat as nbf

NB_PATH = "notebooks/explore_foodon.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells: list = []

# ----------------------------------------------------------------- title
cells.append(
    md(
        """# FoodOn — structure exploration

A **corpus-free** look at the raw FoodOn ontology, built to surface the
structural problems that the **Layer-A projection** exists to fix. Layer A
turns FoodOn into a small, navigable set of *shelves*; to motivate each
projection rule we first quantify what is wrong with FoodOn *as shipped*.

Each section pairs a statistic with a figure and a one-line note tying the
finding to the projection rule that answers it:

| § | Problem | Projection rule that answers it |
|---|---------|---------------------------------|
| §1 | Too deep / over-specific | `max_depth` capping + lifting |
| §2 | Single-child scaffolding | single-child chain collapse |
| §3 | Umbrella / abstract nodes | the umbrella rule |
| §4 | DAG-ness & label noise | why a curated projection is needed at all |

The final cell assembles every figure into a self-contained
`data/viz/foodon_report.html`.

> Run on the **`foodscholar`** kernel. Pure structure — no corpus, Elastic, or
> Neo4j. Reproducible from `data/foodon.owl` alone."""
    )
)

# ----------------------------------------------------------------- §0 header
cells.append(
    md(
        """## §0 — Foundation: load FoodOn, build the graph, derive per-node metrics

`FoodOnAPI` gives us labels, parents, children, ancestors, descendants, and the
obsolete flag — but **no depth**. We build one `networkx.DiGraph` (parent→child
edges over `FOODON:` terms) and compute a single per-node frame that every
section below reads from.

A `REPORT` dict accumulates each section's figures + caption; the final cell
renders it to HTML."""
    )
)

# ----------------------------------------------------------------- §0 imports/setup
cells.append(
    code(
        '''import base64
import io
import re
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed, we embed figures as SVG
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from foodscholar.ontology import FoodOnAPI, load_ontology

# Resolve repo root whether run from notebooks/ or repo root.
HERE = Path.cwd()
ROOT = HERE if (HERE / "data" / "foodon.owl").exists() else HERE.parent
OWL = ROOT / "data" / "foodon.owl"
CACHE = ROOT / "data" / "foodon_cache.parquet"
VIZ_DIR = ROOT / "data" / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = VIZ_DIR / "foodon_report.html"

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "figure.autolayout": True})

# REPORT accumulates ordered sections for the final HTML assembly.
# Each entry: {"id", "title", "summary" (html), "caption" (html), "figures" (list of svg str)}
REPORT: list[dict] = []


def fig_to_svg(fig) -> str:
    """Render a matplotlib figure to an inline <svg> string and close it."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg") :]  # strip XML/doctype preamble


def add_section(section_id, title, summary_html, figures, caption_html):
    REPORT.append(
        {
            "id": section_id,
            "title": title,
            "summary": summary_html,
            "figures": list(figures),
            "caption": caption_html,
        }
    )


print("repo root:", ROOT)'''
    )
)

# ----------------------------------------------------------------- §0 load + graph + frame
cells.append(
    code(
        '''terms = load_ontology(OWL, cache_path=CACHE)
api = FoodOnAPI(terms, prefix_filter=("FOODON:",))
ids = [t.id for t in api]

# parent -> child DiGraph over FOODON: terms only (drops BFO/CHEBI/etc. parents)
G = nx.DiGraph()
G.add_nodes_from(ids)
for t in api:
    for p in t.parent_ids:
        if p in api:
            G.add_edge(p, t.id)

assert nx.is_directed_acyclic_graph(G), "FoodOn graph unexpectedly has a cycle"

# Longest-path depth via a virtual super-root over all roots, then topo-sort.
roots = [n for n in G if G.in_degree(n) == 0]
SUPER = "__ROOT__"
Gd = G.copy()
Gd.add_node(SUPER)
for r in roots:
    Gd.add_edge(SUPER, r)
depth_max: dict[str, int] = {SUPER: -1}
for n in nx.topological_sort(Gd):
    if n == SUPER:
        continue
    depth_max[n] = max(depth_max[p] for p in Gd.predecessors(n)) + 1

# Per-node frame — the single source of truth for every section.
rows = []
for t in api:
    rows.append(
        {
            "id": t.id,
            "label": t.label,
            "depth": depth_max[t.id],
            "n_children": G.out_degree(t.id),
            "n_parents": len(t.parent_ids),
            "subtree": len(nx.descendants(G, t.id)),
            "obsolete": t.obsolete,
            "label_len": len(t.label),
            "has_paren": "(" in t.label,
            "n_synonyms": len(t.synonyms),
        }
    )
df = pd.DataFrame(rows).set_index("id")

N = len(df)
N_INTERNAL = int((df.n_children > 0).sum())
print(f"{N} FOODON terms | {G.number_of_edges()} parent-child edges | {len(roots)} roots")
df.describe()[["depth", "n_children", "n_parents", "subtree", "label_len"]].round(2)'''
    )
)

# ----------------------------------------------------------------- §0 helpers
cells.append(
    code(
        '''def lab(tid: str) -> str:
    """Short label for display: '<label> (<id-tail>)'."""
    return f"{api.id_to_label(tid)} [{tid.split(':')[-1]}]"


def chain_to_root(tid: str) -> list[str]:
    """A single representative root->node path (follows first parent each step)."""
    path = [tid]
    cur = tid
    while True:
        parents = [p for p in api.id_to_parents(cur) if p in api]
        if not parents:
            break
        cur = sorted(parents)[0]
        path.append(cur)
    return list(reversed(path))


def df_table(frame: pd.DataFrame, n: int = 12) -> str:
    """Small HTML table for the report summary blocks."""
    return frame.head(n).to_html(border=0, classes="stat", escape=True)


print("helpers ready")'''
    )
)

# ----------------------------------------------------------------- §1 header
cells.append(md("## §1 — Too deep / over-specific"))

cells.append(
    code(
        '''# Depth distribution + depth-vs-subtree (deep terms are leaves, not destinations).
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

dc = df.depth.value_counts().sort_index()
ax1.bar(dc.index, dc.values, color="#4c72b0")
ax1.set_xlabel("depth (longest root→node path)")
ax1.set_ylabel("term count")
ax1.set_title("FoodOn depth distribution")
median_d = int(df.depth.median())
ax1.axvline(median_d, color="#c44e52", ls="--", lw=1)
ax1.text(median_d + 0.2, dc.max() * 0.9, f"median={median_d}", color="#c44e52")

ax2.scatter(df.depth, df.subtree + 1, s=6, alpha=0.25, color="#55a868")
ax2.set_yscale("log")
ax2.set_xlabel("depth")
ax2.set_ylabel("subtree size (descendants+1, log)")
ax2.set_title("deep terms are leaves, not destinations")
svg1 = fig_to_svg(fig)

# A few example deepest root->leaf paths.
deepest = df.sort_values("depth", ascending=False).head(3).index
example_paths = []
for tid in deepest:
    chain = chain_to_root(tid)
    example_paths.append(
        f"<div class='path'><b>depth {df.loc[tid,'depth']}:</b> "
        + " &rsaquo; ".join(api.id_to_label(c) or c for c in chain)
        + "</div>"
    )

max_d = int(df.depth.max())
deep_share = 100 * (df.depth >= 8).mean()
summary = (
    f"<b>Max depth {max_d}</b>, median {median_d}. "
    f"{deep_share:.0f}% of terms sit at depth ≥ 8 — buried below any plausible "
    f"browsing destination.<br>" + "".join(example_paths)
)
caption = (
    "FoodOn descends 14 levels for some classifications; the right panel shows "
    "the deepest terms carry no subtree — they are leaves, not navigation hubs. "
    "<b>Answered by</b> Layer-A <code>max_depth</code> capping + lifting: deep "
    "terms re-home onto the nearest surviving ancestor at depth ≤ cap."
)
add_section("s1", "§1 — Too deep / over-specific", summary, [svg1], caption)
print(summary.split("<br>")[0])'''
    )
)

# ----------------------------------------------------------------- §2 header
cells.append(md("## §2 — Single-child scaffolding"))

cells.append(
    code(
        '''# Parents with exactly one child = organizational scaffolding, not branches.
single_child = df[df.n_children == 1].index.tolist()


def chain_len_from(tid: str) -> list[str]:
    """Walk down while each node has exactly one child: the scaffolding run."""
    run = [tid]
    cur = tid
    while G.out_degree(cur) == 1:
        (child,) = list(G.successors(cur))
        run.append(child)
        cur = child
    return run


# Longest single-child runs (dedupe by keeping runs not contained in a longer one).
runs = sorted((chain_len_from(t) for t in single_child), key=len, reverse=True)
seen: set[str] = set()
top_runs = []
for r in runs:
    if r[0] in seen:
        continue
    seen.update(r)
    if len(r) >= 4:
        top_runs.append(r)
    if len(top_runs) >= 5:
        break

fig, ax = plt.subplots(figsize=(6, 4))
child_counts = df[df.n_children > 0].n_children
ax.hist(child_counts.clip(upper=20), bins=range(1, 22), color="#8172b3", align="left")
ax.set_xlabel("# direct children (clipped at 20)")
ax.set_ylabel("internal-node count")
ax.set_title("branching factor — the spike at 1 is scaffolding")
ax.axvline(1, color="#c44e52", ls="--", lw=1)
svg2 = fig_to_svg(fig)

runs_html = "".join(
    "<div class='path'>" + " &rsaquo; ".join(api.id_to_label(c) or c for c in r) + "</div>"
    for r in top_runs
)
sc_share = 100 * len(single_child) / N_INTERNAL
summary = (
    f"<b>{len(single_child):,} of {N_INTERNAL:,} internal nodes ({sc_share:.0f}%) "
    f"have exactly one child.</b> Longest pure single-child runs:<br>" + runs_html
)
caption = (
    "More than a third of FoodOn's internal nodes are single-child links — pure "
    "organizational scaffolding a user would never branch at. <b>Answered by</b> "
    "Layer-A single-child chain collapse: <code>A→B→C</code> chains collapse to the "
    "deepest survivor, with collapsed ids recorded in <code>see_also</code>."
)
add_section("s2", "§2 — Single-child scaffolding", summary, [svg2], caption)
print(f"{len(single_child):,} single-child parents ({sc_share:.0f}% of internal)")'''
    )
)

# ----------------------------------------------------------------- §3 header
cells.append(md("## §3 — Umbrella / abstract nodes"))

cells.append(
    code(
        '''# Fan-out giants (many direct children) and the abstract spine (huge subtree).
top_fanout = df.sort_values("n_children", ascending=False).head(12)
top_subtree = df[~df.obsolete].sort_values("subtree", ascending=False).head(12)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
y = range(len(top_fanout))
ax1.barh(y, top_fanout.n_children.values, color="#937860")
ax1.set_yticks(list(y))
ax1.set_yticklabels(top_fanout.label.values, fontsize=8)
ax1.invert_yaxis()
ax1.set_xlabel("# direct children")
ax1.set_title("fan-out giants")

y2 = range(len(top_subtree))
ax2.barh(y2, top_subtree.subtree.values, color="#da8bc3")
ax2.set_yticks(list(y2))
ax2.set_yticklabels(top_subtree.label.values, fontsize=8)
ax2.invert_yaxis()
ax2.set_xlabel("# transitive descendants")
ax2.set_title("the abstract spine")
svg3 = fig_to_svg(fig)

spine = top_subtree.head(5)
spine_html = "".join(
    f"<div class='path'><b>{r.label}</b> — {r.subtree:,} descendants, "
    f"only {r.n_children} direct children</div>"
    for r in spine.itertuples()
)
summary = (
    f"The biggest abstract classes own enormous subtrees while meaning almost "
    f"nothing concrete:<br>{spine_html}"
)
caption = (
    "Classes like <i>food material</i> and <i>organism material</i> accumulate "
    "tens of thousands of descendants but are never what a reader is looking for. "
    "<b>Answered by</b> the Layer-A umbrella rule: drop classes that are big only "
    "because of descendants <i>and</i> almost never linked directly."
)
add_section("s3", "§3 — Umbrella / abstract nodes", summary, [svg3], caption)
print("top subtree:", list(top_subtree.label.head(3)))'''
    )
)

# ----------------------------------------------------------------- §4 header
cells.append(md("## §4 — DAG-ness & label noise"))

cells.append(
    code(
        '''# Multi-parent (not a tree), obsolete terms, verbose/auto-generated labels.
multi_parent = df[df.n_parents > 1]
obsolete = df[df.obsolete]
paren = df[df.has_paren]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
pc = df.n_parents.value_counts().sort_index()
ax1.bar(pc.index, pc.values, color="#4c72b0")
ax1.set_yscale("log")
ax1.set_xlabel("# direct parents")
ax1.set_ylabel("term count (log)")
ax1.set_title("multi-parent = DAG, not a tree")

ax2.hist(df.label_len.clip(upper=80), bins=30, color="#c44e52")
ax2.set_xlabel("label length (chars, clipped 80)")
ax2.set_ylabel("term count")
ax2.set_title("label length — long tail of auto-generated names")
svg4 = fig_to_svg(fig)

worst_mp = multi_parent.sort_values("n_parents", ascending=False).head(5)
worst_html = "".join(
    f"<div class='path'>{r.label} — <b>{r.n_parents} parents</b></div>"
    for r in worst_mp.itertuples()
)
# Example auto-generated verbose labels.
verbose = paren.sort_values("label_len", ascending=False).head(4)
verbose_html = "".join(f"<div class='path'><code>{r.label}</code></div>" for r in verbose.itertuples())

summary = (
    f"<b>{len(multi_parent):,} terms ({100*len(multi_parent)/N:.0f}%) have >1 parent</b> "
    f"— FoodOn is a DAG, not a tree. "
    f"<b>{len(obsolete):,} obsolete</b> terms still ship. "
    f"<b>{len(paren):,} ({100*len(paren)/N:.0f}%)</b> labels carry parenthetical "
    f"modifiers (machine-generated).<br>"
    f"<u>Most-parented terms:</u>{worst_html}"
    f"<u>Verbose auto-labels:</u>{verbose_html}"
)
caption = (
    "A chunk's term can sit under several parents at once, half the labels are "
    "machine-generated modifier strings, and obsolete terms linger. <b>Answered "
    "by</b> projecting onto a curated, single-parent shelf tree at all — the raw "
    "ontology is not directly browsable."
)
add_section("s4", "§4 — DAG-ness & label noise", summary, [svg4], caption)
print(
    f"multi-parent {len(multi_parent):,} | obsolete {len(obsolete):,} | "
    f"paren-labels {len(paren):,}"
)'''
    )
)

# ----------------------------------------------------------------- hierarchical tree
cells.append(
    md(
        """## Hierarchical taxonomy tree — rooted at `food product`

A collapsible hierarchical view of the FoodOn taxonomy under
**food product** (`FOODON:00001002`). Pure nested `<details>`/`<ul>` — no JS,
self-contained, prints cleanly. Click a node to expand/collapse its children.

The tree is bounded (depth + children-per-node) so it stays legible; truncated
branches show a `+N more` marker. Single-child links (the §2 scaffolding) are
flagged inline, so the tree itself shows why the raw taxonomy isn't browsable."""
    )
)

cells.append(
    code(
        '''import html as _html

TREE_ROOT_LABEL = "food product"
tree_root_id = api.name_to_id(TREE_ROOT_LABEL) or df[~df.obsolete].subtree.idxmax()

TREE_MAX_DEPTH = 4          # levels shown below the root
TREE_MAX_CHILDREN = 12      # children rendered per node before "+N more"
TREE_OPEN_DEPTH = 1         # auto-expanded down to this relative depth


def _render_tree(tid: str, rel_depth: int) -> str:
    """Nested <details> for tid. Children sorted by subtree size desc."""
    label = _html.escape(api.id_to_label(tid) or tid)
    nch = G.out_degree(tid)
    subtree = int(df.loc[tid, "subtree"]) if tid in df.index else 0
    badge = f"<span class='tcount'>{nch} children · {subtree:,} desc</span>" if nch else \
            "<span class='tleaf'>leaf</span>"
    scaffold = " <span class='tscaffold'>single-child →</span>" if nch == 1 else ""
    head = f"<span class='tlabel'>{label}</span>{scaffold} {badge}"

    if nch == 0 or rel_depth >= TREE_MAX_DEPTH:
        suffix = f" <span class='tmore'>(+{subtree:,} below)</span>" if subtree else ""
        return f"<li class='tleafrow'>{head}{suffix}</li>"

    children = sorted(G.successors(tid), key=lambda c: -df.loc[c, "subtree"])
    shown = children[:TREE_MAX_CHILDREN]
    hidden = len(children) - len(shown)
    open_attr = " open" if rel_depth < TREE_OPEN_DEPTH else ""
    inner = "".join(_render_tree(c, rel_depth + 1) for c in shown)
    if hidden > 0:
        inner += f"<li class='tmorerow'>+{hidden} more child{'ren' if hidden != 1 else ''}…</li>"
    return f"<li><details{open_attr}><summary>{head}</summary><ul>{inner}</ul></details></li>"


TREE_HTML = f"<ul class='ftree'>{_render_tree(tree_root_id, 0)}</ul>"
n_tree_nodes = TREE_HTML.count("<summary>") + TREE_HTML.count("tleafrow")

summary = (
    f"Taxonomy under <b>{api.id_to_label(tree_root_id)}</b> "
    f"({df.loc[tree_root_id,'subtree']:,} total descendants), rendered "
    f"{TREE_MAX_DEPTH} levels deep, ≤{TREE_MAX_CHILDREN} children per node. "
    f"Single-child links are flagged; collapsed branches show how much is hidden."
)
caption = (
    "Even bounded to 4 levels the tree fans out hard and threads through "
    "single-child scaffolding — the raw taxonomy is a reference structure, not a "
    "browsing surface. <b>Answered by</b> the full Layer-A cascade (cap + lift, "
    "collapse, umbrella prune) that flattens this into a shallow shelf tree."
)
add_section("tree", "Hierarchical taxonomy tree", summary, [TREE_HTML], caption)
print(f"tree: ~{n_tree_nodes} rendered rows under {lab(tree_root_id)}")'''
    )
)

# ----------------------------------------------------------------- headline widget
cells.append(
    md(
        """## Headline — interactive subtree

A collapsible/zoomable `pyvis` view of an abstract node's neighborhood, so the
over-deep, single-child reality is explorable live. Embedded self-contained in
the report. We root it at the **food product** spine and cap the BFS so the
widget stays legible."""
    )
)

cells.append(
    code(
        '''from pyvis.network import Network

# Root the widget at a recognizable abstract node; fall back to the biggest spine.
ROOT_LABEL = "food product"
root_id = api.name_to_id(ROOT_LABEL) or df[~df.obsolete].subtree.idxmax()

# BFS a bounded neighborhood (breadth-limited) so the widget renders fast.
MAX_NODES = 120
MAX_CHILDREN_PER = 8
sub_nodes = [root_id]
frontier = [root_id]
seen = {root_id}
while frontier and len(sub_nodes) < MAX_NODES:
    nxt = []
    for n in frontier:
        children = sorted(G.successors(n), key=lambda c: -df.loc[c, "subtree"])[:MAX_CHILDREN_PER]
        for c in children:
            if c not in seen and len(sub_nodes) < MAX_NODES:
                seen.add(c)
                sub_nodes.append(c)
                nxt.append(c)
    frontier = nxt

net = Network(height="600px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
net.barnes_hut(spring_length=120)
for n in sub_nodes:
    size = 10 + min(30, df.loc[n, "subtree"] ** 0.4)
    color = "#c44e52" if n == root_id else ("#dd8452" if df.loc[n, "n_children"] == 1 else "#4c72b0")
    title = f"{api.id_to_label(n)}\\n{n}\\ndepth={df.loc[n,'depth']} children={df.loc[n,'n_children']} subtree={df.loc[n,'subtree']}"
    net.add_node(n, label=api.id_to_label(n) or n, size=size, color=color, title=title)
for n in sub_nodes:
    for c in G.successors(n):
        if c in seen:
            net.add_edge(n, c)

WIDGET_HTML = net.generate_html(notebook=False)
print(f"widget: {len(sub_nodes)} nodes rooted at {lab(root_id)}")
# orange = single-child scaffolding, red = root, blue = branching'''
    )
)

# ----------------------------------------------------------------- HTML assembly header
cells.append(md("## Assemble the self-contained HTML report"))

cells.append(
    code(
        r'''from jinja2 import Template

TEMPLATE = Template(
    """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>FoodOn — structure exploration</title>
<style>
  body { font: 15px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1000px;
         margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
  h1 { border-bottom: 3px solid #4c72b0; padding-bottom: .3rem; }
  h2 { margin-top: 2.4rem; color: #2a3f5f; border-left: 5px solid #4c72b0; padding-left: .6rem; }
  .summary { background: #f5f7fa; padding: .8rem 1rem; border-radius: 6px; }
  .caption { font-size: .92rem; color: #444; background: #fffbe6;
             border-left: 4px solid #e0b400; padding: .6rem 1rem; margin: .8rem 0; }
  .path { font-size: .88rem; margin: .25rem 0; }
  .path code { background: #eee; padding: 0 .2rem; }
  table.stat { border-collapse: collapse; font-size: .85rem; }
  table.stat td, table.stat th { border: 1px solid #ddd; padding: 2px 8px; }
  .figwrap svg { max-width: 100%; height: auto; }
  .meta { color: #888; font-size: .85rem; }
  .legend span { display:inline-block; padding:.1rem .5rem; border-radius:3px; color:#fff; margin-right:.4rem;}
  ul.ftree, ul.ftree ul { list-style: none; margin: 0; padding-left: 1.1rem;
                          border-left: 1px dotted #cbd2dc; }
  ul.ftree { padding-left: 0; border-left: none; }
  ul.ftree li { margin: .12rem 0; font-size: .9rem; }
  ul.ftree summary { cursor: pointer; }
  ul.ftree summary:hover { background: #eef2f7; }
  .tlabel { font-weight: 600; color: #2a3f5f; }
  .tcount { color: #888; font-size: .82em; margin-left: .35rem; }
  .tleaf { color: #aaa; font-size: .82em; margin-left: .35rem; font-style: italic; }
  .tscaffold { color: #dd8452; font-size: .78em; font-weight: 600; }
  .tmore, .tmorerow { color: #c44e52; font-size: .82em; }
  .tmorerow { font-style: italic; }
  .tleafrow { color: #555; }
</style></head><body>
<h1>FoodOn — structure exploration</h1>
<p class="meta">{{ n_terms }} FOODON terms · {{ n_edges }} parent–child edges ·
generated from <code>data/foodon.owl</code> · corpus-free</p>
<p>Why these problems matter: Layer A turns FoodOn into a small, navigable set of
<em>shelves</em>. Each section below quantifies a structural problem of the raw
ontology and names the projection rule that answers it.</p>

{% for s in sections %}
<h2 id="{{ s.id }}">{{ s.title }}</h2>
<div class="summary">{{ s.summary }}</div>
{% for fig in s.figures %}<div class="figwrap">{{ fig }}</div>{% endfor %}
<div class="caption">{{ s.caption }}</div>
{% endfor %}

<h2 id="widget">Interactive subtree — <code>{{ widget_root }}</code></h2>
<p class="legend"><span style="background:#c44e52">root</span>
<span style="background:#dd8452">single-child scaffolding</span>
<span style="background:#4c72b0">branching node</span> — node size ∝ subtree.
Drag to pan, scroll to zoom.</p>
<iframe srcdoc="{{ widget_srcdoc }}" style="width:100%;height:640px;border:1px solid #ddd;border-radius:6px;"></iframe>
</body></html>"""
)

html = TEMPLATE.render(
    n_terms=f"{N:,}",
    n_edges=f"{G.number_of_edges():,}",
    sections=REPORT,
    widget_root=api.id_to_label(root_id),
    widget_srcdoc=WIDGET_HTML.replace('"', "&quot;"),
)
REPORT_PATH.write_text(html, encoding="utf-8")
print(f"wrote {REPORT_PATH}  ({len(html)/1024:.0f} KB, {len(REPORT)} sections + widget)")'''
    )
)

# ----------------------------------------------------------------- build
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "foodscholar",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH} with {len(cells)} cells")
