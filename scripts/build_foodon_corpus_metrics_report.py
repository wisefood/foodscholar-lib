"""Build a FoodOn OWL structure + corpus link metrics HTML report.

The report is intentionally descriptive: it quantifies the loaded FoodOn OWL
term structure, including non-FOODON prefixes embedded in the ontology, and how
the current corpus attaches to those ontology nodes.

Run:
    /mnt/miniconda3/envs/foodscholar/bin/python scripts/build_foodon_corpus_metrics_report.py
"""

from __future__ import annotations

import html
import math
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

from foodscholar.config import FoodScholarConfig
from foodscholar.facade import FoodScholar


HERE = Path.cwd()
ROOT = HERE if (HERE / "data" / "foodon.owl").exists() else HERE.parent
OUT = ROOT / "data" / "viz" / "foodon_corpus_metrics.html"
OUT.parent.mkdir(parents=True, exist_ok=True)


def pct(n: int | float, d: int | float) -> str:
    return f"{(100 * n / d):.1f}%" if d else "0.0%"


def fmt(n: int | float) -> str:
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def gini(values: list[int]) -> float:
    values = sorted(v for v in values if v >= 0)
    if not values or sum(values) == 0:
        return 0.0
    n = len(values)
    weighted = sum((i + 1) * v for i, v in enumerate(values))
    return (2 * weighted) / (n * sum(values)) - (n + 1) / n


def table(headers: list[str], rows: list[list[object]], *, cls: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body.append(f"<tr>{cells}</tr>")
    return f"<table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def bar(value: int | float, maximum: int | float, label: str | None = None) -> str:
    width = 0 if maximum <= 0 else max(2, min(100, int(100 * value / maximum)))
    text = html.escape(label if label is not None else fmt(value))
    return f"<div class='bar'><span style='width:{width}%'></span><b>{text}</b></div>"


def percent_bar(value: int | float, total: int | float, label: str | None = None) -> str:
    width = 0 if total <= 0 else max(2, min(100, int(100 * value / total)))
    text = html.escape(label if label is not None else pct(value, total))
    return f"<div class='bar pct'><span style='width:{width}%'></span><b>{text}</b></div>"


def label(api, fid: str) -> str:
    return html.escape(api.id_to_label(fid) or fid)


def load_foodscholar() -> FoodScholar:
    cfg = {
        "corpus": {
            "chunks_path": str(ROOT / "tests/fixtures/sample_chunks.jsonl"),
            "annotated_snapshot_path": str(ROOT / "data/annotated.parquet"),
        },
        "ontology": {
            "foodon_path": str(ROOT / "data/foodon.owl"),
            "cache_path": str(ROOT / "data/foodon_cache.parquet"),
            "prefix_filter": None,
        },
        "layer_a": {
            "facets": ["foods"],
            "min_support": 25,
            "max_depth": 6,
            "blacklist_terms": ["material entity", "physical object", "manufactured product"],
        },
        "storage": {
            "chunk_store": {"backend": "memory"},
            "graph_store": {"backend": "memory"},
        },
    }
    fs = FoodScholar.from_config(FoodScholarConfig.model_validate(cfg))
    api = fs.load_ontology()
    fs.attach_ontology(api)
    fs.load_chunks(str(ROOT / "data/annotated.parquet"))
    return fs


def build_depths(api) -> dict[str, int]:
    graph = nx.DiGraph()
    ids = [t.id for t in api]
    graph.add_nodes_from(ids)
    for t in api:
        for parent in t.parent_ids:
            if parent in api:
                graph.add_edge(parent, t.id)

    roots = [n for n in graph if graph.in_degree(n) == 0]
    super_root = "__ROOT__"
    graph.add_node(super_root)
    for root in roots:
        graph.add_edge(super_root, root)

    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("FoodOn parent graph is not acyclic")

    depths = {super_root: -1}
    for node in nx.topological_sort(graph):
        if node == super_root:
            continue
        depths[node] = max(depths[parent] for parent in graph.predecessors(node)) + 1
    return depths


def prefix_of(fid: str) -> str:
    return fid.split(":", 1)[0] if ":" in fid else "(no prefix)"


def chunk_ontology_terms(api, chunk) -> set[str]:
    fids = set()
    for fid in getattr(chunk, "foodon_ids", []) or []:
        if fid in api:
            fids.add(fid)
    for link in getattr(chunk, "entity_links", []) or []:
        if link.ontology_id in api:
            fids.add(link.ontology_id)
    return fids


def choose_tree_parent(api, fid: str, depths: dict[str, int]) -> str | None:
    parents = [p for p in api.id_to_parents(fid) if p in api]
    if not parents:
        return None
    return max(parents, key=lambda p: depths.get(p, 0))


def render_tree(api, root: str, children: dict[str, list[str]], counts: dict[str, int]) -> str:
    max_depth = 7
    max_children = 20

    def rank(fid: str) -> tuple[int, int, str]:
        return (counts.get(fid, 0), len(children.get(fid, [])), api.id_to_label(fid) or fid)

    def node(fid: str, depth: int) -> str:
        kids = sorted(children.get(fid, []), key=rank, reverse=True)
        shown = kids[:max_children]
        hidden = len(kids) - len(shown)
        count = counts.get(fid, 0)
        name = label(api, fid) if fid in api else html.escape(fid)
        badge = f"<span class='c'>{len(kids):,} sub · {count:,} chunks</span>"
        empty = " empty" if count == 0 else ""
        if not kids or depth >= max_depth:
            return f"<li class='{empty}'>{name}{badge}</li>"
        inner = "".join(node(child, depth + 1) for child in shown)
        if hidden:
            inner += f"<li class='more'>+{hidden:,} more...</li>"
        open_attr = " open" if depth < 2 else ""
        return (
            f"<li class='{empty}'><details{open_attr}><summary><b>{name}</b>{badge}</summary>"
            f"<ul>{inner}</ul></details></li>"
        )

    return f"<ul class='tree'>{node(root, 0)}</ul>"


def main() -> None:
    fs = load_foodscholar()
    api = fs.ontology
    chunks = list(fs.chunk_store.scan())
    food_product = api.name_to_id("food product")

    ids = [t.id for t in api]
    terms_by_id = {t.id: t for t in api}
    depths = build_depths(api)

    children_all: dict[str, list[str]] = defaultdict(list)
    roots = []
    for fid in ids:
        parent = choose_tree_parent(api, fid, depths)
        if parent is None:
            roots.append(fid)
        else:
            children_all[parent].append(fid)

    direct_chunk_ids: dict[str, set[str]] = defaultdict(set)
    rollup_chunk_ids: dict[str, set[str]] = defaultdict(set)
    chunk_terms: dict[str, set[str]] = {}

    for chunk in chunks:
        terms = chunk_ontology_terms(api, chunk)
        if not terms:
            continue
        chunk_terms[chunk.chunk_id] = terms
        for fid in terms:
            direct_chunk_ids[fid].add(chunk.chunk_id)
            for node in [fid] + [a for a in api.id_to_ancestors(fid) if a in api]:
                rollup_chunk_ids[node].add(chunk.chunk_id)

    direct_counts = {fid: len(direct_chunk_ids.get(fid, set())) for fid in ids}
    rollup_counts = {fid: len(rollup_chunk_ids.get(fid, set())) for fid in ids}
    child_counts = {fid: len(api.id_to_children(fid)) for fid in ids}
    parent_counts = {fid: len([p for p in api.id_to_parents(fid) if p in api]) for fid in ids}
    descendant_counts = {fid: len(api.id_to_descendants(fid)) for fid in ids}

    food_product_nodes = {food_product, *api.id_to_descendants(food_product)} if food_product else set()
    foodon_nodes = {fid for fid in ids if fid.startswith("FOODON:")}
    non_foodon_nodes = set(ids) - foodon_nodes
    supported_nodes = {fid for fid, n in rollup_counts.items() if n > 0}
    direct_supported_nodes = {fid for fid, n in direct_counts.items() if n > 0}
    leaf_nodes = {fid for fid in ids if child_counts[fid] == 0}
    internal_nodes = set(ids) - leaf_nodes
    empty_nodes = set(ids) - supported_nodes

    def scope_row(name: str, scope: set[str]) -> list[object]:
        empty = scope - supported_nodes
        supported = scope & supported_nodes
        direct = scope & direct_supported_nodes
        leaves = scope & leaf_nodes
        internal = scope & internal_nodes
        return [
            html.escape(name),
            fmt(len(scope)),
            fmt(len(leaves)),
            fmt(len(internal)),
            fmt(len(supported)),
            pct(len(supported), len(scope)),
            fmt(len(direct)),
            fmt(len(empty)),
            fmt(len(empty & leaf_nodes)),
            fmt(len(empty & internal_nodes)),
        ]

    overview_rows = [
        ["Loaded ontology terms", fmt(len(ids))],
        ["Loaded ontology parent-child edges", fmt(sum(child_counts.values()))],
        ["Loaded ontology roots", fmt(len(roots))],
        ["FOODON-prefixed terms", fmt(len(foodon_nodes))],
        ["Non-FOODON-prefixed terms", fmt(len(non_foodon_nodes))],
        ["Corpus chunks", fmt(len(chunks))],
        ["Chunks with linked ontology evidence", fmt(len(chunk_terms))],
        ["Chunks without linked ontology evidence", fmt(len(chunks) - len(chunk_terms))],
        ["Directly supported ontology ids", fmt(len(direct_supported_nodes))],
        ["Rollup-supported ontology nodes", fmt(len(supported_nodes))],
        ["Empty loaded ontology nodes", fmt(len(empty_nodes))],
        ["Food product subtree nodes", fmt(len(food_product_nodes))],
    ]

    split_rows = [
        scope_row("All loaded ontology terms", set(ids)),
        scope_row("FOODON-prefixed terms", foodon_nodes),
        scope_row("Non-FOODON-prefixed terms", non_foodon_nodes),
        scope_row("food product subtree", food_product_nodes),
    ]

    composition_rows = [
        ["FOODON-prefixed terms", fmt(len(foodon_nodes)), percent_bar(len(foodon_nodes), len(ids))],
        ["Non-FOODON-prefixed terms", fmt(len(non_foodon_nodes)), percent_bar(len(non_foodon_nodes), len(ids))],
    ]
    chunk_evidence_rows = [
        [
            "Chunks with linked ontology evidence",
            fmt(len(chunk_terms)),
            percent_bar(len(chunk_terms), len(chunks)),
        ],
        [
            "Chunks without linked ontology evidence",
            fmt(len(chunks) - len(chunk_terms)),
            percent_bar(len(chunks) - len(chunk_terms), len(chunks)),
        ],
    ]
    node_support_rows = [
        [
            "Rollup-supported ontology nodes",
            fmt(len(supported_nodes)),
            percent_bar(len(supported_nodes), len(ids)),
        ],
        [
            "Empty loaded ontology nodes",
            fmt(len(empty_nodes)),
            percent_bar(len(empty_nodes), len(ids)),
        ],
    ]

    prefix_counts = Counter(prefix_of(fid) for fid in ids)
    prefix_direct_supported = Counter(prefix_of(fid) for fid in direct_supported_nodes)
    prefix_rollup_supported = Counter(prefix_of(fid) for fid in supported_nodes)
    prefix_direct_chunks: dict[str, set[str]] = defaultdict(set)
    prefix_rollup_chunks: dict[str, set[str]] = defaultdict(set)
    for fid, cids in direct_chunk_ids.items():
        prefix_direct_chunks[prefix_of(fid)].update(cids)
    for fid, cids in rollup_chunk_ids.items():
        prefix_rollup_chunks[prefix_of(fid)].update(cids)

    prefix_order = sorted(
        prefix_counts,
        key=lambda p: (prefix_rollup_supported[p], prefix_direct_supported[p], prefix_counts[p]),
        reverse=True,
    )
    prefix_rows = [
        [
            html.escape(prefix),
            fmt(prefix_counts[prefix]),
            fmt(prefix_direct_supported[prefix]),
            fmt(prefix_rollup_supported[prefix]),
            fmt(len(prefix_direct_chunks.get(prefix, set()))),
            fmt(len(prefix_rollup_chunks.get(prefix, set()))),
            pct(prefix_rollup_supported[prefix], prefix_counts[prefix]),
        ]
        for prefix in prefix_order
    ]
    prefix_chart_rows = [
        [
            html.escape(prefix),
            fmt(len(prefix_direct_chunks.get(prefix, set()))),
            percent_bar(len(prefix_direct_chunks.get(prefix, set())), len(chunk_terms)),
        ]
        for prefix in prefix_order[:12]
    ]
    scope_support_chart_rows = [
        [
            html.escape(name),
            pct(len(scope & supported_nodes), len(scope)),
            percent_bar(len(scope & supported_nodes), len(scope)),
        ]
        for name, scope in [
            ("All loaded ontology terms", set(ids)),
            ("FOODON-prefixed terms", foodon_nodes),
            ("Non-FOODON-prefixed terms", non_foodon_nodes),
            ("food product subtree", food_product_nodes),
        ]
    ]

    support_values = [rollup_counts[fid] for fid in ids]
    positive_support_values = [v for v in support_values if v > 0]
    direct_term_counts = [len(v) for v in chunk_terms.values()]
    rollup_terms_per_chunk = [
        len({node for fid in terms for node in [fid] + [a for a in api.id_to_ancestors(fid) if a in api]})
        for terms in chunk_terms.values()
    ]

    support_summary_rows = [
        ["Node support min", fmt(min(support_values) if support_values else 0)],
        ["Node support median", fmt(quantile(support_values, 0.50))],
        ["Node support p90", fmt(quantile(support_values, 0.90))],
        ["Node support p99", fmt(quantile(support_values, 0.99))],
        ["Node support max", fmt(max(support_values) if support_values else 0)],
        ["Positive-node support median", fmt(quantile(positive_support_values, 0.50))],
        ["Positive-node support p90", fmt(quantile(positive_support_values, 0.90))],
        ["Node support Gini", fmt(gini(support_values))],
        ["Direct ontology ids per supported chunk median", fmt(quantile(direct_term_counts, 0.50))],
        ["Direct ontology ids per supported chunk p90", fmt(quantile(direct_term_counts, 0.90))],
        ["Rollup ontology nodes per supported chunk median", fmt(quantile(rollup_terms_per_chunk, 0.50))],
        ["Rollup ontology nodes per supported chunk p90", fmt(quantile(rollup_terms_per_chunk, 0.90))],
    ]

    support_bins = [
        ("0", lambda n: n == 0),
        ("1", lambda n: n == 1),
        ("2-4", lambda n: 2 <= n <= 4),
        ("5-9", lambda n: 5 <= n <= 9),
        ("10-24", lambda n: 10 <= n <= 24),
        ("25-49", lambda n: 25 <= n <= 49),
        ("50-99", lambda n: 50 <= n <= 99),
        ("100-249", lambda n: 100 <= n <= 249),
        ("250-499", lambda n: 250 <= n <= 499),
        ("500+", lambda n: n >= 500),
    ]
    support_bin_counts = [(name, sum(1 for v in support_values if pred(v))) for name, pred in support_bins]
    max_bin = max((n for _, n in support_bin_counts), default=1)
    support_bin_rows = [[name, fmt(n), bar(n, max_bin)] for name, n in support_bin_counts]

    depth_buckets: dict[int, list[str]] = defaultdict(list)
    for fid in ids:
        depth_buckets[depths.get(fid, 0)].append(fid)
    depth_rows = []
    max_depth_count = max((len(v) for v in depth_buckets.values()), default=1)
    for depth in sorted(depth_buckets):
        bucket = set(depth_buckets[depth])
        supported = len(bucket & supported_nodes)
        empty = len(bucket - supported_nodes)
        depth_rows.append([
            fmt(depth),
            fmt(len(bucket)),
            fmt(supported),
            pct(supported, len(bucket)),
            fmt(empty),
            bar(len(bucket), max_depth_count),
        ])

    fanout_values = [child_counts[fid] for fid in ids]
    fanout_rows = [
        ["Fan-out max", fmt(max(fanout_values) if fanout_values else 0)],
        ["Fan-out median", fmt(quantile(fanout_values, 0.50))],
        ["Fan-out p90", fmt(quantile(fanout_values, 0.90))],
        ["Nodes with exactly 1 child", fmt(sum(1 for n in fanout_values if n == 1))],
        ["Nodes with >= 10 children", fmt(sum(1 for n in fanout_values if n >= 10))],
        ["Nodes with >= 50 children", fmt(sum(1 for n in fanout_values if n >= 50))],
    ]

    dag_rows = [
        ["Nodes with multiple parents", fmt(sum(1 for fid in ids if parent_counts[fid] > 1))],
        ["Supported nodes with multiple parents", fmt(sum(1 for fid in supported_nodes if parent_counts[fid] > 1))],
        ["Leaf nodes with multiple parents", fmt(sum(1 for fid in leaf_nodes if parent_counts[fid] > 1))],
        ["Max parents for one node", fmt(max(parent_counts.values()) if parent_counts else 0)],
    ]

    def top_rows(nodes: list[str], cols: list[str]) -> list[list[object]]:
        rows = []
        for fid in nodes:
            data = {
                "label": label(api, fid),
                "id": f"<code>{html.escape(fid)}</code>",
                "rollup_chunks": fmt(rollup_counts[fid]),
                "direct_chunks": fmt(direct_counts[fid]),
                "children": fmt(child_counts[fid]),
                "parents": fmt(parent_counts[fid]),
                "depth": fmt(depths.get(fid, 0)),
                "descendants": fmt(descendant_counts[fid]),
            }
            rows.append([data[col] for col in cols])
        return rows

    top_rollup = sorted(ids, key=lambda fid: rollup_counts[fid], reverse=True)[:25]
    top_direct = sorted(ids, key=lambda fid: direct_counts[fid], reverse=True)[:25]
    top_empty_internal = sorted(
        [fid for fid in empty_nodes if child_counts[fid] > 0],
        key=lambda fid: (descendant_counts[fid], child_counts[fid]),
        reverse=True,
    )[:25]
    top_fanout = sorted(ids, key=lambda fid: child_counts[fid], reverse=True)[:25]
    top_multiparent = sorted(
        [fid for fid in ids if parent_counts[fid] > 1],
        key=lambda fid: (parent_counts[fid], rollup_counts[fid]),
        reverse=True,
    )[:25]

    tree_root = "FoodOn OWL"
    tree_children: dict[str, list[str]] = defaultdict(list)
    tree_children[tree_root] = roots
    for parent, kids in children_all.items():
        tree_children[parent].extend(kids)
    tree_counts = {fid: rollup_counts.get(fid, 0) for fid in ids}
    tree_counts[tree_root] = len(chunk_terms)
    tree_html = render_tree(api, tree_root, tree_children, tree_counts)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FoodOn OWL Structure and Corpus Link Metrics</title>
<style>
body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;margin:1.5rem auto;max-width:1500px;padding:0 1rem;color:#24313f;}}
h1{{border-bottom:3px solid #4d6f8c;padding-bottom:.35rem;}}
h2{{margin-top:1.4rem;color:#2e4f65;}}
h3{{margin:.2rem 0 .45rem;color:#35586d;font-size:1rem;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem;align-items:start;}}
.panel{{border:1px solid #d9e0e6;border-radius:8px;padding:.75rem;background:#fff;overflow-x:auto;}}
.wide{{grid-column:1/-1;}}
.wide-table table{{min-width:980px;}}
table{{border-collapse:collapse;width:100%;font-size:.84rem;}}
th,td{{border:1px solid #dfe5ea;padding:.28rem .38rem;text-align:left;vertical-align:top;}}
th{{background:#f3f6f8;color:#2d4658;}}
code{{font-size:.78rem;color:#31506a;}}
.bar{{position:relative;background:#eef2f5;border-radius:4px;min-height:1.15rem;overflow:hidden;}}
.bar span{{position:absolute;inset:0 auto 0 0;background:#9fc2ba;}}
.bar.pct span{{background:#89a8c8;}}
.bar b{{position:relative;padding-left:.3rem;font-weight:500;}}
ul.tree,ul.tree ul{{list-style:none;margin:0;padding-left:.9rem;border-left:1px dotted #c8d0d8;}}
ul.tree{{padding-left:0;border-left:none;max-height:78vh;overflow:auto;}}
ul.tree li{{margin:.1rem 0;font-size:.84rem;}}
ul.tree summary{{cursor:pointer;}}
li.empty>*{{color:#a2a9b0;}}
.c{{color:#85919b;font-size:.82em;margin-left:.35rem;}}
.more{{color:#b95050;font-style:italic;}}
.note{{color:#697783;font-size:.88rem;}}
.explain{{color:#526475;font-size:.88rem;margin:.25rem 0 .65rem;}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.7rem;margin:1rem 0;}}
.tile{{border:1px solid #d9e0e6;border-radius:8px;padding:.65rem;background:#f8fafb;}}
.tile b{{display:block;font-size:1.25rem;color:#28475d;}}
.tile span{{color:#667684;font-size:.82rem;}}
</style>
</head>
<body>
<h1>FoodOn OWL Structure and Corpus Link Metrics</h1>
<p class="note">Counts are computed from all ontology terms loaded from the FoodOn OWL file, including non-FOODON prefixes. Direct evidence counts all chunk ontology links present in the loaded ontology. Rollup chunks count chunks attached to a node or any of its descendants.</p>

<div class="kpi">
  <div class="tile"><b>{fmt(len(ids))}</b><span>loaded ontology terms</span></div>
  <div class="tile"><b>{fmt(len(supported_nodes))}</b><span>rollup-supported nodes</span></div>
  <div class="tile"><b>{fmt(len(empty_nodes))}</b><span>empty ontology nodes</span></div>
  <div class="tile"><b>{fmt(len(chunk_terms))}</b><span>chunks with linked ontology evidence</span></div>
</div>

<div class="grid">
<section class="panel">
<h3>Ontology Composition</h3>
<p class="explain">How much of the loaded OWL term set is native FOODON vocabulary versus imported or referenced ontology vocabulary.</p>
{table(["group", "terms", "share"], composition_rows)}
</section>
<section class="panel">
<h3>Chunk Evidence Coverage</h3>
<p class="explain">A chunk is counted as covered when it has at least one linked ontology id present in the loaded ontology.</p>
{table(["chunk group", "chunks", "share"], chunk_evidence_rows)}
</section>
<section class="panel">
<h3>Node Support Coverage</h3>
<p class="explain">A supported node has at least one chunk attached directly to it or to one of its descendants.</p>
{table(["node group", "nodes", "share"], node_support_rows)}
</section>
</div>

<div class="grid">
<section class="panel">
<h3>Overview</h3>
<p class="explain">Headline counts for ontology size, corpus coverage, and supported versus empty nodes.</p>
{table(["metric", "value"], overview_rows)}
</section>
<section class="panel">
<h3>Support Summary</h3>
<p class="explain">Distribution statistics for rolled-up chunk support across ontology nodes and per-chunk ontology breadth.</p>
{table(["support metric", "value"], support_summary_rows)}
</section>
<section class="panel">
<h3>Structure Summary</h3>
<p class="explain">Fan-out and multi-parent metrics describe how tree-like or graph-like the loaded ontology structure is.</p>
{table(["structure metric", "value"], fanout_rows + dag_rows)}
</section>
</div>

<section class="panel wide wide-table">
<h2>Scope Support Split</h2>
<p class="explain">This table separates leaf versus intermediate ontology nodes, and empty versus supported nodes, for the whole loaded ontology and key subsets.</p>
{table(["scope", "nodes", "leaf nodes", "intermediate nodes", "rollup-supported", "supported %", "direct-supported", "empty", "empty leaves", "empty intermediate"], split_rows)}
</section>

<div class="grid">
<section class="panel">
<h2>Support Rate By Scope</h2>
<p class="explain">The same scope split as above, summarized visually by percentage of nodes with rollup support.</p>
{table(["scope", "supported %", "bar"], scope_support_chart_rows)}
</section>
<section class="panel">
<h2>Top Prefixes By Direct Chunk Coverage</h2>
<p class="explain">Prefixes are ranked by how many chunks link directly to at least one id in that prefix. A chunk may contribute to more than one prefix.</p>
{table(["prefix", "direct chunks", "share of covered chunks"], prefix_chart_rows)}
</section>
</div>

<h2>Ontology Tree With Corpus Link Support</h2>
<section class="panel wide">
<p class="note">Rendered as a single-parent tree for readability. Nodes with multiple ontology parents are placed under one nearest parent in this view. The counts remain complete rollup support counts.</p>
{tree_html}
</section>

<div class="grid">
<section class="panel wide wide-table">
<h2>Prefix Coverage</h2>
<p class="explain">Prefix-level support separates ontology vocabulary size from actual corpus attachment. Direct chunks count exact linked ids; rollup chunks include ancestor propagation inside that prefix.</p>
{table(["prefix", "ontology terms", "direct-supported ids", "rollup-supported nodes", "direct chunks", "rollup chunks", "supported %"], prefix_rows)}
</section>
<section class="panel">
<h2>Rollup Support Distribution</h2>
<p class="explain">Most ontology nodes have no corpus support; the nonzero bins show where support begins to concentrate.</p>
{table(["chunks on node", "nodes", "bar"], support_bin_rows)}
</section>
<section class="panel">
<h2>Depth Distribution</h2>
<p class="explain">Depth is computed from a representative longest path from ontology roots. Supported and empty counts show where corpus evidence lands by level.</p>
{table(["depth", "nodes", "supported", "supported %", "empty", "bar"], depth_rows)}
</section>
</div>

<div class="grid">
<section class="panel wide wide-table">
<h2>Top Nodes By Rollup Chunk Support</h2>
<p class="explain">Rollup support can make broad ancestors large even when direct attachment is low or zero.</p>
{table(["label", "id", "rollup chunks", "direct chunks", "children", "parents", "depth"], top_rows(top_rollup, ["label", "id", "rollup_chunks", "direct_chunks", "children", "parents", "depth"]))}
</section>
<section class="panel wide wide-table">
<h2>Top Nodes By Direct Corpus Attachment</h2>
<p class="explain">Direct attachment identifies the ontology ids actually linked by corpus annotations before ancestor rollup.</p>
{table(["label", "id", "direct chunks", "rollup chunks", "children", "parents", "depth"], top_rows(top_direct, ["label", "id", "direct_chunks", "rollup_chunks", "children", "parents", "depth"]))}
</section>
<section class="panel wide wide-table">
<h2>Largest Empty Intermediate Nodes</h2>
<p class="explain">These nodes have children or descendants but no rolled-up corpus support in the current corpus snapshot.</p>
{table(["label", "id", "descendants", "children", "parents", "depth"], top_rows(top_empty_internal, ["label", "id", "descendants", "children", "parents", "depth"]))}
</section>
<section class="panel wide wide-table">
<h2>Highest Fan-Out Nodes</h2>
<p class="explain">High fan-out nodes are structurally broad and can create large navigation menus if used directly as browse shelves.</p>
{table(["label", "id", "children", "rollup chunks", "direct chunks", "parents", "depth"], top_rows(top_fanout, ["label", "id", "children", "rollup_chunks", "direct_chunks", "parents", "depth"]))}
</section>
<section class="panel wide wide-table">
<h2>Multi-Parent Nodes</h2>
<p class="explain">Nodes with multiple parents show where the ontology is graph-like rather than a strict tree. The report tree chooses one display parent, but this table preserves the multi-parent signal.</p>
{table(["label", "id", "parents", "rollup chunks", "direct chunks", "children", "depth"], top_rows(top_multiparent, ["label", "id", "parents", "rollup_chunks", "direct_chunks", "children", "depth"]))}
</section>
</div>
</body>
</html>"""

    OUT.write_text(html_doc, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
