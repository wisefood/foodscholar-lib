"""Assemble notebooks/projection_bakeoff.ipynb from source cells.

Compares competing Layer-A projection methodologies on the foods facet, side by
side, judged by eye. Kept as a script so the notebook source stays reviewable
and regenerable. Run with the foodscholar env:

    /mnt/miniconda3/envs/foodscholar/bin/python scripts/build_projection_bakeoff_nb.py

Spec: docs/superpowers/specs/2026-05-28-layer-a-projection-bakeoff-design.md
"""

from __future__ import annotations

import nbformat as nbf

NB_PATH = "notebooks/projection_bakeoff.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells: list = []

# ----------------------------------------------------------------- title
cells.append(
    md(
        """# Layer-A projection bake-off — foods facet

The current projection produces a flat, un-navigable foods facet, and the
re-tiering patch made it worse. The methodology is wrong for *browsing*, so this
notebook renders **competing projection methodologies on the same foods data,
side by side, judged by eye**. No production code until one wins.

**The reframe under test:** stop deriving the browse tree from corpus support.
Choose a category **backbone** first (designed for browsing), then **attach**
evidence by lifting each chunk to its nearest backbone node. Support *decorates*
the tree; it doesn't *define* it.

| Col | Methodology |
|-----|-------------|
| **0 — Baseline** | current `build_layer_a` (the flat blob, reference) |
| **2 — Structural cut** | fixed-depth FoodOn cut, keep real tiers regardless of support |
| **1a — Auto backbone** | structural-rule backbone + support decorates |
| **1a+ — Auto backbone + controlled expansion** | same top backbone, recursively opens only supported/capped tiers |
| **1b — LLM backbone** | `llama-3.1-8b-instant` proposes the backbone + support decorates |
| **3 — Multi-facet** | a node may sit under several backbone axes (DAG) |

FoodOn stays the entity backbone — only the projection changes. Runs in-process
(no Neo4j/Elastic). Col 1b needs `GROQ_API_KEY`; degrades gracefully."""
    )
)

# ----------------------------------------------------------------- §0 foundation
cells.append(md("## §0 — Load foods evidence + shared helpers"))

cells.append(
    code(
        '''import html as _html
import os
from collections import Counter, defaultdict
from pathlib import Path

from foodscholar.config import FoodScholarConfig
from foodscholar.facade import FoodScholar
from foodscholar.layer_a.facet import route_link_to_facet

HERE = Path.cwd()
ROOT = HERE if (HERE / "data" / "foodon.owl").exists() else HERE.parent
VIZ_DIR = ROOT / "data" / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

HAVE_GROQ = bool(os.environ.get("GROQ_API_KEY"))
_cfg = {
    "corpus": {
        "chunks_path": str(ROOT / "tests/fixtures/sample_chunks.jsonl"),
        "annotated_snapshot_path": str(ROOT / "data/annotated.parquet"),
    },
    "ontology": {
        "foodon_path": str(ROOT / "data/foodon.owl"),
        "cache_path": str(ROOT / "data/foodon_cache.parquet"),
        "prefix_filter": ["FOODON:"],
    },
    "layer_a": {"facets": ["foods"], "min_support": 25, "max_depth": 6,
                "blacklist_terms": ["material entity", "physical object", "manufactured product"]},
    "storage": {"chunk_store": {"backend": "memory"}, "graph_store": {"backend": "memory"}},
}
if HAVE_GROQ:
    _cfg["llm"] = {"primary": {"provider": "groq", "model": "llama-3.1-8b-instant"}}

fs = FoodScholar.from_config(FoodScholarConfig.model_validate(_cfg))
api = fs.load_ontology()
fs.attach_ontology(api)
fs.load_chunks(str(ROOT / "data/annotated.parquet"))
chunks = list(fs.chunk_store.scan())

FOOD_PRODUCT = api.name_to_id("food product")  # FOODON:00001002

# Per-chunk foods FoodOn ids (direct evidence): foodon_ids + foods-facet links.
def chunk_food_terms(c):
    fids = set()
    for fid in getattr(c, "foodon_ids", []) or []:
        if fid in api:
            fids.add(fid)
    for ln in getattr(c, "entity_links", []) or []:
        if ln.ontology_id in api and route_link_to_facet(ln) == "foods":
            fids.add(ln.ontology_id)
    # keep only terms under food product (the foods sub-ontology)
    return {f for f in fids if f == FOOD_PRODUCT or api.is_subclass_of(f, FOOD_PRODUCT)}


CHUNK_TERMS = {c.chunk_id: chunk_food_terms(c) for c in chunks}
CHUNK_TERMS = {cid: t for cid, t in CHUNK_TERMS.items() if t}
TERM_DOC_FREQ = Counter()
for t in CHUNK_TERMS.values():
    for fid in t:
        TERM_DOC_FREQ[fid] += 1
print(f"{len(chunks)} chunks · {len(CHUNK_TERMS)} with food terms · "
      f"{len(TERM_DOC_FREQ)} distinct FoodOn terms · llm={fs.llm.model_id}")'''
    )
)

# shared lift + render + stats helpers
cells.append(
    code(
        r'''# ---- lift evidence onto a backbone ------------------------------------------
def lift_to_backbone(backbone_ids):
    """For each chunk term, find the nearest (most-specific) backbone ancestor.

    Returns:
      homed[backbone_id] -> set(chunk_id)     (chunk counted once per backbone it hits)
      multi_home[chunk_id] -> set(backbone_id) for chunks hitting >1 backbone
      unhomed -> set(chunk_id) with terms but no backbone ancestor
    """
    backbone = set(backbone_ids)
    # precompute nearest-backbone for each distinct term
    term_backbones = {}
    for fid in TERM_DOC_FREQ:
        if fid in backbone:
            term_backbones[fid] = {fid}
            continue
        ancs = [a for a in api.id_to_ancestors(fid) if a in backbone]
        if not ancs:
            term_backbones[fid] = set()
            continue
        # most-specific = the backbone ancestor with the FEWEST descendants in backbone
        # (closest to the term); approximate by max ontology-depth among ancs.
        deepest = max(ancs, key=lambda a: len(api.id_to_ancestors(a)))
        term_backbones[fid] = {deepest}

    homed = defaultdict(set)
    multi_home = {}
    unhomed = set()
    for cid, terms in CHUNK_TERMS.items():
        hits = set()
        for fid in terms:
            hits |= term_backbones.get(fid, set())
        if not hits:
            unhomed.add(cid)
            continue
        for b in hits:
            homed[b].add(cid)
        if len(hits) > 1:
            multi_home[cid] = hits
    return homed, multi_home, unhomed


# ---- tree rendering ----------------------------------------------------------
def render_tree_from_edges(root_id, children_map, count_map, label_fn,
                           max_depth=4, open_depth=1, max_children=18):
    """Generic nested <details> renderer over an explicit children_map."""
    def node(nid, rel):
        kids = sorted(children_map.get(nid, []), key=lambda c: -count_map.get(c, 0))
        nch = len(kids)
        cnt = count_map.get(nid, 0)
        badge = f"<span class='c'>{nch} sub · {cnt:,} chunks</span>"
        empty = " empty" if cnt == 0 else ""
        label = _html.escape(label_fn(nid))
        if nch == 0 or rel >= max_depth:
            return f"<li class='{empty}'>{label}{badge}</li>"
        shown = kids[:max_children]
        hidden = nch - len(shown)
        inner = "".join(node(c, rel + 1) for c in shown)
        if hidden:
            inner += f"<li class='m'>+{hidden} more…</li>"
        op = " open" if rel < open_depth else ""
        return f"<li class='{empty}'><details{op}><summary><b>{label}</b>{badge}</summary><ul>{inner}</ul></details></li>"
    return f"<ul class='ftree'>{node(root_id, 0)}</ul>"


def stats_line(top_fanout, depth, homed_chunks, total_chunks, n_empty, n_multi):
    pct = 100 * homed_chunks / total_chunks if total_chunks else 0
    return (f"top fan-out <b>{top_fanout}</b> · depth <b>{depth}</b> · "
            f"<b>{pct:.0f}%</b> chunks homed · <b>{n_empty}</b> empty cats · "
            f"<b>{n_multi}</b> multi-home chunks")


COLUMNS = []  # each: {"title","stats","tree"}
TOTAL_FOOD_CHUNKS = len(CHUNK_TERMS)
print("helpers ready")'''
    )
)

# ----------------------------------------------------------------- Col 0 baseline
cells.append(md("## Col 0 — Baseline (current `build_layer_a`)"))

cells.append(
    code(
        '''fs.build_layer_a()
base_shelves = [s for s in fs.graph_store.list_shelves() if s.facet == "foods"]
base_by_id = {s.shelf_id: s for s in base_shelves}
base_children = defaultdict(list)
for s in base_shelves:
    if s.parent_shelf_id:
        base_children[s.parent_shelf_id].append(s.shelf_id)
base_counts = {s.shelf_id: s.chunk_count for s in base_shelves}
base_root = next((s.shelf_id for s in base_shelves if s.parent_shelf_id is None), None)


def base_label(sid):
    return base_by_id[sid].label if sid in base_by_id else sid


base_tree = render_tree_from_edges(base_root, base_children, base_counts, base_label)
base_fanout = max((len(v) for v in base_children.values()), default=0)
base_depth = max((s.depth for s in base_shelves), default=0)
base_empty = sum(1 for s in base_shelves if s.chunk_count == 0)
COLUMNS.append({
    "title": "0 — Baseline (current build)",
    "stats": stats_line(base_fanout, base_depth, TOTAL_FOOD_CHUNKS, TOTAL_FOOD_CHUNKS, base_empty, 0),
    "tree": base_tree,
})
print(f"baseline: {len(base_shelves)} shelves, top fan-out {base_fanout}, depth {base_depth}")'''
    )
)

# ----------------------------------------------------------------- Col 2 structural cut
cells.append(
    md(
        """## Col 2 — Structural cut (no support collapse)

Take FoodOn's real hierarchy under `food product` and cut it at a fixed depth
horizon, keeping the **real intermediate tiers** regardless of support. Parent
edges are FoodOn's own is-a edges (single-parent pick for the tree), not
nearest-*surviving*-ancestor. Counts are evidence lifted to each node."""
    )
)

cells.append(
    code(
        '''# Real FoodOn subtree under food product, capped at CUT_DEPTH levels.
CUT_DEPTH = 3  # levels below food product to keep as the browse tree

# Build the node set: food product + descendants within CUT_DEPTH.
cut_nodes = {FOOD_PRODUCT}
frontier = {FOOD_PRODUCT}
for _ in range(CUT_DEPTH):
    nxt = set()
    for n in frontier:
        for c in api.id_to_children(n):
            if c in api:
                cut_nodes.add(c)
                nxt.add(c)
    frontier = nxt

# Single-parent tree edges: each node's parent = its FoodOn parent that's in cut_nodes.
cut_children = defaultdict(list)
for n in cut_nodes:
    if n == FOOD_PRODUCT:
        continue
    parents = [p for p in api.id_to_parents(n) if p in cut_nodes]
    parent = parents[0] if parents else FOOD_PRODUCT
    cut_children[parent].append(n)

# Counts: lift every chunk term to the deepest cut-node ancestor it has.
cut_counts = Counter()
for cid, terms in CHUNK_TERMS.items():
    nodes = set()
    for fid in terms:
        ancs = [a for a in ([fid] + api.id_to_ancestors(fid)) if a in cut_nodes]
        if ancs:
            nodes.add(max(ancs, key=lambda a: len(api.id_to_ancestors(a))))
    for nd in nodes:
        cut_counts[nd] += 1

cut_tree = render_tree_from_edges(FOOD_PRODUCT, cut_children, cut_counts,
                                  lambda n: api.id_to_label(n) or n)
cut_fanout = max((len(v) for v in cut_children.values()), default=0)
cut_empty = sum(1 for n in cut_nodes if cut_counts.get(n, 0) == 0)
homed = sum(1 for cid, terms in CHUNK_TERMS.items()
            if any(a in cut_nodes for fid in terms for a in [fid] + api.id_to_ancestors(fid)))
COLUMNS.append({
    "title": f"2 — Structural cut (depth {CUT_DEPTH})",
    "stats": stats_line(cut_fanout, CUT_DEPTH, homed, TOTAL_FOOD_CHUNKS, cut_empty, 0),
    "tree": cut_tree,
})
print(f"structural cut: {len(cut_nodes)} nodes, top fan-out {cut_fanout}, {cut_empty} empty")'''
    )
)

# ----------------------------------------------------------------- Col 1a/1b backbone
cells.append(
    md(
        """## Col 1a / 1b — Backbone-first (support decorates)

Choose a small top-level category backbone, then lift every chunk to its nearest
backbone node. The tree is exactly two tiers (backbone → its FoodOn children
that have evidence), so it's browsable by construction. **1a** picks the
backbone by a structural rule; **1b** asks the LLM to propose it."""
    )
)

cells.append(
    code(
        '''def backbone_column(title, backbone_ids):
    backbone_ids = [b for b in backbone_ids if b in api]
    homed, multi_home, unhomed = lift_to_backbone(backbone_ids)
    # tree: synthetic root -> backbone nodes -> (their direct FoodOn children w/ evidence)
    ROOT = "__backbone_root__"
    children = defaultdict(list)
    counts = {}
    labels = {ROOT: "foods"}
    counts[ROOT] = sum(len(v) for v in homed.values())
    for b in backbone_ids:
        children[ROOT].append(b)
        counts[b] = len(homed.get(b, set()))
        labels[b] = api.id_to_label(b) or b
        # second tier: backbone's FoodOn children that themselves got evidence
        for c in api.id_to_children(b):
            ev = sum(1 for cid, terms in CHUNK_TERMS.items()
                     if c in terms or any(c == a for fid in terms for a in api.id_to_ancestors(fid)))
            if ev > 0:
                children[b].append(c)
                counts[c] = ev
                labels[c] = api.id_to_label(c) or c

    tree = render_tree_from_edges(ROOT, children, counts, lambda n: labels.get(n, n),
                                  max_depth=2, open_depth=1)
    fanout = max(len(children.get(ROOT, [])), max((len(children.get(b, [])) for b in backbone_ids), default=0))
    n_empty = sum(1 for b in backbone_ids if counts.get(b, 0) == 0)
    homed_chunks = TOTAL_FOOD_CHUNKS - len(unhomed)
    COLUMNS.append({
        "title": title,
        "stats": stats_line(fanout, 2, homed_chunks, TOTAL_FOOD_CHUNKS, n_empty, len(multi_home)),
        "tree": tree,
    })
    return multi_home


# 1a — auto backbone = the direct children of food product (10 real FoodOn cats).
auto_backbone = api.id_to_children(FOOD_PRODUCT)
MULTI_HOME_1A = backbone_column(f"1a — Auto backbone ({len(auto_backbone)} cats)", auto_backbone)
print(f"1a auto backbone: {[api.id_to_label(b) for b in auto_backbone]}")
print(f"1a multi-home chunks: {len(MULTI_HOME_1A)} / {TOTAL_FOOD_CHUNKS}")'''
    )
)

cells.append(
    md(
        """## Col 1a+ — Auto backbone + controlled expansion

Keep the exact same auto backbone as 1a, but make large buckets less opaque by
opening deeper FoodOn tiers under strict browse constraints. This is still a
projection experiment: support controls visibility, caps prevent runaway
fan-out, and low-value single-child chains may be skipped."""
    )
)

cells.append(
    code(
        '''# 1a+ — same auto backbone, but expand large buckets with constrained tiers.
# Counts are chunk support for each FoodOn node or any of its descendants.
NODE_CHUNKS = defaultdict(set)
TERM_ROLLUPS = {}
for fid in TERM_DOC_FREQ:
    rollups = [fid] + [
        a for a in api.id_to_ancestors(fid)
        if a == FOOD_PRODUCT or api.is_subclass_of(a, FOOD_PRODUCT)
    ]
    TERM_ROLLUPS[fid] = rollups

for cid, terms in CHUNK_TERMS.items():
    for fid in terms:
        for nd in TERM_ROLLUPS.get(fid, ()):
            NODE_CHUNKS[nd].add(cid)


EXPAND_MIN_CHUNKS = 25
EXPAND_MAX_DEPTH = 6       # root -> backbone -> ... -> concrete high-support terms
EXPAND_MAX_CHILDREN = 12   # hard cap per opened parent
EXPAND_MIN_CHILDREN = 2    # skip pure filing chains when possible


def node_support(fid):
    return len(NODE_CHUNKS.get(fid, set()))


def supported_children(fid):
    kids = []
    for child in api.id_to_children(fid):
        if child not in api:
            continue
        support = node_support(child)
        if support >= EXPAND_MIN_CHUNKS:
            kids.append(child)
    return kids


def evidence_descendant_count(fid):
    """How many distinct directly mentioned FoodOn terms live under this node."""
    return sum(
        1 for term_id, n in TERM_DOC_FREQ.items()
        if n > 0 and (term_id == fid or api.is_subclass_of(term_id, fid))
    )


def display_rank(fid):
    # Prefer useful grouping nodes first, then high-support concrete terms.
    return (
        node_support(fid),
        evidence_descendant_count(fid),
        TERM_DOC_FREQ.get(fid, 0),
        -(len(api.id_to_label(fid) or fid)),
    )


def collapsed_supported_children(fid, *, depth_left):
    """Return display children, skipping low-value single-child filing chains.

    If FoodOn says A -> B -> C but B is the only supported child and B is not
    directly evidenced, the projection may show C under A. This keeps navigation
    shorter while all displayed nodes remain real FoodOn ids.
    """
    kids = supported_children(fid)
    while (
        depth_left > 1
        and len(kids) == 1
        and TERM_DOC_FREQ.get(kids[0], 0) == 0
        and len(supported_children(kids[0])) >= EXPAND_MIN_CHILDREN
    ):
        fid = kids[0]
        kids = supported_children(fid)
        depth_left -= 1
    return sorted(kids, key=display_rank, reverse=True)[:EXPAND_MAX_CHILDREN]


def controlled_backbone_column(title, backbone_ids):
    backbone_ids = [b for b in backbone_ids if b in api]
    homed, multi_home, unhomed = lift_to_backbone(backbone_ids)

    ROOT = "__controlled_backbone_root__"
    children = defaultdict(list)
    counts = {ROOT: sum(len(v) for v in homed.values())}
    labels = {ROOT: "foods"}
    seen_edges = set()

    def add_edge(parent, child):
        edge = (parent, child)
        if edge not in seen_edges:
            children[parent].append(child)
            seen_edges.add(edge)

    def expand(parent, rel_depth):
        if rel_depth >= EXPAND_MAX_DEPTH:
            return
        depth_left = EXPAND_MAX_DEPTH - rel_depth
        kids = collapsed_supported_children(parent, depth_left=depth_left)
        for child in kids:
            add_edge(parent, child)
            counts[child] = node_support(child)
            labels[child] = api.id_to_label(child) or child
            expand(child, rel_depth + 1)

    for b in sorted(backbone_ids, key=display_rank, reverse=True):
        add_edge(ROOT, b)
        counts[b] = len(homed.get(b, set()))
        labels[b] = api.id_to_label(b) or b
        expand(b, 1)

    tree = render_tree_from_edges(
        ROOT,
        children,
        counts,
        lambda n: labels.get(n, n),
        max_depth=EXPAND_MAX_DEPTH,
        open_depth=2,
        max_children=EXPAND_MAX_CHILDREN,
    )
    fanout = max((len(v) for v in children.values()), default=0)
    n_empty = sum(1 for b in backbone_ids if counts.get(b, 0) == 0)
    homed_chunks = TOTAL_FOOD_CHUNKS - len(unhomed)
    COLUMNS.append({
        "title": title,
        "stats": (
            stats_line(fanout, EXPAND_MAX_DEPTH, homed_chunks, TOTAL_FOOD_CHUNKS,
                       n_empty, len(multi_home))
            + f"<br>min support {EXPAND_MIN_CHUNKS} · cap {EXPAND_MAX_CHILDREN}/parent"
        ),
        "tree": tree,
    })
    return multi_home


MULTI_HOME_1A_PLUS = controlled_backbone_column(
    f"1a+ — Auto backbone + controlled expansion ({len(auto_backbone)} cats)",
    auto_backbone,
)
print(
    "1a+ controlled expansion: "
    f"fan-out cap {EXPAND_MAX_CHILDREN}, min chunks {EXPAND_MIN_CHUNKS}, "
    f"multi-home chunks {len(MULTI_HOME_1A_PLUS)} / {TOTAL_FOOD_CHUNKS}"
)'''
    )
)

cells.append(
    code(
        r'''# 1b — LLM-proposed backbone. The LLM proposes intuitive top-level food
# categories BY NAME; we resolve each to a real FoodOn id (drop unresolved) so
# we never leave FoodOn.
LLM_BACKBONE = []
if not HAVE_GROQ:
    print("GROQ_API_KEY not set — skipping 1b (LLM backbone). Set it and re-run.")
else:
    BACKBONE_SCHEMA = {
        "type": "object",
        "properties": {"categories": {"type": "array", "items": {"type": "string"}}},
        "required": ["categories"],
    }
    # Give the model the available FoodOn vocabulary cues: the 10 real children +
    # a sample of high-frequency linked terms, so its names map back cleanly.
    sample_terms = [api.id_to_label(fid) for fid, _ in TERM_DOC_FREQ.most_common(40)]
    prompt = (
        "Propose 12-20 intuitive TOP-LEVEL food categories for browsing a "
        "nutrition knowledge base. Use common, human category names (e.g. "
        "'vegetables', 'fruits', 'dairy', 'grains', 'meat', 'legumes', 'nuts and "
        "seeds', 'seafood', 'beverages', 'oils and fats'). They must correspond "
        "to real food groupings. Here are frequent foods in the corpus for "
        f"context:\n{', '.join(sample_terms)}\n\n"
        'Return JSON {"categories": ["...", ...]}.'
    )
    obj = fs.llm.generate_json(prompt, BACKBONE_SCHEMA, max_tokens=512)
    proposed = obj.get("categories", [])
    for name in proposed:
        fid = api.name_to_id(name) or api.name_to_id(name.rstrip("s")) or api.name_to_id(name + " food product")
        if fid is None:
            hits = api.search(name, limit=1)
            fid = hits[0] if hits else None
        if fid and (fid == FOOD_PRODUCT or api.is_subclass_of(fid, FOOD_PRODUCT)):
            LLM_BACKBONE.append(fid)
    LLM_BACKBONE = list(dict.fromkeys(LLM_BACKBONE))  # dedupe, keep order
    print(f"LLM proposed {len(proposed)} names → {len(LLM_BACKBONE)} resolved to FoodOn ids")
    MULTI_HOME_1B = backbone_column(f"1b — LLM backbone ({len(LLM_BACKBONE)} cats)", LLM_BACKBONE)
    print("resolved:", [api.id_to_label(b) for b in LLM_BACKBONE])'''
    )
)

# ----------------------------------------------------------------- Col 3 multi-facet
cells.append(
    md(
        """## Col 3 — Multi-facet (DAG), informed by multi-home stats

If chunks are frequently multi-home (Col 1a's stat), a single tree is lossy.
Here a chunk attaches to **all** applicable backbone categories — the browse
structure is a small DAG. We render it as the same backbone but annotate how
much overlap there is, so 'tree vs DAG' is decided on the real numbers."""
    )
)

cells.append(
    code(
        '''auto_backbone = api.id_to_children(FOOD_PRODUCT)
homed, multi_home, unhomed = lift_to_backbone(auto_backbone)

# Multi-home now means: a chunk's DIFFERENT terms map to different backbones
# (already captured). For the DAG view, show each backbone with its full homed
# count (chunks counted under every category they touch — overlaps allowed).
ROOT = "__dag_root__"
children = defaultdict(list)
counts = {ROOT: TOTAL_FOOD_CHUNKS}
labels = {ROOT: "foods (multi-facet)"}
for b in auto_backbone:
    children[ROOT].append(b)
    counts[b] = len(homed.get(b, set()))
    labels[b] = api.id_to_label(b) or b

dag_tree = render_tree_from_edges(ROOT, children, counts, lambda n: labels.get(n, n),
                                  max_depth=1, open_depth=1)
# overlap stat: how many chunks land in >1 backbone, and the worst pair
pair = Counter()
for cid, bs in multi_home.items():
    for a in bs:
        for b2 in bs:
            if a < b2:
                pair[(a, b2)] += 1
top_pairs = pair.most_common(5)
overlap_html = "".join(
    f"<div class='path'>{api.id_to_label(a)} ∩ {api.id_to_label(b2)}: {n} chunks</div>"
    for (a, b2), n in top_pairs
)
COLUMNS.append({
    "title": "3 — Multi-facet (DAG)",
    "stats": (stats_line(len(auto_backbone), 1, TOTAL_FOOD_CHUNKS - len(unhomed),
                          TOTAL_FOOD_CHUNKS, 0, len(multi_home))
              + "<br><u>top category overlaps:</u>" + overlap_html),
    "tree": dag_tree,
})
print(f"multi-facet: {len(multi_home)} multi-home chunks; top overlaps: "
      + ", ".join(f"{api.id_to_label(a)}∩{api.id_to_label(b2)}={n}" for (a,b2),n in top_pairs))'''
    )
)

# ----------------------------------------------------------------- assembly
cells.append(md("## Assemble side-by-side report"))

cells.append(
    code(
        r'''from jinja2 import Template

REPORT = Template(
    """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Layer-A projection bake-off — foods</title><style>
  body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1500px;margin:1.5rem auto;padding:0 1rem;}
  h1{border-bottom:3px solid #4c72b0;padding-bottom:.3rem;}
  .grid{display:flex;gap:1rem;align-items:flex-start;overflow-x:auto;}
  .col{flex:1 0 320px;min-width:320px;border:1px solid #e0e4ea;border-radius:8px;padding:.6rem .8rem;}
  .col h2{font-size:1rem;color:#2a3f5f;margin:.2rem 0 .4rem;}
  .stats{font-size:.8rem;background:#f5f7fa;padding:.4rem .6rem;border-radius:5px;margin-bottom:.5rem;}
  ul.ftree,ul.ftree ul{list-style:none;margin:0;padding-left:.9rem;border-left:1px dotted #cbd2dc;}
  ul.ftree{padding-left:0;border-left:none;}
  ul.ftree li{margin:.1rem 0;font-size:.84rem;}
  ul.ftree summary{cursor:pointer;}
  li.empty>*{color:#aaa;}
  .c{color:#8a93a0;font-size:.8em;margin-left:.3rem;}
  .m{color:#c44e52;font-size:.82em;font-style:italic;}
  .path{font-size:.8rem;margin:.1rem 0;}
  .meta{color:#888;font-size:.85rem;}
</style></head><body>
<h1>Layer-A projection bake-off — foods facet</h1>
<p class="meta">{{ total }} chunks with food terms · model {{ model }} ·
all categories are real FoodOn ids · judge by eye{% if not have_groq %} · <i>1b skipped (no GROQ_API_KEY)</i>{% endif %}</p>
<div class="grid">
{% for c in columns %}
  <div class="col"><h2>{{ c.title }}</h2><div class="stats">{{ c.stats }}</div>{{ c.tree }}</div>
{% endfor %}
</div></body></html>"""
)

out = VIZ_DIR / "projection_bakeoff_foods.html"
out.write_text(
    REPORT.render(total=TOTAL_FOOD_CHUNKS, model=fs.llm.model_id,
                  have_groq=HAVE_GROQ, columns=COLUMNS),
    encoding="utf-8",
)
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, {len(COLUMNS)} columns)")'''
    )
)

# ----------------------------------------------------------------- build
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "foodscholar", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH} with {len(cells)} cells")
