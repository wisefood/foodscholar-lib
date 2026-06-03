"""Assemble notebooks/layer_a_method_bakeoff.ipynb from source cells.

Compares competing Layer-A construction methods on the foods facet, side by
side, with a metric-driven scorecard on top (coverage, findability, nameability,
fan-out, depth, faithfulness) plus the eyeball tree columns. Kept as a script so
the notebook source stays reviewable and regenerable. Run with the foodscholar
env:

    /mnt/miniconda3/envs/foodscholar/bin/python scripts/build_layer_a_method_bakeoff_nb.py

Specs: docs/methods_layer_a_bakeoff_brief.md +
docs/superpowers/plans/2026-06-02-layer-a-method-bakeoff-harness.md
"""

from __future__ import annotations

import nbformat as nbf

NB_PATH = "notebooks/layer_a_method_bakeoff.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells: list = []

# ----------------------------------------------------------------- title
cells.append(
    md(
        """# Layer-A method bake-off — foods facet

This notebook renders **competing Layer-A construction methods on the same foods
data, side by side**, with a **metric-driven scorecard on top** (coverage,
findability, nameability, fan-out, depth, faithfulness, llm-calls) so methods are
compared on numbers, not just by eye. No production code until one wins.

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
| **B-probe — Active FoodOn shelf communities** | Layer A as full FoodOn evidence; Layer B scopes by corpus support |

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


# Bake-off harness: wrap each method's tree into a MethodResult for the scorecard.
from foodscholar.layer_a.bakeoff.result import from_children_map, from_shelves

COLUMNS = []  # each: {"title","stats","tree"}
RESULTS = []  # each: a MethodResult, for the cross-method scorecard
MENTIONED = set(TERM_DOC_FREQ)  # corpus-mentioned food leaves (shared denominator)
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
RESULTS.append(from_shelves("0 — Baseline", base_shelves, ontology=api, mentioned_leaves=MENTIONED))
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
RESULTS.append(from_children_map(
    "2 — Structural cut", root=FOOD_PRODUCT, children_map=cut_children,
    counts=dict(cut_counts), labels={n: api.id_to_label(n) or n for n in cut_nodes},
    ontology=api, mentioned_leaves=MENTIONED,
))
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
    RESULTS.append(from_children_map(
        title.split(" —")[0].strip(), root=ROOT, children_map=children,
        counts=counts, labels=labels, ontology=api, mentioned_leaves=MENTIONED,
    ))
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
    RESULTS.append(from_children_map(
        title.split(" —")[0].strip(), root=ROOT, children_map=children,
        counts=counts, labels=labels, ontology=api, mentioned_leaves=MENTIONED,
    ))
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
DAG_ROOT = "__dag_root__"  # local synthetic root — must NOT shadow the global ROOT path
children = defaultdict(list)
counts = {DAG_ROOT: TOTAL_FOOD_CHUNKS}
labels = {DAG_ROOT: "foods (multi-facet)"}
for b in auto_backbone:
    children[DAG_ROOT].append(b)
    counts[b] = len(homed.get(b, set()))
    labels[b] = api.id_to_label(b) or b

dag_tree = render_tree_from_edges(DAG_ROOT, children, counts, lambda n: labels.get(n, n),
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
RESULTS.append(from_children_map(
    "3 — Multi-facet", root=DAG_ROOT, children_map=children, counts=counts,
    labels=labels, ontology=api, mentioned_leaves=MENTIONED,
))
print(f"multi-facet: {len(multi_home)} multi-home chunks; top overlaps: "
      + ", ".join(f"{api.id_to_label(a)}∩{api.id_to_label(b2)}={n}" for (a,b2),n in top_pairs))'''
    )
)

# ----------------------------------------------------------------- agentic (Plan B)
cells.append(
    md(
        """## Agentic MCP method (Plan B) — GROQ-gated

An LLM agent walks the FoodOn support DAG and makes local KEEP/COLLAPSE/REPARENT
decisions through a read-only tool layer. Membership stays is-a (this increment
shows relation bridges in the lens but does not yet act on them). Skipped without
`GROQ_API_KEY`."""
    )
)

cells.append(
    code(
        '''# Agentic method (Plan B). Uses the SAME food-product-filtered leaf universe
# (CHUNK_TERMS) as the other columns, so the scorecard denominator is consistent.
AGENTIC_RESULT = None
if not HAVE_GROQ:
    print("GROQ_API_KEY not set — skipping agentic method.")
else:
    from foodscholar.layer_a.bakeoff.agentic.relations import load_relation_index
    from foodscholar.layer_a.bakeoff.agentic.agent import build_agentic_result

    agentic_leaf_chunks = defaultdict(set)
    for cid, terms in CHUNK_TERMS.items():
        for fid in terms:
            agentic_leaf_chunks[fid].add(cid)
    # NB: the multi-facet cell rebinds the global ROOT to "__dag_root__", so derive
    # the repo root from HERE (Path.cwd(), never rebound) rather than ROOT.
    repo_root = HERE if (HERE / "data" / "foodon.owl").exists() else HERE.parent
    rel_index = load_relation_index(str(repo_root / "data" / "foodon.owl"))
    print(f"relation index: {len(rel_index)} FOODON terms with non-is-a relations")
    AGENTIC_RESULT = build_agentic_result(
        dict(agentic_leaf_chunks), api, relation_index=rel_index, llm=fs.llm,
        root=FOOD_PRODUCT, min_support=25, max_depth=6, max_children=12,
    )
    RESULTS.append(AGENTIC_RESULT)
    print(f"agentic: {len(AGENTIC_RESULT.edges)} internal nodes, "
          f"{AGENTIC_RESULT.llm_calls} llm calls, {len(AGENTIC_RESULT.leaf_home)} leaves homed")

    # Side-by-side eyeball column: render the agent-built tree (mirrors the DAG col).
    from foodscholar.layer_a.bakeoff.result import node_depths

    _agentic_depths = node_depths(AGENTIC_RESULT)
    _agentic_homed_chunks = len({
        c for fid in AGENTIC_RESULT.leaf_home for c in agentic_leaf_chunks.get(fid, ())
    })
    agentic_tree = render_tree_from_edges(
        AGENTIC_RESULT.root, AGENTIC_RESULT.edges, AGENTIC_RESULT.counts,
        lambda n: AGENTIC_RESULT.labels.get(n) or api.id_to_label(n) or n,
        max_depth=4, open_depth=1,
    )
    COLUMNS.append({
        "title": f"agentic (Plan B) — {len(AGENTIC_RESULT.edges)} internal nodes",
        "stats": stats_line(
            len(AGENTIC_RESULT.edges.get(AGENTIC_RESULT.root, [])),
            max(_agentic_depths.values()) if _agentic_depths else 0,
            _agentic_homed_chunks, TOTAL_FOOD_CHUNKS,
            sum(1 for _c in AGENTIC_RESULT.counts.values() if _c == 0), 0,
        ),
        "tree": agentic_tree,
    })'''
    )
)

# ----------------------------------------------------------------- grouping + scorecard
cells.append(
    md(
        """## Grouping method (from `main`) + cross-method scorecard

The merged bottom-up + LLM-grouping method (`build_grouped_shelves`, on `main`)
joins as one more entry, then every method is scored on the same metrics. The
scorecard is the headline: faithfulness vs navigability, on numbers."""
    )
)

cells.append(
    code(
        '''# The merged bottom-up + LLM grouping method (on main) as a scorecard entry.
from foodscholar.config import BottomUpGroupingConfig
from foodscholar.layer_a.grouping import build_grouped_shelves

_grouping_shelves = build_grouped_shelves(
    iter(chunks), api, BottomUpGroupingConfig(enabled=True),
    facet="foods", min_link_confidence=0.0, llm=fs.llm,
)
GROUPING_RESULT = from_shelves("grouping (main)", _grouping_shelves,
                               ontology=api, mentioned_leaves=MENTIONED)
RESULTS.append(GROUPING_RESULT)
print(f"grouping column: {len(_grouping_shelves)} shelves")'''
    )
)

cells.append(
    code(
        '''# ---- Cross-method scorecard: every method on the same metrics --------------
from IPython.display import HTML
from foodscholar.layer_a.bakeoff.metrics import sample_query_leaves
from foodscholar.layer_a.bakeoff.scorecard import build_scorecard, render_scorecard_html

QUERY_LEAVES = sample_query_leaves(dict(TERM_DOC_FREQ), n=100)
SCORECARD = build_scorecard(
    RESULTS, mentioned_leaves=MENTIONED, query_leaves=QUERY_LEAVES, k=3,
    llm=(fs.llm if HAVE_GROQ else None), nameability_sample=25,
)
print("methods scored:", [row["method"] for row in SCORECARD])
display(HTML(render_scorecard_html(SCORECARD)))'''
    )
)

# ----------------------------------------------------------------- Layer B active-shelf probe
cells.append(
    md(
        """## B-probe — full FoodOn evidence → active-shelf communities

This is **not** a production Layer-B change and it is **not** another Layer-A
tree method. It tests the alternative direction: let Layer A associate chunks
with FoodOn as faithfully as possible, then let Layer B choose the corpus-active
FoodOn nodes worth opening and run community detection inside those scopes."""
    )
)

cells.append(
    code(
        r'''# ---- Full FoodOn evidence layer ---------------------------------------------
# Unlike CHUNK_TERMS above, this keeps every FOODON id routed to the foods facet.
# That is the point of this probe: Layer A is FoodOn evidence, not a pruned tree.
def chunk_full_foodon_terms(c):
    fids = set()
    for fid in getattr(c, "foodon_ids", []) or []:
        if fid in api:
            fids.add(fid)
    for ln in getattr(c, "entity_links", []) or []:
        if ln.ontology_id in api and route_link_to_facet(ln) == "foods":
            fids.add(ln.ontology_id)
    return fids


FULL_CHUNK_TERMS = {c.chunk_id: chunk_full_foodon_terms(c) for c in chunks}
FULL_CHUNK_TERMS = {cid: terms for cid, terms in FULL_CHUNK_TERMS.items() if terms}
FULL_TERM_DOC_FREQ = Counter()
for terms in FULL_CHUNK_TERMS.values():
    for fid in terms:
        FULL_TERM_DOC_FREQ[fid] += 1

FULL_NODE_CHUNKS = defaultdict(set)
FULL_NODE_LEAVES = defaultdict(set)
for cid, terms in FULL_CHUNK_TERMS.items():
    for fid in terms:
        rollups = [fid] + [a for a in api.id_to_ancestors(fid) if a in api]
        for node in rollups:
            FULL_NODE_CHUNKS[node].add(cid)
            FULL_NODE_LEAVES[node].add(fid)

print(
    f"full FoodOn evidence: {len(FULL_CHUNK_TERMS)} chunks · "
    f"{len(FULL_TERM_DOC_FREQ)} direct FoodOn ids · "
    f"{len(FULL_NODE_CHUNKS)} rollup nodes"
)'''
    )
)

cells.append(
    code(
        r'''# ---- Active FoodOn shelf selection -------------------------------------------
# Goal: choose Layer-B scopes, not a browse taxonomy. A good active shelf is
# corpus-supported, reasonably specific, non-generic, and not near-duplicate
# with an already selected scope.
ACTIVE_MIN_SUPPORT = 50
ACTIVE_MAX_SUPPORT_FRAC = 0.25
ACTIVE_MIN_LEAVES = 2
ACTIVE_MAX_SCOPES = 36
ACTIVE_REDUNDANCY_JACCARD = 0.72

GENERIC_LABEL_BITS = (
    "entity", "object", "material entity", "physical object", "datum",
    "organism", "species", "taxonomic", "food product type",
    "material", " by ", "consumer group", "characteristic", "process",
    "meal type", "producing plant",
)
GENERIC_LABELS = {
    "food product", "plant food product", "animal food product",
    "edible food", "food material", "food source",
    "animal", "plant", "processed food", "food (cooked)",
}


def _chunks(fid):
    return FULL_NODE_CHUNKS.get(fid, set())


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def label_quality(fid):
    label = (api.id_to_label(fid) or fid).strip()
    low = label.lower()
    if low in GENERIC_LABELS:
        return 0.0
    if any(bit in low for bit in GENERIC_LABEL_BITS):
        return 0.0
    if len(label) > 70:
        return 0.35
    if low[:5].isdigit() or "efsa foodex2" in low or "gs1 gpc" in low:
        return 0.25
    return 0.75 if label.endswith(" food product") else 1.0


def parent_overlap(fid):
    own = _chunks(fid)
    parents = [p for p in api.id_to_parents(fid) if p in FULL_NODE_CHUNKS]
    return max((jaccard(own, _chunks(p)) for p in parents), default=0.0)


def active_score(fid):
    support = len(_chunks(fid))
    leaf_count = len(FULL_NODE_LEAVES.get(fid, set()))
    depth = len(api.id_to_ancestors(fid))
    direct = FULL_TERM_DOC_FREQ.get(fid, 0)
    nonredundant = max(0.05, 1.0 - parent_overlap(fid))
    # High support matters, but specificity/non-redundancy prevents broad hubs
    # from winning purely because nearly everything rolls up into them.
    return (
        (support ** 0.5)
        * (1.0 + min(leaf_count, 50) / 20.0)
        * (1.0 + min(depth, 10) / 12.0)
        * (1.0 + min(direct, 50) / 100.0)
        * label_quality(fid)
        * nonredundant
    )


max_support = ACTIVE_MAX_SUPPORT_FRAC * max(1, len(FULL_CHUNK_TERMS))
active_candidates = []
for fid, chunk_ids in FULL_NODE_CHUNKS.items():
    support = len(chunk_ids)
    if support < ACTIVE_MIN_SUPPORT or support > max_support:
        continue
    if len(FULL_NODE_LEAVES.get(fid, set())) < ACTIVE_MIN_LEAVES and FULL_TERM_DOC_FREQ.get(fid, 0) < ACTIVE_MIN_SUPPORT:
        continue
    if label_quality(fid) <= 0:
        continue
    active_candidates.append(fid)

selected_active_shelves = []
for fid in sorted(active_candidates, key=active_score, reverse=True):
    own = _chunks(fid)
    if any(jaccard(own, _chunks(sel)) >= ACTIVE_REDUNDANCY_JACCARD for sel in selected_active_shelves):
        continue
    selected_active_shelves.append(fid)
    if len(selected_active_shelves) >= ACTIVE_MAX_SCOPES:
        break

active_rows = []
for fid in selected_active_shelves:
    active_rows.append({
        "id": fid,
        "label": api.id_to_label(fid) or fid,
        "support": len(_chunks(fid)),
        "direct": FULL_TERM_DOC_FREQ.get(fid, 0),
        "leaves": len(FULL_NODE_LEAVES.get(fid, set())),
        "parent_overlap": parent_overlap(fid),
        "score": active_score(fid),
    })

print(f"active shelf candidates: {len(active_candidates)} → selected {len(active_rows)}")
for r in active_rows[:20]:
    print(
        f"{r['label']:<45} support={r['support']:>4} "
        f"direct={r['direct']:>3} leaves={r['leaves']:>3} "
        f"parentJ={r['parent_overlap']:.2f}"
    )'''
    )
)

cells.append(
    code(
        r'''# ---- B-probe Layer-A evidence tree -------------------------------------------
# This is the Layer-A shape for the direction we are testing:
# keep FoodOn as-is, associate corpus chunks to FoodOn ids, and roll support up
# through the FoodOn hierarchy. Unlike the Layer-B active shelf report, this is
# not restricted to selected active shelves.
B_FOODON_ROOT = "__b_probe_full_foodon_root__"
B_FOODON_MAX_DEPTH = 5
B_FOODON_MAX_CHILDREN = 18


def full_foodon_rank(fid):
    return (
        len(FULL_NODE_CHUNKS.get(fid, set())),
        len(FULL_NODE_LEAVES.get(fid, set())),
        FULL_TERM_DOC_FREQ.get(fid, 0),
        api.id_to_label(fid) or fid,
    )


def supported_parent(fid, supported_nodes):
    parents = [p for p in api.id_to_parents(fid) if p in supported_nodes]
    if not parents:
        return B_FOODON_ROOT
    # If FoodOn gives multiple supported parents, choose the closest/specific one
    # for this tree view. The ontology evidence itself remains multi-parent.
    return max(parents, key=lambda p: len(api.id_to_ancestors(p)))


def build_full_foodon_support_tree():
    supported_nodes = set(FULL_NODE_CHUNKS)
    children = defaultdict(list)
    counts = {B_FOODON_ROOT: len(FULL_CHUNK_TERMS)}
    labels = {B_FOODON_ROOT: "FoodOn as-is support tree"}

    for fid in sorted(supported_nodes, key=full_foodon_rank, reverse=True):
        parent = supported_parent(fid, supported_nodes)
        children[parent].append(fid)
        counts[fid] = len(FULL_NODE_CHUNKS.get(fid, set()))
        labels[fid] = api.id_to_label(fid) or fid

    # Keep sibling labels unique in the display. FoodOn can expose equivalent
    # labels through different ids/parents, which is valid but noisy here.
    deduped = defaultdict(list)
    for parent, kids in children.items():
        best_by_label = {}
        for child in kids:
            key = labels.get(child, child).strip().lower()
            prev = best_by_label.get(key)
            if prev is None or full_foodon_rank(child) > full_foodon_rank(prev):
                best_by_label[key] = child
        deduped[parent] = sorted(best_by_label.values(), key=full_foodon_rank, reverse=True)
    return deduped, counts, labels


B_FOODON_CHILDREN, B_FOODON_COUNTS, B_FOODON_LABELS = build_full_foodon_support_tree()
B_FOODON_HTML = render_tree_from_edges(
    B_FOODON_ROOT,
    B_FOODON_CHILDREN,
    B_FOODON_COUNTS,
    lambda n: B_FOODON_LABELS.get(n, n),
    max_depth=B_FOODON_MAX_DEPTH,
    open_depth=1,
    max_children=B_FOODON_MAX_CHILDREN,
)
B_FOODON_FANOUT = max((len(v) for v in B_FOODON_CHILDREN.values()), default=0)
COLUMNS.append({
    "title": "B-probe — Full FoodOn + corpus support",
    "stats": (
        f"full FoodOn evidence <b>{len(FULL_CHUNK_TERMS):,}</b> chunks · "
        f"<b>{len(FULL_TERM_DOC_FREQ):,}</b> direct ids · "
        f"<b>{len(FULL_NODE_CHUNKS):,}</b> rollup nodes<br>"
        f"FoodOn as-is support tree · top fan-out <b>{B_FOODON_FANOUT}</b> · "
        f"render cap <b>{B_FOODON_MAX_CHILDREN}</b>/parent"
    ),
    "tree": B_FOODON_HTML,
})
print(
    f"B-probe full FoodOn support tree: {len(FULL_NODE_CHUNKS)} rollup nodes, "
    f"top fan-out {B_FOODON_FANOUT}"
)'''
    )
)

cells.append(
    code(
        r'''# ---- Community detection inside active shelves ------------------------------
# First pass uses the existing Layer-B relatedness graph + Leiden, scoped to the
# selected FoodOn node's rolled-up chunks. This keeps the probe cheap and
# faithful to the current Layer-B code path.
from foodscholar.layer_b.community import run_leiden
from foodscholar.layer_b.relatedness_graph import build_relatedness_graph

COMMUNITY_MAX_SCOPES = 18
COMMUNITY_MAX_CHUNKS_PER_SCOPE = 350
COMMUNITY_MIN_CHUNKS = 20
COMMUNITY_TOP_TERMS = 4


def chunk_label_terms(chunk_ids, *, within_scope=None):
    counts = Counter()
    scope_ids = {within_scope, *api.id_to_descendants(within_scope)} if within_scope else None
    for cid in chunk_ids:
        for fid in FULL_CHUNK_TERMS.get(cid, ()):
            if scope_ids is None or fid in scope_ids:
                counts[fid] += 1
    return counts


def theme_label(chunk_ids, scope_id):
    top = []
    for fid, _ in chunk_label_terms(chunk_ids, within_scope=scope_id).most_common(12):
        label = api.id_to_label(fid) or fid
        if label not in top:
            top.append(label)
        if len(top) >= COMMUNITY_TOP_TERMS:
            break
    return ", ".join(top) or (api.id_to_label(scope_id) or scope_id)


active_theme_rows = []
community_errors = []
for shelf in active_rows[:COMMUNITY_MAX_SCOPES]:
    scope_id = shelf["id"]
    scope_chunk_ids = sorted(_chunks(scope_id))
    if len(scope_chunk_ids) < COMMUNITY_MIN_CHUNKS:
        continue
    sampled_ids = scope_chunk_ids[:COMMUNITY_MAX_CHUNKS_PER_SCOPE]
    scope_chunks = fs.chunk_store.get_many(sampled_ids)
    graph = None
    try:
        graph = build_relatedness_graph(scope_chunks, fs.config.layer_b.relatedness)
        comms = run_leiden(graph, fs.config.layer_b.leiden)
    except Exception as exc:
        community_errors.append((scope_id, str(exc)))
        comms = []

    themes = []
    for comm in sorted(comms, key=len, reverse=True)[:8]:
        ids = {graph.vs[idx]["chunk_id"] for idx in comm}
        themes.append({
            "label": theme_label(ids, scope_id),
            "n_chunks": len(ids),
            "top_terms": [
                (api.id_to_label(fid) or fid, n)
                for fid, n in chunk_label_terms(ids, within_scope=scope_id).most_common(COMMUNITY_TOP_TERMS)
            ],
        })
    active_theme_rows.append({
        **shelf,
        "sampled_chunks": len(sampled_ids),
        "graph_edges": graph.ecount() if graph is not None else 0,
        "themes": themes,
    })

if community_errors:
    print("community errors:", community_errors[:5])
print(f"community-scoped shelves: {len(active_theme_rows)}")
for row in active_theme_rows[:12]:
    print(
        f"{row['label']:<42} chunks={row['support']:>4} "
        f"sampled={row['sampled_chunks']:>3} themes={len(row['themes'])}"
    )
    for th in row["themes"][:3]:
        print(f"  - {th['label']} ({th['n_chunks']} chunks)")'''
    )
)

cells.append(
    code(
        r'''# ---- Render active-shelf community probe ------------------------------------
from IPython.display import HTML
from jinja2 import Template

ACTIVE_REPORT = Template(
    """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Layer B active FoodOn shelf communities</title><style>
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1200px;margin:1.5rem auto;padding:0 1rem;}
h1{border-bottom:3px solid #5b8a72;padding-bottom:.3rem;}
.meta{color:#777;font-size:.9rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:.8rem}
.scope{border:1px solid #d8dfd9;border-radius:6px;padding:.65rem .8rem;background:#fff}
.scope h2{font-size:1rem;margin:.1rem 0 .25rem;color:#2f513f}.stat{font-size:.82rem;color:#69756c;margin-bottom:.45rem}
ul{margin:.25rem 0 .3rem 1.1rem;padding:0}.theme{margin:.18rem 0}.small{font-size:.8rem;color:#7a8580}
table{border-collapse:collapse;width:100%;margin:.8rem 0}td,th{border:1px solid #dde3df;padding:.25rem .35rem;text-align:left;font-size:.82rem}
</style></head><body>
<h1>Layer B active FoodOn shelf communities</h1>
<p class="meta">
Layer A evidence = all FOODON ids routed to foods, rolled up through FoodOn.
Active shelves are selected by support, specificity, label quality, and
anti-redundancy; communities use the existing Layer-B relatedness graph + Leiden.
</p>
<table><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>
<tr><td>chunks with FoodOn foods evidence</td><td>{{ n_chunks }}</td></tr>
<tr><td>direct FoodOn ids</td><td>{{ n_terms }}</td></tr>
<tr><td>rollup nodes</td><td>{{ n_nodes }}</td></tr>
<tr><td>selected active shelves</td><td>{{ n_selected }}</td></tr>
<tr><td>community-scoped shelves rendered</td><td>{{ n_rendered }}</td></tr>
</tbody></table>
<div class="grid">
{% for row in rows %}
<section class="scope">
  <h2>{{ row.label }}</h2>
  <div class="stat">
    {{ row.support }} chunks · {{ row.leaves }} direct/descendant ids ·
    direct {{ row.direct }} · parent J {{ "%.2f"|format(row.parent_overlap) }} ·
    sampled {{ row.sampled_chunks }}
  </div>
  {% if row.themes %}
  <ul>
  {% for th in row.themes %}
    <li class="theme"><b>{{ th.label }}</b> <span class="small">({{ th.n_chunks }} chunks)</span></li>
  {% endfor %}
  </ul>
  {% else %}
    <div class="small">No Leiden communities above the current minimum size.</div>
  {% endif %}
</section>
{% endfor %}
</div>
</body></html>"""
)

active_out = VIZ_DIR / "layer_b_active_shelf_communities.html"
active_out.write_text(
    ACTIVE_REPORT.render(
        n_chunks=len(FULL_CHUNK_TERMS),
        n_terms=len(FULL_TERM_DOC_FREQ),
        n_nodes=len(FULL_NODE_CHUNKS),
        n_selected=len(active_rows),
        n_rendered=len(active_theme_rows),
        rows=active_theme_rows,
    ),
    encoding="utf-8",
)
display(HTML(ACTIVE_REPORT.render(
    n_chunks=len(FULL_CHUNK_TERMS),
    n_terms=len(FULL_TERM_DOC_FREQ),
    n_nodes=len(FULL_NODE_CHUNKS),
    n_selected=len(active_rows),
    n_rendered=len(active_theme_rows),
    rows=active_theme_rows,
)))
print(f"wrote {active_out} ({active_out.stat().st_size/1024:.0f} KB)")'''
    )
)

# ----------------------------------------------------------------- assembly
cells.append(md("## Assemble side-by-side report"))

cells.append(
    code(
        r'''from jinja2 import Template

REPORT = Template(
    """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Layer-A method bake-off — foods</title><style>
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
<h1>Layer-A method bake-off — foods facet</h1>
<p class="meta">{{ total }} chunks with food terms · model {{ model }} ·
all categories are real FoodOn ids · judge by eye{% if not have_groq %} · <i>1b skipped (no GROQ_API_KEY)</i>{% endif %}</p>
<div class="grid">
{% for c in columns %}
  <div class="col"><h2>{{ c.title }}</h2><div class="stats">{{ c.stats }}</div>{{ c.tree }}</div>
{% endfor %}
</div></body></html>"""
)

out = VIZ_DIR / "layer_a_method_bakeoff_foods.html"
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
