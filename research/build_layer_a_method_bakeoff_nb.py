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

import sys
from pathlib import Path as _Path

# This script was archived under research/. Put research/ on the path so the
# relocated bake-off package imports as `bakeoff` (was foodscholar.layer_a.bakeoff).
sys.path.insert(0, str(_Path(__file__).resolve().parent))

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
                           max_depth=4, open_depth=1, max_children=18,
                           direct_map=None):
    """Generic nested <details> renderer over an explicit children_map.

    `count_map[nid]` is the TOTAL distinct chunks under nid (direct + via
    descendants). When `direct_map` is given, the badge breaks it down as
    `Chunks: total (D: direct | L: lifted)` where lifted = total - direct."""
    def node(nid, rel):
        kids = sorted(children_map.get(nid, []), key=lambda c: -count_map.get(c, 0))
        nch = len(kids)
        cnt = count_map.get(nid, 0)
        if direct_map is not None:
            d = direct_map.get(nid, 0)
            lifted = max(cnt - d, 0)
            badge = f"<span class='c'>{nch} sub · Chunks: {cnt:,} (D: {d:,} | L: {lifted:,})</span>"
        else:
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
from bakeoff.result import from_children_map, from_shelves

COLUMNS = []  # each: {"title","stats","tree"}
RESULTS = []  # each: a MethodResult, for the cross-method scorecard
MENTIONED = set(TERM_DOC_FREQ)  # corpus-mentioned food leaves (shared denominator)
TOTAL_FOOD_CHUNKS = len(CHUNK_TERMS)
print("helpers ready")'''
    )
)

# ----------------------------------------------------------------- Col 0 baseline

# ----------------------------------------------------------------- Col 2 structural cut

# ----------------------------------------------------------------- Col 1a/1b backbone

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
NODE_CHUNKS = defaultdict(set)   # node -> distinct chunks under it (direct OR via descendants)
DIRECT_CHUNKS = defaultdict(set) # node -> distinct chunks that mention it EXACTLY (direct)
TERM_ROLLUPS = {}
for fid in TERM_DOC_FREQ:
    rollups = [fid] + [
        a for a in api.id_to_ancestors(fid)
        if a == FOOD_PRODUCT or api.is_subclass_of(a, FOOD_PRODUCT)
    ]
    TERM_ROLLUPS[fid] = rollups

for cid, terms in CHUNK_TERMS.items():
    for fid in terms:
        DIRECT_CHUNKS[fid].add(cid)
        for nd in TERM_ROLLUPS.get(fid, ()):
            NODE_CHUNKS[nd].add(cid)


def dl_maps(node_ids):
    """(total_map, direct_map) over `node_ids`, both distinct-chunk counts.
    total = chunks under node (direct + descendants); direct = chunks mentioning
    the exact FoodOn id. No node double-counts (set semantics)."""
    total = {n: len(NODE_CHUNKS.get(n, ())) for n in node_ids}
    direct = {n: len(DIRECT_CHUNKS.get(n, ())) for n in node_ids}
    return total, direct

EXPAND_MIN_CHUNKS = 25
EXPAND_MAX_DEPTH = 6       # root -> backbone -> ... -> concrete high-support terms
EXPAND_MAX_CHILDREN = 12   # hard cap per opened parent

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

def resolve_filing_tier(fid):
    """Descend through single-child filing tiers that carry NO direct chunks of
    their own, returning the first meaningful node.

    A node with 0 direct chunks and exactly one supported child is a pure filing
    tier (e.g. `maize kernel` over `corn kernel`); we show its descendant instead.
    Faithful — the kept node is still a real is-a descendant. Chains of any length
    collapse, and a node that branches (>=2 children) or has direct chunks stops
    the descent (it's a real grouping / leaf)."""
    seen = set()
    while (
        fid not in seen
        and TERM_DOC_FREQ.get(fid, 0) == 0      # no direct chunks of its own
        and len(supported_children(fid)) == 1   # a single-child filing tier
    ):
        seen.add(fid)
        fid = supported_children(fid)[0]
    return fid


def collapsed_supported_children(fid):
    """Display children of `fid`, with each child resolved through filing tiers so
    e.g. `Corn -> maize kernel(0) -> corn kernel(83)` shows `corn kernel` directly
    under `Corn`. Deduped (two tiers can resolve to the same node), ranked, capped."""
    out, seen = [], set()
    for child in supported_children(fid):
        resolved = resolve_filing_tier(child)
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return sorted(out, key=display_rank, reverse=True)[:EXPAND_MAX_CHILDREN]


def prune_empty_leaves(children, root):
    """Remove display-leaf nodes that have NO displayed descendants and NO direct
    chunks (pure filing dead-ends). Cascades: a parent left empty + zero-direct is
    pruned in turn. `root` is never pruned. Mutates and returns `children`."""
    changed = True
    while changed:
        changed = False
        for parent in list(children):
            kept = []
            for c in children[parent]:
                is_leaf = not children.get(c)
                if is_leaf and c != root and len(DIRECT_CHUNKS.get(c, ())) == 0:
                    changed = True  # prune: empty filing dead-end
                else:
                    kept.append(c)
            if kept:
                children[parent] = kept
            else:
                del children[parent]  # parent now childless; drop its edge list
    return children

def controlled_backbone_column(title, backbone_ids):
    backbone_ids = [b for b in backbone_ids if b in api]
    homed, multi_home, unhomed = lift_to_backbone(backbone_ids)

    ROOT = "__controlled_backbone_root__"
    children = defaultdict(list)
    counts = {ROOT: sum(len(v) for v in homed.values())}
    labels = {ROOT: "foods"}
    # FoodOn is a multi-parent DAG (e.g. broccoli is-a 3 categories). Project to a
    # single-parent TREE: each node is placed under exactly ONE parent — the first
    # to reach it in this support-sorted DFS — so a node (and its chunks) never
    # appears twice. Matches the structural-cut column + production prune.
    placed: set[str] = set()

    def expand(parent, rel_depth):
        if rel_depth >= EXPAND_MAX_DEPTH:
            return
        for child in collapsed_supported_children(parent):
            if child in placed:
                continue
            placed.add(child)
            children[parent].append(child)
            counts[child] = node_support(child)
            labels[child] = api.id_to_label(child) or child
            expand(child, rel_depth + 1)

    for b in sorted(backbone_ids, key=display_rank, reverse=True):
        if b in placed:
            continue
        placed.add(b)
        children[ROOT].append(b)
        counts[b] = len(homed.get(b, set()))
        labels[b] = api.id_to_label(b) or b
        expand(b, 1)

    # Faithful view cleanup: drop empty filing dead-ends (no descendants, no direct
    # chunks). Collapse of single-child tiers already happened in expand().
    prune_empty_leaves(children, ROOT)

    tree_node_ids = {ROOT, *children, *(c for kids in children.values() for c in kids)}
    total_map, direct_map = dl_maps(tree_node_ids)
    total_map[ROOT] = TOTAL_FOOD_CHUNKS  # synthetic root carries all food chunks
    tree = render_tree_from_edges(
        ROOT,
        children,
        total_map,
        lambda n: labels.get(n, n),
        max_depth=EXPAND_MAX_DEPTH,
        open_depth=2,
        max_children=EXPAND_MAX_CHILDREN,
        direct_map=direct_map,
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
    result = from_children_map(
        title.split(" —")[0].strip(), root=ROOT, children_map=children,
        counts=counts, labels=labels, ontology=api, mentioned_leaves=MENTIONED,
    )
    RESULTS.append(result)
    # Return the frozen MethodResult too — the agentic alias pass reuses this exact
    # backbone (structure + is-a homing) and only adds aliases on top of it.
    return multi_home, result

# auto backbone = the direct children of food product (real FoodOn cats).
auto_backbone = api.id_to_children(FOOD_PRODUCT)
MULTI_HOME_1A_PLUS, BASE_1APLUS_RESULT = controlled_backbone_column(
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

# ----------------------------------------------------------------- agentic (Plan B)
cells.append(
    md(
        """## Agentic aliasing pass (Plan B) — GROQ-gated

The agent does **one** thing: add layperson **aliases** to node labels. It does
**not** edit structure and does **not** reparent anything — not nodes, not
chunks. It takes the **frozen 1a+ backbone** and, through its read-only lens
(support, supported-children, and FoodOn object-property relation bridges —
`has defining ingredient`, `has ingredient`, `derives from`, `member of`, …),
proposes a short everyday name for any jargon node. The relations *inform the
wording*, never the placement. Edges, ids, original labels, and chunk homing are
copied through unchanged — so this column scores **identically to 1a+** on
coverage / faithfulness / specificity / findability and differs **only on
nameability**. Skipped without `GROQ_API_KEY`."""
    )
)

cells.append(
    code(
        '''# Agentic ALIASING pass (Plan B) — aliases ONLY. Takes the frozen 1a+ backbone
# (BASE_1APLUS_RESULT) and adds layperson aliases; structure + chunk homing are copied
# verbatim. No node or chunk is reparented, so only nameability can differ from 1a+.
AGENTIC_RESULT = None
if not HAVE_GROQ:
    print("GROQ_API_KEY not set — skipping agentic aliasing pass.")
else:
    from bakeoff.agentic.alias import build_aliased_result
    from bakeoff.agentic.relations import load_relation_index
    from bakeoff.agentic.tools import GraphTools

    repo_root = HERE if (HERE / "data" / "foodon.owl").exists() else HERE.parent
    rel_index = load_relation_index(str(repo_root / "data" / "foodon.owl"))
    print(f"relation index: {len(rel_index)} FOODON terms with FoodOn object-property relations")

    # Lens only — relations + rolled-up support inform the alias wording.
    _support = {n: len(cs) for n, cs in NODE_CHUNKS.items()}
    _tools = GraphTools(api, rel_index, node_support=_support, min_support=EXPAND_MIN_CHUNKS)

    AGENTIC_RESULT = build_aliased_result(BASE_1APLUS_RESULT, tools=_tools, llm=fs.llm)
    RESULTS.append(AGENTIC_RESULT)
    print(f"agentic aliasing: {len(AGENTIC_RESULT.aliases)} aliases over "
          f"{AGENTIC_RESULT.llm_calls} nodes (structure + homing identical to 1a+)")

    # Same backbone as 1a+, so same D/L counts — only labels change (alias if present).
    _nodes = {AGENTIC_RESULT.root, *AGENTIC_RESULT.edges,
              *(c for kids in AGENTIC_RESULT.edges.values() for c in kids)}
    _total, _direct = dl_maps(_nodes)
    _total[AGENTIC_RESULT.root] = TOTAL_FOOD_CHUNKS
    agentic_tree = render_tree_from_edges(
        AGENTIC_RESULT.root, AGENTIC_RESULT.edges, _total, AGENTIC_RESULT.display,
        max_depth=EXPAND_MAX_DEPTH, open_depth=2, max_children=EXPAND_MAX_CHILDREN,
        direct_map=_direct,
    )
    _fanout = max((len(v) for v in AGENTIC_RESULT.edges.values()), default=0)
    _depth = max((AGENTIC_RESULT.home_distance.values()), default=0)
    COLUMNS.append({
        "title": f"agentic (aliasing) — {len(AGENTIC_RESULT.aliases)} aliases",
        "stats": (
            stats_line(_fanout, EXPAND_MAX_DEPTH,
                       TOTAL_FOOD_CHUNKS - (TOTAL_FOOD_CHUNKS - len(AGENTIC_RESULT.leaf_home)),
                       TOTAL_FOOD_CHUNKS, 0, 0)
            + f"<br>{len(AGENTIC_RESULT.aliases)} aliases added · structure = 1a+"
        ),
        "tree": agentic_tree,
    })'''
    )
)

# ----------------------------------------------------------------- grouping + scorecard

cells.append(
    code(
        '''# ---- Cross-method scorecard: every method on the same metrics --------------
from IPython.display import HTML
from bakeoff.metrics import sample_query_leaves
from bakeoff.scorecard import build_scorecard, render_scorecard_html

QUERY_LEAVES = sample_query_leaves(dict(TERM_DOC_FREQ), n=100)
SCORECARD = build_scorecard(
    RESULTS, mentioned_leaves=MENTIONED, query_leaves=QUERY_LEAVES, k=3,
    llm=(fs.llm if HAVE_GROQ else None), nameability_sample=25,
)
print("methods scored:", [row["method"] for row in SCORECARD])
display(HTML(render_scorecard_html(SCORECARD)))'''
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
all categories are real FoodOn ids · judge by eye{% if not have_groq %} · <i>agentic aliasing skipped (no GROQ_API_KEY)</i>{% endif %}</p>
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
