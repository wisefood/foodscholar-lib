"""Assemble notebooks/retier_layer_a.ipynb from source cells.

Prototype for Layer-A category re-tiering. Kept as a script (not hand-edited
JSON) so the notebook source stays reviewable and regenerable. Run with the
foodscholar env:

    /mnt/miniconda3/envs/foodscholar/bin/python scripts/build_retier_layer_a_nb.py

Spec: docs/superpowers/specs/2026-05-28-layer-a-category-retiering-design.md
"""

from __future__ import annotations

import nbformat as nbf

NB_PATH = "notebooks/retier_layer_a.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells: list = []

# ----------------------------------------------------------------- title
cells.append(
    md(
        """# Layer-A category re-tiering — prototype

The live **foods** facet is a flat, un-navigable blob: `food product` alone
carries ~111 direct children (plus ~81 on the synthetic root) — a user lands on
~186 sibling categories with no intuitive grouping. This notebook prototypes a
fix that **re-cuts FoodOn's own hierarchy** for browsability, using *only* real
FoodOn ids and is-a edges.

Three coupled parts:

| Part | What | Why |
|------|------|-----|
| **A** | Reconcile pruning (less aggressive) | keep FoodOn's mid-level grouping nodes that current thresholds dissolve |
| **B** | LLM re-tiering | have `llama-3.1-8b-instant` pick *real FoodOn intermediate ancestors* as grouping tiers and assign children under them |
| **C** | Before/after tree | visualize `vegetable → green/root/steamed…` vs today's flat fan-out |

**Constraint:** stay 100% within FoodOn. The LLM only ever *selects among FoodOn
ids we hand it* — it cannot invent or relabel categories.

> Run on the **`foodscholar`** kernel. Builds the foods facet **in-process** from
> `data/annotated.parquet` with in-memory stores — no Neo4j/Elastic needed.
> Part B needs `GROQ_API_KEY`; if unset, A+C still run and B degrades gracefully."""
    )
)

# ----------------------------------------------------------------- foundation header
cells.append(
    md(
        """## §0 — Build the foods facet in-process

`load_chunks(annotated.parquet)` rehydrates the in-memory chunk store (the
parquet already carries `entity_links`/`foodon_ids`), then `build_layer_a()`
projects the foods facet. No NER/embedding re-run, no external services."""
    )
)

cells.append(
    code(
        '''import html as _html
import os
from collections import Counter, defaultdict
from pathlib import Path

from foodscholar.config import FoodScholarConfig
from foodscholar.facade import FoodScholar

HERE = Path.cwd()
ROOT = HERE if (HERE / "data" / "foodon.owl").exists() else HERE.parent
VIZ_DIR = ROOT / "data" / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Knobs for this prototype run. Part A sweeps these; this is the baseline.
MIN_SUPPORT = 25
MAX_DEPTH = 6
FANOUT_TARGET = 15  # levels wider than this are re-tiering candidates (Part A/C tune)

# Only wire Groq when the key is present — the client is built eagerly in
# from_config and raises if GROQ_API_KEY is unset. Without it the facade falls
# back to a mock LLM, so Parts A & C still run and Part B degrades cleanly.
HAVE_GROQ = bool(os.environ.get("GROQ_API_KEY"))
_cfg_dict = {
    "corpus": {
        "chunks_path": str(ROOT / "tests/fixtures/sample_chunks.jsonl"),
        "annotated_snapshot_path": str(ROOT / "data/annotated.parquet"),
    },
    "ontology": {
        "foodon_path": str(ROOT / "data/foodon.owl"),
        "cache_path": str(ROOT / "data/foodon_cache.parquet"),
        "prefix_filter": ["FOODON:"],
    },
    "layer_a": {
        "min_support": MIN_SUPPORT,
        "max_depth": MAX_DEPTH,
        "facets": ["foods"],
        "blacklist_terms": ["material entity", "physical object", "manufactured product"],
    },
    "storage": {
        "chunk_store": {"backend": "memory"},
        "graph_store": {"backend": "memory"},
    },
}
if HAVE_GROQ:
    # llama-3.1-8b-instant for the Part B tier-picker (constrained index
    # selection, not free generation — 8b is plenty and cheap).
    _cfg_dict["llm"] = {"primary": {"provider": "groq", "model": "llama-3.1-8b-instant"}}
cfg = FoodScholarConfig.model_validate(_cfg_dict)

fs = FoodScholar.from_config(cfg)
api = fs.load_ontology()
fs.attach_ontology(api)
n_chunks = fs.load_chunks(str(ROOT / "data/annotated.parquet"))
fs.build_layer_a()
shelves = [s for s in fs.graph_store.list_shelves() if s.facet == "foods"]
by_id = {s.shelf_id: s for s in shelves}
print(f"{n_chunks} chunks · {len(shelves)} foods shelves · llm={fs.llm.model_id}")'''
    )
)

# ----------------------------------------------------------------- fan-out diagnostic
cells.append(
    code(
        '''def fanout(shelf_list):
    """parent_shelf_id -> list of child shelf_ids, over a shelf list."""
    kids = defaultdict(list)
    for s in shelf_list:
        if s.parent_shelf_id:
            kids[s.parent_shelf_id].append(s.shelf_id)
    return kids


def fanout_report(shelf_list, target=FANOUT_TARGET):
    kids = fanout(shelf_list)
    counts = sorted(((len(v), p) for p, v in kids.items()), reverse=True)
    depth_dist = dict(sorted(Counter(s.depth for s in shelf_list).items()))
    wide = [(c, p) for c, p in counts if c > target]
    print(f"shelves={len(shelf_list)} | depth dist={depth_dist}")
    print(f"max fan-out={counts[0][0] if counts else 0} | parents > {target} kids: {len(wide)}")
    bid = {s.shelf_id: s for s in shelf_list}
    for c, p in counts[:6]:
        lbl = bid[p].label if p in bid else p
        print(f"  {c:4d} children  ←  {lbl!r} (depth {bid[p].depth if p in bid else '?'})")
    return wide, kids


WIDE, KIDS = fanout_report(shelves)'''
    )
)

# ----------------------------------------------------------------- Part A header
cells.append(
    md(
        """## Part A — Reconcile pruning (less aggressive)

The fan-out exists because support pruning dissolves FoodOn's mid-level grouping
nodes; survivors then lift up to a shallow ancestor. Here we sweep `min_support`
and measure not just shelf *count* but **fan-out** and **how many candidate
grouping nodes survive** — looking for a setting that keeps the middle."""
    )
)

cells.append(
    code(
        '''def build_foods(min_support, max_depth=MAX_DEPTH, umbrella_direct_share_max=0.10):
    """Rebuild the foods facet at a given pruning setting; return shelf list."""
    c = cfg.model_copy(deep=True)
    c.layer_a.min_support = min_support
    c.layer_a.max_depth = max_depth
    c.layer_a.umbrella_direct_share_max = umbrella_direct_share_max
    f = FoodScholar.from_config(c)
    f.attach_ontology(api)
    f.load_chunks(str(ROOT / "data/annotated.parquet"))
    f.build_layer_a()
    return [s for s in f.graph_store.list_shelves() if s.facet == "foods"]


print(f"{'min_support':>12} {'shelves':>8} {'max_fanout':>11} {'wide_parents':>13} {'mid_nodes':>10}")
SWEEP = []
for ms in (5, 10, 15, 25, 40):
    sl = build_foods(ms)
    k = fanout(sl)
    counts = sorted((len(v) for v in k.values()), reverse=True) or [0]
    wide = sum(1 for v in k.values() if len(v) > FANOUT_TARGET)
    # mid-level grouping nodes = internal shelves (have children) not at depth 0/1
    mid = sum(1 for s in sl if k.get(s.shelf_id) and s.depth >= 2)
    SWEEP.append((ms, len(sl), counts[0], wide, mid))
    print(f"{ms:>12} {len(sl):>8} {counts[0]:>11} {wide:>13} {mid:>10}")
# Lower min_support keeps more mid-level nodes (more tier candidates) at the cost
# of more total shelves — Part B then groups, not prunes, the excess.'''
    )
)

# ----------------------------------------------------------------- Part B header
cells.append(
    md(
        """## Part B — LLM re-tiering (picks FoodOn intermediates)

For each over-wide parent: enumerate the **real FoodOn intermediate ancestors**
that sit between it and its many children, and ask `llama-3.1-8b-instant` to
choose which of *those existing nodes* are the intuitive grouping tiers and
which child goes under which. The model answers **by index** over candidates we
provide — it physically cannot return a non-FoodOn id.

Mirrors the conventions of
[`semantic_consolidation/judge.py`](../src/foodscholar/layer_a/semantic_consolidation/judge.py):
numbered blocks, `generate_json` with a schema, defensive index→id mapping."""
    )
)

cells.append(
    code(
        '''# Candidate grouping ancestors for a wide parent: FoodOn nodes that are
# (a) a real ancestor of >1 of the parent's children, and (b) a descendant of
# the parent itself (so they sit strictly between parent and children).
def tier_candidates(parent_shelf_id, child_shelf_ids, min_cover=2, max_cand=20):
    parent = by_id[parent_shelf_id]
    if not parent.foodon_id:
        # synthetic facet root — candidates are ancestors-of-children under nothing;
        # use any FoodOn ancestor shared by >=min_cover children.
        parent_fid = None
    else:
        parent_fid = parent.foodon_id
    child_fids = {by_id[c].foodon_id: c for c in child_shelf_ids if by_id[c].foodon_id}
    cover = Counter()
    for fid in child_fids:
        for anc in api.id_to_ancestors(fid):
            if anc == parent_fid:
                continue
            if parent_fid is not None and not api.is_subclass_of(anc, parent_fid):
                continue  # keep only ancestors below the parent
            if anc in child_fids:
                continue  # a child can't be its own tier
            cover[anc] += 1
    cands = [(c, fid) for fid, c in cover.items() if c >= min_cover]
    cands.sort(reverse=True)  # most-covering first
    return [fid for _, fid in cands[:max_cand]], child_fids


# Build the per-parent candidate sets for every wide parent.
RETIER_JOBS = []
for c, p in WIDE:
    cand_fids, child_fids = tier_candidates(p, KIDS[p])
    if cand_fids:
        RETIER_JOBS.append({"parent": p, "children": child_fids, "candidates": cand_fids})
    print(f"{by_id[p].label!r}: {c} children → {len(cand_fids)} FoodOn tier candidates")'''
    )
)

cells.append(
    code(
        r'''import json

RETIER_PROMPT_VERSION = "v0.1-retier"

RETIER_SCHEMA = {
    "type": "object",
    "properties": {
        "tiers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tier_index": {"type": "integer"},  # 1-based into candidates
                    "child_indices": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["tier_index", "child_indices"],
            },
        }
    },
    "required": ["tiers"],
}

RETIER_PROMPT = """You are organizing a flat list of FoodOn food categories into \
an intuitive browsing hierarchy. The parent category is {parent!r} and currently \
has {n_children} direct children — too many to browse.

You may ONLY group children under one of the CANDIDATE GROUPING CATEGORIES below \
(these are real FoodOn classes that sit between the parent and the children). Do \
NOT invent categories. Assign each child to the most intuitive grouping category, \
by index. Leave a child unassigned (omit it) if none of the candidates is a \
natural fit. Pick the smallest set of candidate tiers that makes the list \
browsable (aim for a handful of balanced groups).

CANDIDATE GROUPING CATEGORIES (choose tier_index from these, 1-based):
{candidate_blocks}

CHILDREN TO ASSIGN (child_indices are 1-based into this list):
{child_blocks}

Return JSON: a list of tiers, each {{"tier_index": <candidate#>, "child_indices": [<child#>...]}}.
"""


def _syn(fid, k=4):
    s = api.id_to_synonyms(fid)
    return (" · " + ", ".join(s[:k])) if s else ""


def build_retier_prompt(job):
    cand = job["candidates"]
    children = list(job["children"].keys())
    cand_blocks = "\n".join(
        f"  {i}. {api.id_to_label(fid)}{_syn(fid)}" for i, fid in enumerate(cand, 1)
    )
    child_blocks = "\n".join(
        f"  {i}. {api.id_to_label(fid)}{_syn(fid)}" for i, fid in enumerate(children, 1)
    )
    return RETIER_PROMPT.format(
        parent=api.id_to_label(by_id[job["parent"]].foodon_id) if by_id[job["parent"]].foodon_id else "foods",
        n_children=len(children),
        candidate_blocks=cand_blocks,
        child_blocks=child_blocks,
    )


print(f"{len(RETIER_JOBS)} re-tiering job(s) prepared; prompt {RETIER_PROMPT_VERSION}")
if RETIER_JOBS:
    print("\n--- sample prompt (truncated) ---")
    print(build_retier_prompt(RETIER_JOBS[0])[:1200])'''
    )
)

cells.append(
    code(
        '''# Run the tier-picker. Degrades gracefully if GROQ_API_KEY is absent.
def run_retier(job):
    cand, children = job["candidates"], list(job["children"].keys())
    budget = max(1024, 256 + len(children) * 40)
    obj = fs.llm.generate_json(build_retier_prompt(job), RETIER_SCHEMA, max_tokens=budget)
    # Defensive index->id mapping (mirror judge._parse_cluster): drop OOR indices.
    ops = []
    for tier in obj.get("tiers", []):
        ti = tier.get("tier_index", 0)
        if not (1 <= ti <= len(cand)):
            continue
        tier_fid = cand[ti - 1]
        assigned = []
        for ci in tier.get("child_indices", []):
            if 1 <= ci <= len(children):
                assigned.append(children[ci - 1])
        if assigned:
            ops.append({"tier_fid": tier_fid, "child_fids": assigned})
    return ops


RETIER_OPS = []
if not HAVE_GROQ:
    print("GROQ_API_KEY not set — skipping live Part B tier-picker.")
    print("Set it and re-run this cell to generate the grouping ops; Parts A & C run regardless.")
else:
    for job in RETIER_JOBS:
        ops = run_retier(job)
        RETIER_OPS.append({"parent": job["parent"], "ops": ops})
        for op in ops:
            print(f"  tier '{api.id_to_label(op['tier_fid'])}' ← {len(op['child_fids'])} children")
    print(f"\\n{sum(len(j['ops']) for j in RETIER_OPS)} grouping tier(s) proposed across {len(RETIER_OPS)} parent(s)")'''
    )
)

# ----------------------------------------------------------------- Part C header
cells.append(
    md(
        """## Part C — Before / after tree

Render the foods facet as a nested `<details>` tree (same renderer style as
`explore_foodon`): **before** (today's flat fan-out) and **after** applying the
Part B grouping ops. Written to `data/viz/retier_foods_report.html`."""
    )
)

cells.append(
    code(
        '''from foodscholar.layer_a.prune import shelf_id_for_foodon


def apply_ops(shelf_list, retier_ops):
    """Return a new shelf list with grouping tiers inserted + children reparented.

    Pure structural rewrite over Shelf.parent_shelf_id. Inserted tier shelves
    are real FoodOn nodes (may already exist as shelves; if not, synthesized
    from the FoodOn id). Chunk counts on a new tier = sum of its children
    (lifted; no direct support required — that's the point of re-tiering)."""
    bid = {s.shelf_id: s.model_copy(deep=True) for s in shelf_list}
    for entry in retier_ops:
        parent_sid = entry["parent"]
        for op in entry["ops"]:
            tier_sid = shelf_id_for_foodon(op["tier_fid"])
            if tier_sid not in bid:
                from foodscholar.io.graph import Shelf

                bid[tier_sid] = Shelf(
                    shelf_id=tier_sid,
                    label=api.id_to_label(op["tier_fid"]) or op["tier_fid"],
                    facet="foods",
                    depth=bid[parent_sid].depth + 1,
                    foodon_id=op["tier_fid"],
                    parent_shelf_id=parent_sid,
                    chunk_count=0,
                    support_direct=0,
                    support_lifted=0,
                    see_also=[],
                )
            for child_sid in op["child_fids"]:
                if child_sid in bid:
                    bid[child_sid] = bid[child_sid].model_copy(
                        update={"parent_shelf_id": tier_sid}
                    )
    return list(bid.values())


def render_tree(shelf_list, root_label="food product", max_depth=4, open_depth=1):
    bid = {s.shelf_id: s for s in shelf_list}
    kids = fanout(shelf_list)
    root_fid = api.name_to_id(root_label)
    root_sid = shelf_id_for_foodon(root_fid) if root_fid else None
    if root_sid not in bid:
        # fall back to the widest parent
        root_sid = max(kids, key=lambda p: len(kids.get(p, [])))

    def node(sid, rel):
        s = bid.get(sid)
        label = _html.escape(s.label if s else sid)
        children = sorted(kids.get(sid, []), key=lambda c: -len(kids.get(c, [])))
        nch = len(children)
        badge = f"<span class='c'>{nch} children · {s.chunk_count:,} chunks</span>" if s else ""
        if nch == 0 or rel >= max_depth:
            extra = f" <span class='m'>(+{nch} below)</span>" if nch else ""
            return f"<li>{label}{badge}{extra}</li>"
        shown = children[:14]
        hidden = nch - len(shown)
        inner = "".join(node(c, rel + 1) for c in shown)
        if hidden:
            inner += f"<li class='m'>+{hidden} more…</li>"
        op = " open" if rel < open_depth else ""
        return f"<li><details{op}><summary><b>{label}</b>{badge}</summary><ul>{inner}</ul></details></li>"

    return f"<ul class='ftree'>{node(root_sid, 0)}</ul>"


before_tree = render_tree(shelves)
after_shelves = apply_ops(shelves, RETIER_OPS) if RETIER_OPS else shelves
after_tree = render_tree(after_shelves)
before_wide = sum(1 for v in fanout(shelves).values() if len(v) > FANOUT_TARGET)
after_wide = sum(1 for v in fanout(after_shelves).values() if len(v) > FANOUT_TARGET)
print(f"wide parents: before={before_wide} after={after_wide}")'''
    )
)

cells.append(
    code(
        r'''from jinja2 import Template

REPORT = Template(
    """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Layer-A re-tiering — foods facet</title><style>
  body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;}
  h1{border-bottom:3px solid #4c72b0;padding-bottom:.3rem;}
  h2{color:#2a3f5f;border-left:5px solid #4c72b0;padding-left:.6rem;margin-top:2rem;}
  .cols{display:flex;gap:1.5rem;align-items:flex-start;}
  .col{flex:1;min-width:0;}
  ul.ftree,ul.ftree ul{list-style:none;margin:0;padding-left:1.1rem;border-left:1px dotted #cbd2dc;}
  ul.ftree{padding-left:0;border-left:none;}
  ul.ftree li{margin:.12rem 0;font-size:.88rem;}
  ul.ftree summary{cursor:pointer;}
  .c{color:#888;font-size:.8em;margin-left:.35rem;}
  .m{color:#c44e52;font-size:.82em;font-style:italic;}
  .meta{color:#888;font-size:.85rem;}
  .stat{background:#f5f7fa;padding:.6rem 1rem;border-radius:6px;}
</style></head><body>
<h1>Layer-A re-tiering — foods facet</h1>
<p class="meta">{{ n }} foods shelves · model {{ model }} · grouping tiers are real FoodOn ancestors only</p>
<div class="stat">Wide parents (&gt; {{ target }} children): <b>{{ before_wide }}</b> before →
<b>{{ after_wide }}</b> after.{% if not have_groq %} <i>(Part B skipped: GROQ_API_KEY unset — "after" = "before".)</i>{% endif %}</div>
<div class="cols">
  <div class="col"><h2>Before</h2>{{ before }}</div>
  <div class="col"><h2>After (re-tiered)</h2>{{ after }}</div>
</div></body></html>"""
)

out = VIZ_DIR / "retier_foods_report.html"
out.write_text(
    REPORT.render(
        n=len(shelves), model=fs.llm.model_id, target=FANOUT_TARGET,
        before_wide=before_wide, after_wide=after_wide, have_groq=HAVE_GROQ,
        before=before_tree, after=after_tree,
    ),
    encoding="utf-8",
)
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")'''
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
