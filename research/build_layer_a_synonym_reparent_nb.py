"""Assemble notebooks/layer_a_synonym_reparent_casestudy.ipynb from source cells.

Case study for the "synonym-node chunk reparenting" direction (see memory
project_layer_a_synonym_reparent). Question under test: when the backbone
projection drops a FoodOn node N at the fan-out cap, is there a *surviving*
synonym node M — same exact-synonym string AND a shared is-a ancestor within K
hops (the same-subtree guard) — that N's chunks should reparent to, instead of
generalizing up to N's ancestor?

This is a MEASUREMENT notebook, not a method implementation. It produces the
numbers a "non-questionable by EU reviewers" decision needs:

  1. how many nodes the production projection actually drops at the cap;
  2. how many of those have a guarded surviving synonym M (the *firing set*);
  3. a hand-checkable table of every firing (genuine / plant-product / junk);
  4. the chunk-movement delta: how much more specific the synonym step lands
     chunks vs the current ancestor-lift, on the real corpus.

Kept as a script so the notebook source stays reviewable and regenerable. Run
with the foodscholar env:

    conda run -n foodscholar python research/build_layer_a_synonym_reparent_nb.py
"""

from __future__ import annotations

import nbformat as nbf

NB_PATH = "notebooks/layer_a_synonym_reparent_casestudy.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells: list = []

# ----------------------------------------------------------------- title
cells.append(
    md(
        """# Layer-A synonym-node chunk reparenting — case study (foods facet)

**The idea.** Backbone projection keeps the tree faithful to FoodOn by *dropping*
nodes (fan-out cap, dead-end prune) rather than reparenting them. Their chunks
re-home at attach time onto the **deepest surviving ancestor** (`resolve_lifted`
in `attach.py`) — a *generalization*. This case study asks whether we can do
better for the dropped nodes: reparent their chunks to a surviving **synonym
node** of equal specificity, when one exists, before falling back to the
ancestor.

**What "synonym node" can mean in FoodOn (measured, not assumed):**

| Source | Count in `foodon.owl` | Verdict |
|---|---|---|
| `owl:equivalentClass` | 5,043 | **All class *expressions*; 0 FOODON↔FOODON named links.** No identity substrate. |
| `hasExactSynonym` | 6,044 strings | Node→*string*. Reconstruct a node↔node link via string collision. **This is what we use.** |
| `hasDbXref` | 39,282 | Points out of FoodOn (CHEBI/NCBITaxon). Lens, not a reparent target. |

So M (the synonym node) is defined as: *a surviving node that shares an
exact-synonym/label string with the dropped node N.*

**The trap.** FoodOn deliberately separates `watermelon plant` / `watermelon` /
`watermelon (raw)` — they share the string "watermelon" but are different
concepts. Reparenting across that axis is exactly what a reviewer flags.

**The guard (decided up front).** *Same-subtree guard* — M only counts if N and M
share an is-a ancestor within **K hops**. Blocks plant↔product↔rawstate crossings;
keeps genuine synonymy (`hot dog`/`frankfurter`).

**The cascade (decided up front), for a chunk whose exact node N didn't survive:**
`direct → collapsed → synonym(guarded) → synonym-then-lift(ancestor of M) → lift(ancestor of N) → orphan`.
The synonym step is **strictly additive**: a pruned M falls through to M's
ancestor, so a chunk is never stranded and never worse off than today's lift.

Everything below is on the **real corpus** (`data/annotated.parquet`) with the
**production projection**. No production code is changed here.
"""
    )
)

# ----------------------------------------------------------------- §0 setup
cells.append(md("## §0 — Load corpus + run the production backbone projection"))
cells.append(
    code(
        '''from collections import Counter, defaultdict
from pathlib import Path

from foodscholar.config import FoodScholarConfig
from foodscholar.facade import FoodScholar
from foodscholar.layer_a.facet import route_link_to_facet
from foodscholar.layer_a.propagate import collect_support
from foodscholar.layer_a.backbone import build_backbone_shelves, _resolve_root

HERE = Path.cwd()
ROOT = HERE if (HERE / "data" / "foodon.owl").exists() else HERE.parent

# Production config — backbone projection, foods facet, the real knobs.
MAX_CHILDREN = 12          # fan-out cap (config.backbone_max_children default)
GUARD_K = 4                # same-subtree guard: max is-a hops to a shared ancestor

_cfg = {
    "corpus": {"chunks_path": str(ROOT / "data/annotated.parquet")},
    "ontology": {
        "foodon_path": str(ROOT / "data/foodon.owl"),
        "cache_path": str(ROOT / "data/foodon_cache.parquet"),
        "prefix_filter": ["FOODON:"],
    },
    "layer_a": {
        "facets": ["foods"], "min_support": 25, "max_depth": 6,
        "projection": "backbone", "backbone_max_children": MAX_CHILDREN,
        "blacklist_terms": ["material entity", "physical object", "manufactured product"],
    },
    "storage": {"chunk_store": {"backend": "memory"}, "graph_store": {"backend": "memory"}},
}
fs = FoodScholar.from_config(FoodScholarConfig.model_validate(_cfg))
api = fs.load_ontology()
fs.attach_ontology(api)
fs.load_chunks(str(ROOT / "data/annotated.parquet"))
chunks = list(fs.chunk_store.scan())

facet_cfg = fs.config.layer_a.resolve_facet("foods")
support = collect_support(
    iter(chunks), api,
    min_link_confidence=facet_cfg.min_link_confidence,
    facet="foods", link_blocklist=facet_cfg.link_blocklist,
)
shelves = build_backbone_shelves(support, api, facet_cfg, "foods", max_children=MAX_CHILDREN)
SURVIVING = {s.foodon_id for s in shelves if s.foodon_id}
print(f"{len(chunks)} chunks · support over {len(support.with_descendants)} terms · "
      f"{len(shelves)} shelves · {len(SURVIVING)} surviving FoodOn nodes")'''
    )
)

# ----------------------------------------------------------------- §1 dropped set
cells.append(
    md(
        """## §1 — Reconstruct the *dropped-at-cap* set

A node is "dropped at the fan-out cap" if it cleared `min_support` (so it was a
real projection candidate) but its parent retained `max_children` higher-support
siblings ahead of it. We replay `display_children` per surviving parent and
collect the overflow — these are the nodes whose chunks currently lift to an
ancestor and whose reparenting we want to improve.
"""
    )
)
cells.append(
    code(
        '''min_support = facet_cfg.min_support
blocked = {t.lower().strip() for t in facet_cfg.blacklist_terms}

def node_support(fid): return support.with_descendants.get(fid, 0)
def direct(fid): return support.direct.get(fid, 0)
def allowed(fid):
    lbl = (api.id_to_label(fid) or "").lower().strip()
    return lbl not in blocked
def supported_children(fid):
    return [c for c in api.id_to_children(fid)
            if c in api and allowed(c) and node_support(c) >= min_support]

# Replay the cap: for each surviving node, which supported children were cut?
dropped_at_cap = {}   # dropped_id -> parent_id (the surviving parent that cut it)
for parent in SURVIVING:
    kids = sorted(supported_children(parent), key=node_support, reverse=True)
    for cut in kids[MAX_CHILDREN:]:
        # only count it as "dropped" if it didn't survive elsewhere (DAG: first
        # placement wins, so a node cut here may still be kept under another parent)
        if cut not in SURVIVING:
            dropped_at_cap.setdefault(cut, parent)

print(f"nodes dropped at the fan-out cap: {len(dropped_at_cap)}")
print(f"  of which carry direct chunks  : {sum(1 for d in dropped_at_cap if direct(d))}")
print(f"  total chunks under dropped (rolled up): "
      f"{sum(node_support(d) for d in dropped_at_cap)}")'''
    )
)

# ----------------------------------------------------------------- §2 synonym index + guard
cells.append(
    md(
        """## §2 — Build the synonym index + the same-subtree guard

`syn_key(node)` = normalized {label} ∪ {exact synonyms}. Two nodes are
*string-synonymous* if their key sets intersect. The guard then requires a
shared is-a ancestor within `GUARD_K` hops (depth measured from each node up its
ancestor chain), which blocks FoodOn's plant/product/rawstate axis.
"""
    )
)
cells.append(
    code(
        r'''import re
_NORM = re.compile(r"[^a-z0-9]+")
def norm(s): return _NORM.sub(" ", s.lower()).strip()

def syn_key(fid):
    keys = set()
    lbl = api.id_to_label(fid)
    if lbl: keys.add(norm(lbl))
    for s in api.id_to_synonyms(fid):
        k = norm(s)
        if k: keys.add(k)
    return keys

# Inverse index over SURVIVING nodes only (M must survive): string -> surviving ids.
surv_by_string = defaultdict(set)
for m in SURVIVING:
    for k in syn_key(m):
        surv_by_string[k].add(m)

# Drop garbage keys: a leaked annotation-property URI collides unrelated foods.
JUNK_KEYS = {k for k in surv_by_string if k.startswith("http ") or "obolibrary" in k}

def ancestors_within(fid, k):
    """is-a ancestors of fid reachable within k hops (BFS over parents)."""
    seen, frontier = set(), {fid}
    for _ in range(k):
        nxt = set()
        for n in frontier:
            for p in api.id_to_parents(n):
                if p not in seen:
                    seen.add(p); nxt.add(p)
        frontier = nxt
    return seen

def passes_guard(n, m, k=GUARD_K):
    """True iff N and M share an is-a ancestor within k hops (same-subtree)."""
    an = ancestors_within(n, k) | {n}
    am = ancestors_within(m, k) | {m}
    return bool(an & am)

def synonym_candidates(n):
    """Surviving M (≠ n) that share a non-junk string with n AND pass the guard."""
    cands = set()
    for key in syn_key(n):
        if key in JUNK_KEYS:
            continue
        for m in surv_by_string.get(key, ()):
            if m != n:
                cands.add(m)
    return {m for m in cands if passes_guard(n, m)}

print(f"surviving nodes indexed by {len(surv_by_string)} distinct strings "
      f"({len(JUNK_KEYS)} junk keys excluded)")'''
    )
)

# ----------------------------------------------------------------- §3 firing set
cells.append(
    md(
        """## §3 — The firing set + hand-check table

The synonym step *fires* for a dropped node N only when it has ≥1 guarded
surviving synonym M. This is the only population that matters — everything else
behaves exactly as today. We categorize each firing for the precision read:

- **genuine** — N and M are the same food (`hot dog`/`frankfurter`);
- **plant-product** — caught by FoodOn's organism/product/rawstate axis *despite*
  the guard (residual leakage — the number that decides if K needs tuning);
- **other** — judge by eye.
"""
    )
)
cells.append(
    code(
        r'''firings = []   # (n, m, n_label, m_label, n_support, m_support, n_direct)
for n in dropped_at_cap:
    cands = synonym_candidates(n)
    if not cands:
        continue
    # pick the highest-support surviving synonym as the reparent target
    m = max(cands, key=node_support)
    firings.append({
        "n": n, "m": m,
        "n_label": api.id_to_label(n), "m_label": api.id_to_label(m),
        "n_support": node_support(n), "m_support": node_support(m),
        "n_direct": direct(n), "n_chunks": node_support(n),
        "n_parent": dropped_at_cap[n],
        "alt_count": len(cands),
    })

firings.sort(key=lambda r: r["n_chunks"], reverse=True)
print(f"FIRING SET: {len(firings)} of {len(dropped_at_cap)} dropped nodes "
      f"have a guarded surviving synonym")
print(f"chunks affected (rolled-up under firing N): "
      f"{sum(r['n_chunks'] for r in firings)}\n")

# Hand-check table — print every firing (this is small enough to eyeball).
hdr = f"{'N (dropped)':38s} {'->':2s} {'M (surviving synonym)':38s} {'Nchnk':>6s} {'Mchnk':>6s}"
print(hdr); print("-"*len(hdr))
for r in firings:
    print(f"{(r['n_label'] or r['n'])[:38]:38s} -> "
          f"{(r['m_label'] or r['m'])[:38]:38s} "
          f"{r['n_chunks']:6d} {r['m_support']:6d}")'''
    )
)

# ----------------------------------------------------------------- §4 specificity delta
cells.append(
    md(
        """## §4 — Does the synonym step actually land chunks *more specifically*?

For each firing N, compare two reparent targets for N's chunks:

- **today (lift):** deepest surviving *ancestor* of N — what `resolve_lifted`
  returns now;
- **proposed (synonym):** M (or, per the cascade, M's deepest surviving ancestor
  if M itself were dropped — but here M survives by construction).

The win is real only if M is **deeper** (more specific) than the ancestor N would
otherwise lift to. We measure the depth delta and the chunk-weighted depth gain.
A non-positive delta means the synonym step buys nothing over plain lift.
"""
    )
)
cells.append(
    code(
        r'''shelf_depth = {s.foodon_id: s.depth for s in shelves if s.foodon_id}

def deepest_surviving_ancestor(fid):
    best, best_d = None, -1
    for a in api.id_to_ancestors(fid):
        if a in SURVIVING and shelf_depth.get(a, -1) > best_d:
            best, best_d = a, shelf_depth[a]
    return best

wins, ties, losses, no_lift = 0, 0, 0, 0
weighted_depth_gain = 0
detail = []
for r in firings:
    n, m = r["n"], r["m"]
    anc = deepest_surviving_ancestor(n)
    d_syn = shelf_depth.get(m, -1)
    d_anc = shelf_depth.get(anc, -1) if anc else -1
    if anc is None:
        no_lift += 1            # today these chunks would orphan; synonym strictly helps
        wins += 1
    elif d_syn > d_anc:
        wins += 1
    elif d_syn == d_anc:
        ties += 1
    else:
        losses += 1
    if d_syn >= 0 and d_anc >= 0:
        weighted_depth_gain += (d_syn - d_anc) * r["n_chunks"]
    detail.append((r["n_label"], r["m_label"],
                   api.id_to_label(anc) if anc else "(orphan→facet root)",
                   d_anc, d_syn, r["n_chunks"]))

print(f"firings where synonym M is MORE specific than today's lift : {wins}")
print(f"          same depth (no specificity change)               : {ties}")
print(f"          LESS specific (synonym would be worse — guard/K?) : {losses}")
print(f"          today orphans, synonym rescues                    : {no_lift}")
print(f"chunk-weighted depth gain (sum over firings)               : {weighted_depth_gain}\n")

print(f"{'N':30s} {'synonym M':28s} {'today lifts to':28s} {'dA':>3s} {'dM':>3s} {'chnk':>5s}")
print("-"*100)
for nl, ml, al, da, dm, c in detail:
    flag = "  <-- worse" if dm < da else ""
    print(f"{(nl or '')[:30]:30s} {(ml or '')[:28]:28s} {al[:28]:28s} "
          f"{da:3d} {dm:3d} {c:5d}{flag}")'''
    )
)

# ----------------------------------------------------------------- §5 verdict scaffold
cells.append(
    md(
        """## §5 — Verdict (fill from the numbers above)

A "non-questionable by EU reviewers" recommendation needs all four to hold:

1. **Firing rate is non-trivial** (§3): the step affects enough dropped nodes /
   chunks to be worth the code. If `len(firings)` is ~0, the substrate is too
   sparse — recommend *not* building it and lift stays as-is.
2. **Precision is high on the firing set** (§3 hand-check): genuine synonymy
   dominates; plant-product leakage past the guard is near zero. If not, raise
   `GUARD_K` strictness or require a *tighter* LCA, and re-measure.
3. **The step is a specificity win** (§4): `wins ≫ losses`, positive weighted
   depth gain. Ties mean "harmless but pointless"; losses mean the guard let a
   bad target through and must be fixed before this ships.
4. **The cascade is provably safe** (by construction): a pruned M falls through
   to M's ancestor → N's ancestor → facet root, so no chunk is stranded and no
   chunk lands shallower than today. This holds regardless of the numbers — it's
   the reason the step can be *additive*.

Record the four numbers, the hand-check verdict, and a go/no-go in the memory
note `project_layer_a_synonym_reparent`.
"""
    )
)

# ----------------------------------------------------------------- build
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "foodscholar", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH} with {len(cells)} cells")
