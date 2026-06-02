"""Assemble notebooks/bottomup_entrypoints.ipynb from source cells.

Bottom-up Layer-A entry-point construction: start from the FoodOn leaf terms the
corpus actually mentions, then group them into recognizable filter entry points.
Inverts the current top-down prune. Compares grouping rules side by side, judged
by eye.

Run with the foodscholar env:
    /mnt/miniconda3/envs/foodscholar/bin/python scripts/build_bottomup_entrypoints_nb.py
"""

from __future__ import annotations

import nbformat as nbf

NB_PATH = "notebooks/bottomup_entrypoints.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells: list = []

cells.append(
    md(
        """# Bottom-up entry-point construction — foods facet

The current Layer A is **top-down**: start at FoodOn's roots, walk down, prune by
support. Casualty — specific foods the corpus mentions (`bean`, `mackerel`,
`porridge`) get pruned, so they have no filter entry point (the audit found 2,198
such misses).

**Bottom-up inverts it**: start from the FoodOn **leaf terms the corpus actually
mentions** and treat those as entry points; group them only to keep the list
browsable — never drop them. A mentioned food can't disappear, because the
mention is what creates the entry.

| Rule | Grouping |
|------|----------|
| **1 — Keep leaves** | no grouping; only merge near-duplicate label variants |
| **2 — Support floor (N)** | roll a rare leaf up its is-a ancestors until support ≥ N and a recognizable name; merge variants |
| **3 — LLM groups (semantic)** | LLM proposes ~15 human groups (anchored to real FoodOn ids); each leaf is assigned to a group **semantically** (by label), NOT via is-a ancestry — because FoodOn's is-a graph doesn't reliably place foods under their common-sense group |

> Runs in-process (no Neo4j/Elastic). `foodscholar` kernel. Rule 3 needs
> `GROQ_API_KEY`; without it a fixed default grouping shows the mechanism."""
    )
)

cells.append(md("## §0 — Corpus leaf terms + evidence + helpers"))

cells.append(
    code(
        r'''import html as _html
import os
import re
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
    "corpus": {"chunks_path": str(ROOT / "tests/fixtures/sample_chunks.jsonl"),
               "annotated_snapshot_path": str(ROOT / "data/annotated.parquet")},
    "ontology": {"foodon_path": str(ROOT / "data/foodon.owl"),
                 "cache_path": str(ROOT / "data/foodon_cache.parquet"), "prefix_filter": ["FOODON:"]},
    "layer_a": {"facets": ["foods"]},
    "storage": {"chunk_store": {"backend": "memory"}, "graph_store": {"backend": "memory"}},
}
if HAVE_GROQ:
    _cfg["llm"] = {"primary": {"provider": "groq", "model": "llama-3.1-8b-instant"}}
cfg = FoodScholarConfig.model_validate(_cfg)

fs = FoodScholar.from_config(cfg)
api = fs.load_ontology()
fs.attach_ontology(api)
fs.load_chunks(str(ROOT / "data/annotated.parquet"))
FOOD_PRODUCT = api.name_to_id("food product")

# DIRECT evidence: chunk-id SET per FoodOn food leaf the corpus mentions.
# We keep sets (not just counts) so a GROUP's size can be the count of DISTINCT
# chunks (union over its leaves) — a chunk mentioning 3 foods in one group must
# count once, else group totals exceed the corpus size.
DIRECT_CHUNKS = {}  # leaf fid -> set(chunk_id)
for c in fs.chunk_store.scan():
    seen = set()
    for fid in (getattr(c, "foodon_ids", []) or []):
        if fid in api and (fid == FOOD_PRODUCT or api.is_subclass_of(fid, FOOD_PRODUCT)):
            seen.add(fid)
    for ln in (getattr(c, "entity_links", []) or []):
        if ln.ontology_id in api and route_link_to_facet(ln) == "foods":
            seen.add(ln.ontology_id)
    for fid in seen:
        DIRECT_CHUNKS.setdefault(fid, set()).add(c.chunk_id)

DIRECT = Counter({fid: len(s) for fid, s in DIRECT_CHUNKS.items()})  # per-leaf counts

# LIFTED support: a node's support = its direct + all mentioned descendants'.
LIFTED = Counter()
for fid, n in DIRECT.items():
    LIFTED[fid] += n
    for a in api.id_to_ancestors(fid):
        if a in api:
            LIFTED[a] += n

print(f"{len(DIRECT)} distinct food leaf terms mentioned in corpus")'''
    )
)

cells.append(
    code(
        r'''# Nameability guard + variant-merge + display-label cleaning (shared by all rules).
ORG_TOKENS = (
    "food product", "food material", "plant material", "mammal material", " material",
    "edible food", "by taxonomy", "consumer group", "natural extractive", "cultural food",
    "animal-derived food", "ingredient", "analog", "wholesale", "retail", "produce (raw)",
    "species", "family", "(liquid)", "multi-component", "processed food", "food supplement",
    "dietary supplement", "formulation food", "food by meal type",
)
DATA_TOKENS = ("datum", "percent daily value", "serving size", "calorie", "(ec)", "(efsa",
               "mononitrate", "preparation", "legislated")
PROC_TOKENS = ("cooking", "baking", "frying", "modification process")
GARBLED_RE = re.compile(r"^\d|\(\(|\), |\(.+,.+\)|foodex|ground,|artificially|\braw\)")


def organizational(fid) -> bool:
    l = (api.id_to_label(fid) or "").lower()
    return any(t in l for t in ORG_TOKENS + DATA_TOKENS + PROC_TOKENS) or bool(GARBLED_RE.search(l))


QUAL = {"lowfat", "low", "fat", "nonfat", "skim", "whole", "raw", "fresh", "dried",
        "reduced", "fat-free", "cooked", "baked", "2", "1", "powdered", "concentrate"}


def concept_key(fid) -> str:
    l = (api.id_to_label(fid) or "").lower()
    toks = [t for t in re.split(r"[^a-z0-9]+", l) if t and t not in QUAL]
    return " ".join(toks) or l


_SYN_BAD = re.compile(r"\d|\(|,|;|:")  # codes / parenthetical / list-y synonyms


def _clean_synonym(fid):
    """Best clean EXACT FoodOn synonym for display, or None. Projected (FoodOn's
    own data) — prefer a short, plain synonym over the suffixed label."""
    base = re.sub(r"\s+food product$", "", (api.id_to_label(fid) or "")).lower()
    cands = [s for s in api.id_to_synonyms(fid, include_related=False)
             if s and not _SYN_BAD.search(s) and 2 <= len(s) <= 30]
    # prefer a synonym that ISN'T just the base label, shortest first
    cands.sort(key=len)
    for s in cands:
        if s.lower() != base:
            return s
    return cands[0] if cands else None


def clean_label(fid) -> str:
    """Display label: FoodOn synonym if a clean one exists, else strip the
    ' food product' suffix, else raw label. Fully FoodOn-sourced (no invention)."""
    syn = _clean_synonym(fid)
    if syn:
        return syn
    lbl = api.id_to_label(fid) or fid
    return re.sub(r"\s+food product$", "", lbl)


def finalize(entry_of):
    """leaf->entry map -> (entries Counter by representative fid, leaf->rep map).
    Entries sharing a concept_key merge to the highest-support representative."""
    group_support = Counter()
    rep = {}
    for leaf, n in DIRECT.items():
        e = entry_of[leaf]
        k = concept_key(e)
        group_support[k] += n
        if k not in rep or LIFTED[e] > LIFTED[rep[k]]:
            rep[k] = e
    entries = Counter({rep[k]: c for k, c in group_support.items()})
    leaf_to_rep = {leaf: rep[concept_key(entry_of[leaf])] for leaf in DIRECT}
    return entries, leaf_to_rep


COLUMNS = []
TRACK = ["bean", "mackerel", "porridge", "lowfat cow milk", "cow whole milk",
         "olive oil", "broccoli", "apple", "banana", "lard", "cantaloupe"]
TRACK_FIDS = {name: api.name_to_id(name) for name in TRACK}


def column(title, entries, leaf_to_rep, note="", label_of=None, group_reps=None):
    """Render a column.

    `label_of`: optional {fid -> display name} override. For Rule 3 a grouped
    entry's fid is the group anchor, but it should display by the GROUP NAME, not
    the anchor's raw FoodOn label.
    `group_reps`: optional set of fids that ARE group entry points — these bypass
    the organizational guard (the guard judges display labels, not group anchors).
    """
    label_of = label_of or {}
    group_reps = group_reps or set()

    def disp(fid):
        if fid in label_of:
            return label_of[fid]
        return clean_label(fid) if fid in api else str(fid)

    def is_clean(fid):
        if fid in group_reps:
            return True
        return fid in api and not organizational(fid)

    clean = [(e, c) for e, c in entries.most_common() if is_clean(e)]
    leaked = [(e, c) for e, c in entries.most_common() if not is_clean(e)][:12]
    fates = []
    for name, fid in TRACK_FIDS.items():
        fates.append((name, disp(leaf_to_rep[fid]) if fid and fid in leaf_to_rep else "—"))
    COLUMNS.append({
        "title": title, "note": note,
        "n_total": len(entries), "n_clean": len(clean),
        "top": [(disp(e), c) for e, c in clean[:30]],
        "leaked": [(disp(e), c) for e, c in leaked],
        "fates": fates,
    })
    print(f"{title}: {len(entries)} entries ({len(clean)} clean, {len(leaked)} leaked-org)")


print("helpers ready")'''
    )
)

cells.append(md("## Rule 1 — Keep leaves (variant-merge only)"))

cells.append(
    code(
        '''entry_of = {leaf: leaf for leaf in DIRECT}
entries, leaf_to_rep = finalize(entry_of)
column("1 — Keep leaves", entries, leaf_to_rep,
       note="Max coverage: nothing rolls up; only label-variant duplicates merge.")'''
    )
)

cells.append(md("## Rule 2 — Support floor (roll up until grouped support ≥ N)"))

cells.append(
    code(
        r'''def rollup(N):
    """Each mentioned leaf rolls up its FoodOn ancestors (closest first) until it
    hits a node with lifted support >= N that is a recognizable (non-org) name.
    Falls back to the leaf itself if no such ancestor exists below food product."""
    entry_of = {}
    for leaf in DIRECT:
        chain = sorted({leaf} | {a for a in api.id_to_ancestors(leaf) if a in api},
                       key=lambda x: -len(api.id_to_ancestors(x)))  # leaf -> root
        pick = None
        for node in chain:
            if node == FOOD_PRODUCT:
                break
            if LIFTED[node] >= N and not organizational(node):
                pick = node
                break
        entry_of[leaf] = pick or leaf
    return entry_of


for N in (20, 50):
    entries, leaf_to_rep = finalize(rollup(N))
    column(f"2 — Support floor N={N}", entries, leaf_to_rep,
           note=f"Roll up until grouped support ≥ {N} and a recognizable name; then merge variants.")'''
    )
)

cells.append(
    md(
        """## Rule 3 — LLM groups the leaves (semantic assignment)

FoodOn has no single foods tree, and — critically — its **is-a graph does not
reliably place common foods under their common-sense food group** (e.g. the right
fruit ancestor for `apple` is `plant fruit food product`, not the `fruit (raw)`
node a naive anchor picks; many foods sit on a different axis entirely). So
grouping a leaf by walking is-a ancestry is fragile.

Instead: the LLM proposes ~15 human food groups (each still **anchored to a real
FoodOn id** so we stay in FoodOn), and every leaf is assigned to a group
**semantically by its label** — not by is-a ancestry. This is the
"infer-in-the-process" grouping the structure can't provide. A leaf the LLM
can't place keeps its own entry (coverage held)."""
    )
)

cells.append(
    code(
        r'''def split_concepts(group_name):
    parts = re.split(r"\s+and\s+|\s*,\s*|\s*/\s*", group_name.strip())
    return [p.strip().lower() for p in parts if p.strip()]


def anchor_for_concept(concept):
    """A real FoodOn id to anchor a group concept to (for display/grounding).
    Exact/suffix match preferred; else shortest clean search hit; else None."""
    singular = concept.rstrip("s")
    for cand in (concept, singular, concept + " food product", singular + " food product"):
        fid = api.name_to_id(cand)
        if fid:
            return fid
    clean = [h for h in api.search(concept, limit=12)
             if not organizational(h) and "(" not in (api.id_to_label(h) or "")
             and concept_key(h) in {concept, singular}]
    return min(clean, key=lambda h: len(api.id_to_label(h) or "")) if clean else None


DEFAULT_GROUPS = [
    "Vegetables", "Fruits", "Dairy and Eggs", "Grains and Pasta", "Bread and Bakery",
    "Meat and Poultry", "Fish and Seafood", "Legumes and Beans", "Nuts and Seeds",
    "Oils and Fats", "Beverages", "Sweets and Snacks", "Herbs and Spices", "Sauces and Condiments", "Soups",
]


def propose_groups():
    if not HAVE_GROQ:
        print("GROQ_API_KEY not set — using FIXED default groups to show the mechanism.")
        return DEFAULT_GROUPS, False
    schema = {"type": "object",
              "properties": {"groups": {"type": "array", "items": {"type": "string"}}},
              "required": ["groups"]}
    sample = ", ".join(api.id_to_label(fid) for fid, _ in DIRECT.most_common(50))
    prompt = (
        "Propose 12-16 intuitive TOP-LEVEL food groups for browsing a nutrition "
        "knowledge base (human category names like 'Vegetables', 'Dairy and Eggs', "
        "'Fish and Seafood'). Frequent corpus foods for context:\n"
        f"{sample}\n\nReturn JSON {{\"groups\": [\"...\"]}}."
    )
    try:
        raw = fs.llm.generate_json(prompt, schema, max_tokens=400)
        names = (raw or {}).get("groups", [])
        if not names:
            print("LLM proposal returned no groups; raw object was:", repr(raw)[:300])
    except Exception as exc:
        print("LLM group proposal failed, using defaults:", exc)
        names = []
    # `live` reflects whether the KEY is available, not whether THIS parse worked
    # — leaf assignment (the part that matters) should still run live even if the
    # proposal fell back to defaults.
    return (names or DEFAULT_GROUPS), HAVE_GROQ


GROUP_NAMES, live = propose_groups()
print(f"[debug] live={live} (HAVE_GROQ={HAVE_GROQ}), proposal source="
      f"{'LLM' if GROUP_NAMES is not DEFAULT_GROUPS else 'DEFAULT'}")
# Anchor each group to a real FoodOn id where possible (grounding only).
GROUP_ANCHORS = {}
for nm in GROUP_NAMES:
    anchors = [a for a in (anchor_for_concept(c) for c in split_concepts(nm)) if a]
    GROUP_ANCHORS[nm] = anchors
print(f"{'LLM' if live else 'default'}: {len(GROUP_NAMES)} groups")
for nm in GROUP_NAMES:
    print(f"   {nm:24s} anchors: {[api.id_to_label(a) for a in GROUP_ANCHORS[nm]]}")'''
    )
)

cells.append(
    code(
        r'''# Semantic leaf -> group assignment by LABEL (not is-a). Batched: we assign each
# DISTINCT leaf label once. With GROQ, the LLM classifies leaf labels into the
# group list in batches; without it, a keyword heuristic stands in to show shape.
GROUP_LIST = list(GROUP_NAMES)
NONE_LABEL = "(other)"

# keyword hints per group for the no-LLM heuristic fallback.
_HINTS = {
    "Vegetables": ["vegetable", "broccoli", "carrot", "spinach", "lettuce", "cabbage",
                   "onion", "pepper", "tomato", "potato", "celery", "kale", "squash",
                   "cucumber", "beet", "greens", "sprout"],
    "Fruits": ["fruit", "apple", "banana", "orange", "berry", "melon", "grape", "peach",
               "pear", "mango", "citrus", "avocado", "cantaloupe", "raisin", "apricot"],
    "Dairy and Eggs": ["milk", "cheese", "yogurt", "cream", "butter", "dairy", "egg"],
    "Grains and Pasta": ["grain", "rice", "pasta", "oat", "wheat", "barley", "cereal", "corn", "flour"],
    "Bread and Bakery": ["bread", "bagel", "muffin", "cake", "cookie", "pastry", "roll", "bun", "biscuit", "tortilla"],
    "Meat and Poultry": ["meat", "beef", "pork", "chicken", "turkey", "poultry", "ham",
                          "bacon", "sausage", "lamb", "veal"],
    "Fish and Seafood": ["fish", "salmon", "tuna", "shrimp", "seafood", "sardine", "mackerel",
                         "cod", "crab", "shellfish"],
    "Legumes and Beans": ["bean", "legume", "lentil", "pea", "chickpea", "soy", "tofu", "hummus", "tempeh"],
    "Nuts and Seeds": ["nut", "almond", "walnut", "peanut", "seed", "sesame", "sunflower", "cashew"],
    "Oils and Fats": ["oil", "fat", "lard", "margarine", "shortening"],
    "Beverages": ["beverage", "drink", "juice", "coffee", "tea", "soda", "wine", "beer", "water", "cola"],
    "Sweets and Snacks": ["candy", "chocolate", "sweet", "snack", "chip", "popcorn", "pretzel", "sugar", "dessert", "pudding"],
    "Herbs and Spices": ["herb", "spice", "seasoning", "salt", "pepper spice"],
    "Sauces and Condiments": ["sauce", "condiment", "ketchup", "mustard", "mayonnaise", "dressing", "salsa", "relish", "jam", "spread"],
    "Soups": ["soup", "broth", "stew", "chowder"],
}


def assign_heuristic(label):
    low = label.lower()
    for g in GROUP_LIST:
        for kw in _HINTS.get(g, []):
            if kw in low:
                return g
    return NONE_LABEL


def assign_llm_batch(labels):
    """Classify a batch of leaf labels into GROUP_LIST. Returns {label: group}."""
    schema = {"type": "object", "properties": {"assignments": {"type": "array",
              "items": {"type": "object",
                        "properties": {"food": {"type": "string"}, "group": {"type": "string"}},
                        "required": ["food", "group"]}}}, "required": ["assignments"]}
    numbered = "\n".join(f"  - {l}" for l in labels)
    groups = ", ".join(GROUP_LIST)
    prompt = (
        f"Assign each food to ONE of these groups: {groups}, or '{NONE_LABEL}' if none fits.\n"
        f"Foods:\n{numbered}\n\nReturn JSON "
        '{"assignments": [{"food": "<food>", "group": "<group>"}, ...]} for every food.'
    )
    try:
        obj = fs.llm.generate_json(prompt, schema, max_tokens=4096)
    except Exception:
        return {}
    valid = set(GROUP_LIST) | {NONE_LABEL}
    return {a["food"]: a["group"] for a in obj.get("assignments", [])
            if a.get("group") in valid and a.get("food")}


# Assign each distinct leaf label. (Only the leaves actually mentioned — the rep
# label after variant-merge would be ideal, but leaves map 1:1 here for clarity.)
leaf_labels = {leaf: api.id_to_label(leaf) or leaf for leaf in DIRECT}
assignment = {}  # leaf_label -> group name
if HAVE_GROQ:
    uniq = sorted(set(leaf_labels.values()))
    BATCH = 60
    n_batches = (len(uniq) + BATCH - 1) // BATCH
    n_llm = 0
    for bi, i in enumerate(range(0, len(uniq), BATCH), 1):
        got = assign_llm_batch(uniq[i:i + BATCH])
        n_llm += len(got)
        assignment.update(got)
        print(f"[assign] batch {bi}/{n_batches}: {len(got)} labels classified "
              f"(running total {n_llm})")
    # fill any the LLM skipped with the heuristic so coverage holds
    n_heur = 0
    for lbl in uniq:
        if lbl not in assignment:
            assignment[lbl] = assign_heuristic(lbl)
            n_heur += 1
    mode = f"LLM ({n_llm} by LLM, {n_heur} by heuristic fallback)"
else:
    for lbl in set(leaf_labels.values()):
        assignment[lbl] = assign_heuristic(lbl)
    mode = "heuristic(no key)"

# Each group gets a STABLE synthetic entry-point id (so two groups can't collide
# on a shared anchor, and the entry displays by GROUP NAME not the anchor label).
# Grouped leaves attach to their group's entry; unassigned leaves keep their own.
group_eid = {nm: f"group::{nm}" for nm in GROUP_LIST}      # entry id per group
group_label = {group_eid[nm]: nm for nm in GROUP_LIST}     # entry id -> display name

# Accumulate DISTINCT chunk-id sets per entry (group or kept-leaf), then count
# the union — so a chunk mentioning several foods in one group counts ONCE.
entry_chunks = {}             # entry id -> set(chunk_id)
leaf_to_rep = {}              # tracked-food fates use this
ungrouped = 0
for leaf in DIRECT:
    g = assignment.get(leaf_labels[leaf], NONE_LABEL)
    if g == NONE_LABEL or g not in group_eid:
        rep = leaf  # keep own entry (coverage held)
        ungrouped += 1 if g == NONE_LABEL else 0
    else:
        rep = group_eid[g]
    leaf_to_rep[leaf] = rep
    entry_chunks.setdefault(rep, set()).update(DIRECT_CHUNKS.get(leaf, ()))

entries = Counter({eid: len(s) for eid, s in entry_chunks.items()})  # distinct chunks

# group entry ids are recognizable by construction; ungrouped leaf ids judged by guard
group_reps = set(group_label)
label_of = dict(group_label)  # entry id -> group name for grouped entries
note = (f"{mode}: {len(GROUP_LIST)} semantic groups. {ungrouped} leaves unassigned "
        f"→ keep own entry (coverage held). Leaf→group is by LABEL, not is-a. "
        f"Grouped entries display by GROUP NAME; group anchors: "
        + "; ".join(f"{nm}={[api.id_to_label(a) for a in GROUP_ANCHORS.get(nm, [])]}"
                    for nm in GROUP_LIST[:4]) + " …")
column("3 — LLM groups (semantic)", entries, leaf_to_rep, note=note,
       label_of=label_of, group_reps=group_reps)
group_chunk = Counter({group_label[e]: c for e, c in entries.items() if e in group_label})
print("group sizes (chunks, by semantic assignment):")
for g, c in group_chunk.most_common():
    print(f"   {c:5d}  {g}")'''
    )
)

cells.append(
    md(
        """## Rule 4 — PROJECTED depth-preserving Layer A (the closing approach)

Everything here is projected from FoodOn — entities, structure, depth, and labels:

- **Bottom-up coverage:** every corpus-mentioned leaf is kept.
- **Loosened support floor `K`:** a FoodOn node survives if `≥ K` distinct chunks
  reach it or a mentioned descendant. Low `K` keeps the mid-level grouping tiers
  that aggressive pruning used to dissolve.
- **Keep FoodOn's real is-a structure + depth:** the tree is the surviving FoodOn
  subtree under `food product`; each node's parent is its nearest surviving
  is-a ancestor. No flattening, no invented tiers.
- **Leaves attach by is-a** (a leaf's entry is itself; its ancestors are the
  browsable tiers above it).
- **Labels from FoodOn synonyms** (`clean_label`): synonym → strip-suffix → raw.
  No LLM, no invention.

We render the natural tree at a chosen `K` and print the depth/width profile so
the floor and any depth cap can be decided by eye. Output:
`data/viz/layer_a_projected_foods.html`."""
    )
)

cells.append(
    code(
        r'''import html as _html2

K_FLOOR = 5  # loosened support floor (probe showed K=5 → ~460 grouping tiers, fan-out ~26)

# Node support = distinct chunks reaching the node or any MENTIONED descendant.
node_chunks = {}
for leaf, n in DIRECT.items():
    chunks = DIRECT_CHUNKS.get(leaf, set())
    node_chunks.setdefault(leaf, set()).update(chunks)
    for a in api.id_to_ancestors(leaf):
        if a in api:
            node_chunks.setdefault(a, set()).update(chunks)
node_support = {fid: len(s) for fid, s in node_chunks.items()}

# Survivors: food-product descendants with support >= K, PLUS every mentioned leaf
# (coverage: a leaf is never dropped even if below K).
under_fp = lambda f: f == FOOD_PRODUCT or api.is_subclass_of(f, FOOD_PRODUCT)
survivors = {f for f, sup in node_support.items() if sup >= K_FLOOR and under_fp(f)}
survivors |= {leaf for leaf in DIRECT if under_fp(leaf)}
survivors.add(FOOD_PRODUCT)

# Projected parent = nearest surviving is-a ancestor (preserves real depth).
def nearest_surviving_parent(fid):
    ancs = [a for a in api.id_to_ancestors(fid) if a in survivors and a != fid]
    if not ancs:
        return None
    # nearest = the ancestor with the MOST ancestors itself (deepest / closest above)
    return max(ancs, key=lambda a: len(api.id_to_ancestors(a)))

# Parent map (nearest surviving is-a ancestor) + children, rebuildable after curation.
parent_of = {f: (nearest_surviving_parent(f) or FOOD_PRODUCT) for f in survivors if f != FOOD_PRODUCT}

def build_children(parent_map):
    ch = defaultdict(list)
    for f, p in parent_map.items():
        ch[p].append(f)
    return ch

proj_children = build_children(parent_of)

def recompute(tag):
    """Depth + stats over the current proj_children. Returns nothing; prints + sets globals."""
    global proj_depth, depth_dist, fanout, n_internal
    proj_depth = {FOOD_PRODUCT: 0}
    stack = [(FOOD_PRODUCT, 0)]
    while stack:
        node, d = stack.pop()
        proj_depth[node] = d
        for c in proj_children.get(node, []):
            stack.append((c, d + 1))
    depth_dist = Counter(proj_depth.get(f, 0) for f in proj_depth)
    fanout = sorted((len(v) for v in proj_children.values()), reverse=True)
    n_internal = sum(1 for f in proj_children if proj_children.get(f))
    print(f"[{tag}] K={K_FLOOR}: {len(proj_depth)} nodes | {n_internal} grouping tiers | "
          f"max depth {max(proj_depth.values())} | top fan-out {fanout[:8]}")
    print(f"   depth dist: {dict(sorted(depth_dist.items()))}")

def subtree_chunks(node):
    s = set(DIRECT_CHUNKS.get(node, set()))
    for ch in proj_children.get(node, []):
        s |= subtree_chunks(ch)
    return s

recompute("raw is-a")
print(f"   every mentioned leaf present: {all(l in survivors for l in DIRECT if under_fp(l))}")'''
    )
)

cells.append(
    md(
        """### Rule 4b — LLM tier-curation (keep-tier vs demote), real FoodOn nodes only

The raw is-a tree is faithful but noisy: at one parent, real sub-categories
(`berry`, `citrus fruit`) sit beside specific foods (`banana`, `mango`),
botanical tiers (`solanaceous fruit`, `pomes`), and junk (`produce (raw)`).

Per parent with many children, the LLM judges each child **keep-as-navigation-tier
vs demote** — choosing only among the real FoodOn nodes we hand it (by index). A
**demoted** node is spliced out: its children re-home to its nearest *kept*
ancestor; it survives as a leaf only if it has direct chunks. Membership stays
is-a (projected); the LLM only decides tier-vs-leaf. Degrades to a heuristic
(demote botanical/junk labels) without `GROQ_API_KEY`."""
    )
)

cells.append(
    code(
        r'''# Heuristic tier-vs-leaf judgment (stand-in / fallback): demote botanical taxonomy
# and junk; keep recognizable category words.
_BOTANICAL = ("solanaceous", "pome", "pomes", "cucurbit", "drupe", "aggregate fruit",
              "rosaceae", "brassica", "allium", "by taxonomy", "true berry", "false fruit")
_TIER_WORDS = ("vegetable", "fruit", "berry", "citrus", "stone fruit", "melon", "nut",
               "seed", "grain", "cereal", "legume", "bean", "milk", "cheese", "meat",
               "poultry", "fish", "seafood", "oil", "fat", "bread", "dairy", "egg",
               "beverage", "juice", "herb", "spice", "sauce", "soup", "root vegetable",
               "leafy", "shellfish")

def heuristic_keep_tier(fid):
    """CONSERVATIVE fallback: demote ONLY clear botanical/organizational/junk
    labels; keep everything else as a tier. (The LLM is the real judge — an
    aggressive heuristic over-demotes and blows up fan-out when leaves re-home.)"""
    l = (api.id_to_label(fid) or "").lower()
    if organizational(fid):
        return False
    if any(b in l for b in _BOTANICAL):
        return False
    return True  # keep by default — only the LLM aggressively prunes tiers

def llm_keep_tiers(parent, children):
    """Return the subset of `children` to KEEP as tiers. LLM picks by index."""
    if not HAVE_GROQ:
        return {c for c in children if heuristic_keep_tier(c)}
    schema = {"type": "object", "properties": {"keep": {"type": "array",
              "items": {"type": "integer"}}}, "required": ["keep"]}
    blocks = "\n".join(f"  {i}. {clean_label(c)}  ({_sub(c)} chunks)"
                       for i, c in enumerate(children, 1))
    prompt = (
        f"Under the food category {clean_label(parent)!r}, which of these are real "
        f"NAVIGATION SUB-CATEGORIES a person would browse (e.g. 'berry', 'citrus "
        f"fruit', 'root vegetable') versus specific foods, botanical/scientific "
        f"groupings ('solanaceous fruit', 'pomes'), or junk ('produce (raw)') that "
        f"should be demoted to leaves? Keep only genuine browse sub-categories.\n"
        f"{blocks}\n\nReturn JSON {{\"keep\": [<indices to keep as sub-category>]}}."
    )
    try:
        raw_idx = fs.llm.generate_json(prompt, schema, max_tokens=512).get("keep", [])
    except Exception:
        return {c for c in children if heuristic_keep_tier(c)}
    # Groq structured output sometimes returns indices as strings — coerce to int.
    keep_idx = set()
    for x in raw_idx:
        try:
            keep_idx.add(int(x))
        except (TypeError, ValueError):
            continue
    return {children[i - 1] for i in keep_idx if 1 <= i <= len(children)}


# Curate: for each internal node, decide which children stay tiers; demote the rest
# (splice out — their children re-home to the nearest KEPT ancestor). Iterate top-down.
CURATE_MIN_CHILDREN = 6  # only bother curating parents wider than this
def _sub(node):  # subtree size cache (re-declared so this cell is self-contained)
    s = SUB_CACHE.get(node)
    if s is None:
        s = len(subtree_chunks(node)); SUB_CACHE[node] = s
    return s
SUB_CACHE = {}

kept_tier = set()   # fids confirmed as tiers
demoted = set()     # fids spliced out as tiers (still leaves if direct chunks)
# walk parents by depth (shallow first) so re-homing lands on already-decided tiers
for parent in sorted(proj_children, key=lambda p: proj_depth.get(p, 0)):
    kids = [c for c in proj_children.get(parent, []) if proj_children.get(c)]  # only sub-trees are tier candidates
    if len(proj_children.get(parent, [])) < CURATE_MIN_CHILDREN or not kids:
        continue
    keep = llm_keep_tiers(parent, kids)
    for c in kids:
        (kept_tier if c in keep else demoted).add(c)

# Rebuild parent_of: a demoted node is removed as a tier — its children point to the
# demoted node's parent (nearest kept ancestor, found transitively).
def nearest_kept(fid):
    p = parent_of.get(fid, FOOD_PRODUCT)
    while p in demoted:
        p = parent_of.get(p, FOOD_PRODUCT)
    return p

new_parent = {}
for f in survivors:
    if f == FOOD_PRODUCT:
        continue
    if f in demoted and not DIRECT_CHUNKS.get(f):
        continue  # spliced out entirely (no direct chunks → not even a leaf)
    new_parent[f] = nearest_kept(f)
parent_of = new_parent
proj_children = build_children(parent_of)
SUB_CACHE = {}  # subtree sizes change after re-homing
recompute("curated")
print(f"   kept {len(kept_tier)} tiers, demoted {len(demoted)} "
      f"({'LLM' if HAVE_GROQ else 'heuristic'})")

# Single-child / near-duplicate chain collapse: FoodOn has chains of near-identical
# nodes (apple / apple (whole) / apple food product) that render as
# `apple ← apple ← apple`. Collapse a parent INTO its child when the parent has
# exactly one child AND (parent has no direct chunks of its own OR shares the
# child's concept_key). The child takes the parent's parent. Iterate to fixed point.
collapsed = 0
changed = True
while changed:
    changed = False
    for parent in list(proj_children):
        if parent == FOOD_PRODUCT:
            continue
        kids = proj_children.get(parent, [])
        # Same-label/concept child: merge even if the parent has other children
        # (the apple/apple/apple case — a near-duplicate child folds into parent).
        # Compare on the DISPLAY label too, since FoodOn near-dups share a label
        # even when concept_key differs slightly.
        pk, pl = concept_key(parent), clean_label(parent).lower()
        same_kids = [c for c in kids
                     if concept_key(c) == pk or clean_label(c).lower() == pl]
        if same_kids:
            child = same_kids[0]
            # re-home the merged child's children to `parent`, then drop child
            for gc in proj_children.get(child, []):
                parent_of[gc] = parent
            parent_of.pop(child, None)
            proj_children = build_children(parent_of)
            collapsed += 1
            changed = True
            break
        if len(kids) != 1:
            continue
        child = kids[0]
        same = concept_key(parent) == concept_key(child)
        no_own = not DIRECT_CHUNKS.get(parent)
        if not (same or no_own):
            continue
        # rewire child to parent's parent; drop parent
        gp = parent_of.get(parent, FOOD_PRODUCT)
        parent_of[child] = gp
        parent_of.pop(parent, None)
        proj_children = build_children(parent_of)
        collapsed += 1
        changed = True
        break
SUB_CACHE = {}
recompute("collapsed")
print(f"   collapsed {collapsed} single-child / near-duplicate chain links")'''
    )
)

cells.append(
    code(
        r'''# Render the projected subtree as a collapsible <details> tree with synonym labels.
SUB_CACHE = {}
def _sub(node):
    if node not in SUB_CACHE:
        SUB_CACHE[node] = len(subtree_chunks(node))
    return SUB_CACHE[node]

def render_node(node, rel, max_depth=None, open_depth=1, max_children=40):
    label = _html2.escape(clean_label(node))
    kids = sorted(proj_children.get(node, []), key=lambda c: -_sub(c))
    nch = len(kids)
    direct = len(DIRECT_CHUNKS.get(node, set()))
    badge = (f"<span class='c'>{nch} sub · {_sub(node):,} chunks"
             + (f" · {direct} direct" if direct else "") + "</span>")
    if not kids or (max_depth is not None and rel >= max_depth):
        extra = f" <span class='m'>(+{nch} below)</span>" if nch else ""
        return f"<li>{label}{badge}{extra}</li>"
    shown = kids[:max_children]
    hidden = nch - len(shown)
    inner = "".join(render_node(c, rel + 1, max_depth, open_depth, max_children) for c in shown)
    if hidden:
        inner += f"<li class='m'>+{hidden} more…</li>"
    op = " open" if rel < open_depth else ""
    return f"<li><details{op}><summary><b>{label}</b>{badge}</summary><ul>{inner}</ul></details></li>"

# DEPTH_CAP: None = natural depth (decide cap after seeing it). Set an int to cap.
DEPTH_CAP = None
proj_tree = f"<ul class='ftree'>{render_node(FOOD_PRODUCT, 0, max_depth=DEPTH_CAP)}</ul>"

from jinja2 import Template
PROJ_REPORT = Template("""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Layer A — projected (depth-preserving)</title><style>
  body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:1.5rem auto;padding:0 1rem;}
  h1{border-bottom:3px solid #4c72b0;padding-bottom:.3rem;}
  .meta{color:#888;font-size:.85rem;} .stat{background:#f5f7fa;padding:.5rem .8rem;border-radius:6px;font-size:.85rem;}
  ul.ftree,ul.ftree ul{list-style:none;margin:0;padding-left:1rem;border-left:1px dotted #cbd2dc;}
  ul.ftree{padding-left:0;border-left:none;} ul.ftree li{margin:.1rem 0;font-size:.86rem;}
  ul.ftree summary{cursor:pointer;} .c{color:#8a93a0;font-size:.8em;margin-left:.35rem;} .m{color:#c44e52;font-size:.82em;font-style:italic;}
</style></head><body>
<h1>Layer A — projected, depth-preserving (foods)</h1>
<p class="meta">Fully projected from FoodOn: entities, is-a structure, depth, and synonym labels. Support floor K={{k}}, depth cap={{cap}}.</p>
<div class="stat">{{ nodes }} surviving FoodOn nodes · {{ internal }} grouping tiers · max depth {{ maxd }} · top fan-out {{ fan }}<br>
depth distribution: {{ ddist }}</div>
{{ tree }}
</body></html>""")

proj_out = VIZ_DIR / "layer_a_projected_foods.html"
proj_out.write_text(PROJ_REPORT.render(
    k=K_FLOOR, cap=DEPTH_CAP, nodes=len(survivors), internal=n_internal,
    maxd=max(proj_depth.values()), fan=fanout[:6], ddist=dict(sorted(depth_dist.items())),
    tree=proj_tree), encoding="utf-8")
print(f"wrote {proj_out} ({proj_out.stat().st_size/1024:.0f} KB)")
print("tracked foods + their FINAL (curated+collapsed) ancestor path:")
for nm in ("apple", "bean", "mackerel", "broccoli", "cow milk"):
    fid = api.name_to_id(nm)
    # a tracked leaf may have been merged away by collapse — follow to a present node
    cur = fid
    while cur is not None and cur not in parent_of and cur != FOOD_PRODUCT:
        cur = nearest_surviving_parent(cur)
    path = []
    while cur is not None and len(path) < 6:
        path.append(clean_label(cur))
        cur = parent_of.get(cur) if cur != FOOD_PRODUCT else None
    print(f"   {nm}: " + " ← ".join(path))'''
    )
)

cells.append(md("## Assemble side-by-side report"))

cells.append(
    code(
        r'''from jinja2 import Template

REPORT = Template(
    """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Bottom-up entry points — foods</title><style>
  body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1600px;margin:1.5rem auto;padding:0 1rem;}
  h1{border-bottom:3px solid #4c72b0;padding-bottom:.3rem;}
  .grid{display:flex;gap:1rem;align-items:flex-start;overflow-x:auto;}
  .col{flex:1 0 360px;min-width:360px;border:1px solid #e0e4ea;border-radius:8px;padding:.6rem .8rem;}
  .col h2{font-size:1rem;color:#2a3f5f;margin:.2rem 0;} .note{font-size:.8rem;color:#666;margin-bottom:.5rem;}
  .count{background:#eef6ee;border-radius:5px;padding:.3rem .6rem;font-size:.82rem;margin-bottom:.5rem;}
  table{border-collapse:collapse;font-size:.82rem;width:100%;} td,th{border:1px solid #eef;padding:2px 7px;text-align:left;}
  td.n{color:#888;text-align:right;width:3.2em;}
  .fates{font-size:.8rem;margin:.5rem 0;} .fates td{border:none;padding:1px 6px;} .fates .k{color:#2a3f5f;font-weight:600;}
  .leak{margin-top:.5rem;font-size:.78rem;color:#a0322b;} .leak b{color:#a0322b;}
  .meta{color:#888;font-size:.85rem;}
</style></head><body>
<h1>Bottom-up entry points — foods facet</h1>
<p class="meta">{{ n_leaves }} corpus-mentioned food leaves · bottom-up: leaves are entry points, grouped only to stay browsable · judge by eye</p>
<div class="grid">
{% for c in columns %}
  <div class="col">
    <h2>{{ c.title }}</h2><div class="note">{{ c.note }}</div>
    <div class="count"><b>{{ c.n_clean }}</b> recognizable entry points <span class="meta">({{ c.n_total }} total)</span></div>
    <div class="fates"><table>
      <tr><td colspan=2 class="meta">where tracked foods land:</td></tr>
      {% for k,v in c.fates %}<tr><td class="k">{{k}}</td><td>→ {{v}}</td></tr>{% endfor %}
    </table></div>
    <table><tr><th>top entry points</th><th>chunks</th></tr>
    {% for lbl,n in c.top %}<tr><td>{{lbl}}</td><td class="n">{{n}}</td></tr>{% endfor %}</table>
    {% if c.leaked %}<div class="leak"><b>guard misses</b> (organizational labels leaked in):<br>
    {% for lbl,n in c.leaked %}{{lbl}} ({{n}}){% if not loop.last %} · {% endif %}{% endfor %}</div>{% endif %}
  </div>
{% endfor %}
</div></body></html>"""
)

out = VIZ_DIR / "bottomup_entrypoints_foods.html"
out.write_text(REPORT.render(n_leaves=len(DIRECT), columns=COLUMNS), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, {len(COLUMNS)} rules)")'''
    )
)

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "foodscholar", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH} with {len(cells)} cells")
