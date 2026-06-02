"""Assemble notebooks/entrypoint_audit.ipynb from source cells.

Audits the foods facet of Layer A as a FACET/FILTER INDEX (not a browse tree):
for the foods a user would name, does a recognizable entry point exist? Three
diagnostics: (1) nameability classification of existing shelves, (2) named foods
with evidence that were pruned (missing entry points), (3) redundant/fragmented
entry points. No tree-building.

Run with the foodscholar env:
    /mnt/miniconda3/envs/foodscholar/bin/python scripts/build_entrypoint_audit_nb.py

Spec basis: project memory layer-a-projection-rethink (CRITICAL REFRAME).
"""

from __future__ import annotations

import nbformat as nbf

NB_PATH = "notebooks/entrypoint_audit.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells: list = []

cells.append(
    md(
        """# Layer-A entry-point audit — foods facet

Layer A is a **facet / filter index**: a user arrives with a *specific question*
(\"does olive oil help heart disease?\") and clicks entry points (`olive oil`,
`cardiovascular health`) to filter chunks. So **\"works\" = for the foods a user
would name, a recognizable, well-named entry point exists and is findable.**

This notebook audits the current foods shelves against that yardstick — it does
**not** build a tree. Three diagnostics:

1. **Nameability** — classify every shelf: recognizable food vs organizational /
   data-artifact / garbled-label / process.
2. **Coverage gaps** — foods with real corpus evidence that have **no shelf**
   (pruned out → user names them, finds nothing).
3. **Redundancy** — one concept fragmented into many near-duplicate entry points.

> Runs in-process (no Neo4j/Elastic). `foodscholar` kernel."""
    )
)

cells.append(md("## §0 — Build the current foods facet"))

cells.append(
    code(
        '''import html as _html
import re
from collections import Counter
from pathlib import Path

from foodscholar.config import FoodScholarConfig
from foodscholar.facade import FoodScholar
from foodscholar.layer_a.facet import route_link_to_facet

HERE = Path.cwd()
ROOT = HERE if (HERE / "data" / "foodon.owl").exists() else HERE.parent
VIZ_DIR = ROOT / "data" / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

cfg = FoodScholarConfig.model_validate(
    {
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
)
fs = FoodScholar.from_config(cfg)
api = fs.load_ontology()
fs.attach_ontology(api)
fs.load_chunks(str(ROOT / "data/annotated.parquet"))
fs.build_layer_a()
shelves = [s for s in fs.graph_store.list_shelves() if s.facet == "foods"]
shelf_fids = {s.foodon_id for s in shelves if s.foodon_id}
FOOD_PRODUCT = api.name_to_id("food product")
MIN_SUPPORT = cfg.layer_a.min_support
print(f"{len(shelves)} foods shelves at min_support={MIN_SUPPORT}")'''
    )
)

cells.append(
    md(
        """## §1 — Nameability: is each shelf a food a user would name?

Heuristic classifier (transparent, no LLM). Buckets:

- **recognizable** — a clean, common food name a user would type/click
- **organizational** — FoodOn filing terms (`food product`, `edible food`, `ingredient`, `… by taxonomy`)
- **data_artifact** — non-foods that leaked in (`percent daily value`, `food calorie datum`, EC/datum terms)
- **garbled** — auto-generated / OCR / parenthetical-modifier labels nobody types (`38910 - …`, `fruit ((whole or pieces), raw)`)
- **process** — actions, not foods (`food cooking`, `stir-frying`)"""
    )
)

cells.append(
    code(
        r'''ORG_TOKENS = (
    "food product", "food material", "edible food", "by taxonomy", "consumer group",
    "natural extractive", "cultural food", "animal-derived food", "produce (raw)",
    "ingredient", "manufactured", "analog", "wholesale", "retail",
    "dietetic food", "nutritious food", "leftover", "food (solid)",
)
DATA_TOKENS = (
    "datum", "percent daily value", "serving size", "calorie", "(ec)", "(efsa",
    "mononitrate", "preparation", "supplement form", "legislated",
)
PROCESS_TOKENS = (
    "cooking", "baking", "-frying", "frying", "modification process", "food baking",
)
GARBLED_RE = re.compile(r"^\d|\(\(|\), |, raw\)|\(raw\)|\(.+,.+\)|foodex|\bground,|artificially")


def classify(label: str) -> str:
    low = label.lower().strip()
    if GARBLED_RE.search(low):
        return "garbled"
    if any(t in low for t in DATA_TOKENS):
        return "data_artifact"
    if any(t in low for t in PROCESS_TOKENS):
        return "process"
    if any(t in low for t in ORG_TOKENS):
        return "organizational"
    return "recognizable"


buckets = {k: [] for k in ("recognizable", "organizational", "data_artifact", "garbled", "process")}
for s in shelves:
    buckets[classify(s.label)].append(s)

print("nameability of current shelves:")
for k, v in buckets.items():
    print(f"  {k:15s} {len(v):3d}  ({100*len(v)/len(shelves):.0f}%)")
print("\n--- examples of NON-recognizable shelves (the noise in the filter) ---")
for k in ("organizational", "data_artifact", "garbled", "process"):
    ex = ", ".join(s.label for s in sorted(buckets[k], key=lambda s: s.label)[:8])
    print(f"  [{k}] {ex}")'''
    )
)

cells.append(
    md(
        """## §2 — Coverage gaps: named foods with evidence but NO entry point

For every FOODON food term, count chunks that directly link to it. A term with
real evidence but **no shelf** means: a user names that food → filter finds
nothing, even though the corpus has chunks about it. These are the misses that
matter most. We classify the misses too (a missing *organizational* node is
fine; a missing *recognizable food* is the bug)."""
    )
)

cells.append(
    code(
        '''freq = Counter()
for c in fs.chunk_store.scan():
    seen = set()
    for fid in (getattr(c, "foodon_ids", []) or []):
        if fid in api and (fid == FOOD_PRODUCT or api.is_subclass_of(fid, FOOD_PRODUCT)):
            seen.add(fid)
    for ln in (getattr(c, "entity_links", []) or []):
        if ln.ontology_id in api and route_link_to_facet(ln) == "foods":
            seen.add(ln.ontology_id)
    for fid in seen:
        freq[fid] += 1

missing = [(n, fid) for fid, n in freq.items() if fid not in shelf_fids]
missing.sort(reverse=True)
# split the misses by nameability
missing_reco = [(n, fid) for n, fid in missing if classify(api.id_to_label(fid) or "") == "recognizable"]
missing_other = [(n, fid) for n, fid in missing if classify(api.id_to_label(fid) or "") != "recognizable"]

print(f"{len(freq)} foods with evidence · {len(missing)} have NO shelf "
      f"({len(missing_reco)} are recognizable foods — the real coverage bug)")
print(f"\\nTop 25 MISSING *recognizable* foods (chunk count · label) — user names these, finds nothing:")
for n, fid in missing_reco[:25]:
    print(f"  {n:4d}  {api.id_to_label(fid)}")
print(f"\\n(for contrast — top missing non-recognizable, mostly fine to omit:)")
for n, fid in missing_other[:8]:
    print(f"  {n:4d}  {api.id_to_label(fid)}  [{classify(api.id_to_label(fid) or '')}]")'''
    )
)

cells.append(
    md(
        """## §3 — Redundancy: one concept fragmented into many entry points

Near-duplicate shelves split a single filter intent across several entries
(`cow milk` / `cow whole milk` / `lowfat cow milk`; `wheat bread` / `whole wheat
bread`). Cheap signal: shelves whose labels share a head noun / are
substrings of each other."""
    )
)

cells.append(
    code(
        r'''STOP = {"food", "product", "raw", "whole", "fresh", "dried", "low", "fat", "lowfat",
        "nonfat", "reduced", "based", "or", "and", "the", "of", "a", "with"}


def head_tokens(label):
    toks = [t for t in re.split(r"[^a-z0-9]+", label.lower()) if t and t not in STOP]
    return frozenset(toks)


reco = sorted(buckets["recognizable"], key=lambda s: s.label)
groups = {}
for s in reco:
    key = head_tokens(s.label)
    matched = None
    for k in groups:
        if key & k and (key <= k or k <= key or len(key & k) >= 1 and (len(key) == 1 or len(k) == 1)):
            matched = k
            break
    if matched:
        groups[matched].append(s)
    else:
        groups[key] = [s]

frag = [(k, v) for k, v in groups.items() if len(v) > 1]
frag.sort(key=lambda kv: -len(kv[1]))
print(f"{len(frag)} fragmented concepts among recognizable shelves:")
for k, v in frag[:15]:
    print(f"  {', '.join(sorted(s.label for s in v))}")'''
    )
)

cells.append(md("## §4 — Assemble the audit report"))

cells.append(
    code(
        r'''from jinja2 import Template

def chips(items, cls):
    return "".join(f"<span class='chip {cls}'>{_html.escape(x)}</span>" for x in items)

REPORT = Template(
    """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Layer-A entry-point audit — foods</title><style>
  body{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:1.5rem auto;padding:0 1rem;}
  h1{border-bottom:3px solid #4c72b0;padding-bottom:.3rem;} h2{color:#2a3f5f;margin-top:1.8rem;}
  .meta{color:#888;font-size:.85rem;}
  .bar{display:flex;height:26px;border-radius:5px;overflow:hidden;margin:.5rem 0;font-size:.75rem;color:#fff;}
  .bar div{display:flex;align-items:center;justify-content:center;}
  .reco{background:#55a868;} .organizational{background:#c44e52;} .data_artifact{background:#937860;}
  .garbled{background:#8172b3;} .process{background:#ccb974;}
  .chip{display:inline-block;padding:.05rem .4rem;margin:.1rem;border-radius:4px;font-size:.8rem;background:#eef;}
  .chip.bad{background:#fdecea;color:#a0322b;} .chip.miss{background:#fff4e0;color:#8a5a00;}
  table{border-collapse:collapse;font-size:.85rem;width:100%;} td,th{border:1px solid #e0e4ea;padding:3px 8px;text-align:left;}
  .key{background:#f5f7fa;padding:.6rem 1rem;border-radius:6px;}
</style></head><body>
<h1>Layer-A entry-point audit — foods facet</h1>
<p class="meta">{{ n }} shelves · judged as a filter index: does a recognizable entry point exist for foods users name?</p>

<div class="key"><b>Verdict:</b> {{ reco }}/{{ n }} shelves ({{ reco_pct }}%) are recognizable food entry points;
the rest is filter noise. <b>{{ n_missing_reco }}</b> recognizable foods with corpus evidence have <b>no entry point at all</b>.</div>

<h2>§1 Nameability of current entry points</h2>
<div class="bar">
{% for k,c,pct in nameability %}<div class="{{k}}" style="width:{{pct}}%">{{c}}</div>{% endfor %}
</div>
<p class="meta">green=recognizable · red=organizational · brown=data-artifact · purple=garbled · yellow=process</p>
{% for k,items in noise.items() %}<p><b>{{k}}</b> ({{items|length}}): {{ chip(items, 'bad') }}</p>{% endfor %}

<h2>§2 Missing entry points (evidence but no shelf)</h2>
<p>Foods a user would name that return nothing — the real coverage failure:</p>
<table><tr><th>chunks</th><th>food (no entry point)</th></tr>
{% for n,l in missing %}<tr><td>{{n}}</td><td>{{l}}</td></tr>{% endfor %}</table>

<h2>§3 Fragmented entry points</h2>
<p>One filter intent split across several near-duplicate entries:</p>
{% for g in frag %}<p>{{ chip(g, '') }}</p>{% endfor %}
</body></html>"""
)
REPORT.globals["chip"] = chips

nameability = [(k, len(v), round(100 * len(v) / len(shelves)))
               for k, v in buckets.items()]
out = VIZ_DIR / "entrypoint_audit_foods.html"
out.write_text(
    REPORT.render(
        n=len(shelves), reco=len(buckets["recognizable"]),
        reco_pct=round(100 * len(buckets["recognizable"]) / len(shelves)),
        n_missing_reco=len(missing_reco),
        nameability=nameability,
        noise={k: sorted(s.label for s in buckets[k])
               for k in ("organizational", "data_artifact", "garbled", "process")},
        missing=[(n, api.id_to_label(fid)) for n, fid in missing_reco[:30]],
        frag=[sorted(s.label for s in v) for _, v in frag[:20]],
    ),
    encoding="utf-8",
)
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")'''
    )
)

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "foodscholar", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH} with {len(cells)} cells")
