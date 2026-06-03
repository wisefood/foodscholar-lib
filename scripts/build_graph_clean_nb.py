"""Assemble notebooks/graph_build.ipynb — a clean, linear graph-build notebook.

One section per phase (Configure -> Load -> Embed -> Entities -> Layer A ->
Attach -> Layer B -> Interactive tree), using the build_graph config (env-driven
GROQ key, no secrets) and the validated Layer-A path: 1a+ backbone projection
(`projection="backbone"`, the default) + the LLM aliasing pass. A `BACKEND` toggle
runs it either fully offline (memory + the annotated parquet) or against
Elastic/Neo4j. Ends by rendering data/viz/layer_a_tree_foods.html via
`fs.viz.layer_a_tree(...).render("tree")`.

    /mnt/miniconda3/envs/foodscholar/bin/python scripts/build_graph_clean_nb.py
"""

from __future__ import annotations

import nbformat as nbf

NB_PATH = "notebooks/graph_build.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells: list = []

cells.append(md(
    """# Graph build — Layer A + Layer B

Clean, linear build of the FoodScholar knowledge graph:

| # | Phase | Call |
|---|-------|------|
| 1 | Configure | `FoodScholar.from_config(...)` |
| 2 | Load corpus + annotations | `load_chunks` (offline) or `ingest` (Elastic) |
| 3 | Embed chunks | `fs.embed()` |
| 4 | Build entities | `fs.build_entities()` |
| 5 | **Layer A** (FoodOn projection + aliasing) | `fs.build_layer_a()` |
| 6 | Attach chunks to shelves | `fs.attach()` |
| 7 | **Layer B** (themes) | `fs.build_layer_b()` |
| 8 | **Interactive tree** | `fs.viz.layer_a_tree(...).render("tree")` |

Layer A uses the 1a+ backbone projection (umbrella rule + single-parent + single-child
collapse) and then an LLM **aliasing** pass that gives jargon shelves a friendly
`display_label` — additive, ids/structure untouched. Set `BACKEND='memory'` to run
fully offline from `data/annotated.parquet` (build it once with
`scripts/make_annotated_parquet.py`); set `BACKEND='elastic'` for the real stores."""
))

# ── 1. Configure ────────────────────────────────────────────────────────────
cells.append(md("## 1. Configure"))
cells.append(code(
    '''import os
from pathlib import Path

from foodscholar import FoodScholar
from foodscholar.config import FoodScholarConfig
from foodscholar.logging import configure_logging

configure_logging(level="INFO")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

# Secrets come from the environment — never hardcode. Without a key the LLM steps
# (aliasing, LLM theme labels) fall back to the mock client.
if not os.environ.get("GROQ_API_KEY"):
    print("\\u26a0 GROQ_API_KEY not set — LLM steps will use the MOCK client.")

BACKEND = "elastic"   # "elastic" (real Elasticsearch + Neo4j stores) or "memory" (offline, reads the parquet)

CORPUS_DIR = ROOT / "data" / "foodscholar" / "corpus"
NEL_DIR    = ROOT / "data" / "foodscholar" / "ner"
SNAPSHOT   = ROOT / "data" / "annotated.parquet"

_storage = (
    {"chunk_store": {"backend": "memory"}, "graph_store": {"backend": "memory"}}
    if BACKEND == "memory" else
    {"chunk_store": {"backend": "elastic", "url": "http://localhost:9200",
                     "index": "foodscholar_chunks"},
     "graph_store": {"backend": "neo4j", "url": "bolt://localhost:7687",
                     "user": "neo4j", "password": "password"}}
)

cfg = FoodScholarConfig.model_validate({
    "corpus": {"chunks_path": str(CORPUS_DIR), "annotated_snapshot_path": str(SNAPSHOT)},
    "ontology": {"foodon_path": str(ROOT / "data" / "foodon.owl"),
                 "cache_path": str(ROOT / "data" / "foodon_cache.parquet"),
                 "prefix_filter": ["FOODON:"]},
    "llm": {"primary": {"provider": "groq", "model": "llama-3.1-8b-instant"}},
    # Layer A: 1a+ backbone projection (the default) + LLM aliasing.
    # Same link_blocklist as build_graph — keeps generic mentions off the
    # 'food product' umbrella so the same nodes are dropped. (The deliberate
    # difference vs build_graph is the METHOD: 1a+ backbone + aliasing here,
    # not bottom_up_grouping.)
    "layer_a": {
        "projection": "backbone",   # 1a+ backbone-first controlled expansion (default)
        "link_blocklist": [
            {"surface": "fish", "ontology_id": "FOODON:00002281"},
            {"surface": "foods", "ontology_id": "FOODON:00001002"},
            {"surface": "food products", "ontology_id": "FOODON:00001002"},
            {"surface": "food product", "ontology_id": "FOODON:00001002"},
            {"surface": "whole foods", "ontology_id": "FOODON:00001002"},
            {"surface": "perishable foods", "ontology_id": "FOODON:00001002"},
            {"surface": "perishable products", "ontology_id": "FOODON:00001002"},
            {"surface": "foods and beverages", "ontology_id": "FOODON:00001002"},
            {"surface": "food and beverages", "ontology_id": "FOODON:00001002"},
            {"surface": "certain foods", "ontology_id": "FOODON:00001002"},
            {"surface": "specific food", "ontology_id": "FOODON:00001002"},
            {"surface": "real food", "ontology_id": "FOODON:00001002"},
            {"surface": "superfoods", "ontology_id": "FOODON:00001002"},
            {"surface": "imported foods", "ontology_id": "FOODON:00001002"},
            {"surface": "competitive foods", "ontology_id": "FOODON:00001002"},
            {"surface": "local foods", "ontology_id": "FOODON:00001002"},
        ],
    },
    "storage": _storage,
})

fs = FoodScholar.from_config(cfg)
print("backend:", BACKEND, "· llm:", fs.llm.model_id)'''
))

# ── 2. Load corpus + annotations ─────────────────────────────────────────────
cells.append(md(
    """## 2. Load corpus + annotations

Offline (`memory`): read the annotated parquet (run `scripts/make_annotated_parquet.py`
once to build it — fast, no Elastic). Real stores (`elastic`): ingest corpus + the
pre-computed NEL CSVs."""
))
cells.append(code(
    '''if BACKEND == "memory":
    if not SNAPSHOT.exists():
        raise SystemExit("Run scripts/make_annotated_parquet.py first to build "
                         f"{SNAPSHOT}.")
    n = fs.load_chunks(str(SNAPSHOT))
    print(f"loaded {n} chunks from {SNAPSHOT.name}")
else:
    fs.init()
    meta = fs.ingest(CORPUS_DIR, nel_dir=NEL_DIR, ignore_source_types=["abstract"])
    print(meta)'''
))

# ── 3. Embed ─────────────────────────────────────────────────────────────────
cells.append(md(
    """## 3. Embed chunks

Needed by Layer B Pass 1 (similarity) and kNN search. A `memory` backend uses a
deterministic **mock** embedder (instant; Pass-1 themes are then non-semantic). A
real backend uses BGE-base."""
))
cells.append(code('meta = fs.embed()\nprint(meta)'))

# ── 4. Build entities ────────────────────────────────────────────────────────
cells.append(md("## 4. Build entities"))
cells.append(code('meta = fs.build_entities()\nprint(f"entities: {len(fs.entities)}")'))

# ── 5. Build Layer A ─────────────────────────────────────────────────────────
cells.append(md(
    """## 5. Build Layer A — 1a+ backbone projection + aliasing

`build_layer_a` uses the **1a+** method (`projection="backbone"`, the default):
pick the facet root's supported children as a backbone, then expand each down the
real FoodOn tiers — **collapsing** single-child filing tiers, placing every node
under a **single parent**, capping fan-out, and **pruning** empty dead-ends. Then
the **aliasing pass** gives jargon shelves a human `display_label`. Faithful —
FoodOn ids / is-a edges / original labels intact."""
))
cells.append(code(
    '''meta = fs.build_layer_a()
shelves = fs.graph_store.list_shelves()
by_facet = {}
for s in shelves:
    by_facet[s.facet] = by_facet.get(s.facet, 0) + 1
n_aliased = sum(1 for s in shelves if s.display_label)
print(f"{len(shelves)} shelves · by facet {by_facet} · {n_aliased} aliased")'''
))

# ── 6. Attach ────────────────────────────────────────────────────────────────
cells.append(md("## 6. Attach chunks to shelves"))
cells.append(code('meta = fs.attach()\nprint(meta)'))

# ── 7. Build Layer B ─────────────────────────────────────────────────────────
cells.append(md(
    """## 7. Build Layer B — themes (per-shelf + merged, LLM topics)

Both passes run **per shelf** — Pass 1 (embedding similarity) and Pass 2 (FoodOn
entity relatedness) — then **merge** into themes. Topic labels are **LLM-generated**
(`labeling="llm"`, needs `GROQ_API_KEY`; falls back to keywords without it). With
the mock embedder, Pass-1 similarity is non-semantic — use a real backend/embedder
for meaningful Pass-1 themes."""
))
cells.append(code(
    '''fs.config.layer_b.pass1_mode = "per_shelf"      # Pass 1 per shelf (not global)
fs.config.layer_b.labeling.strategy = "llm"      # LLM-generated topic labels
art = fs.build_layer_b(facet="foods", dry_run=False)
print(f"themes: {art.n_themes_total} · by pass {art.n_themes_by_pass} · "
      f"shelves themed {art.n_shelves_themed}")'''
))

# ── 8. Interactive tree ──────────────────────────────────────────────────────
cells.append(md(
    """## 8. Interactive Layer A tree

Render the constructed Layer A foods tree to a self-contained HTML file and show it
inline. Left pane: the full shelf hierarchy with a `Chunks: total (D: direct |
L: lifted)` badge and a `[theme count]` per node (sub-threshold shelves greyed).
Click a shelf to see its Layer B topics on the right, grouped by **origin** —
Merged / Similarity / Relatedness — with a per-origin filter."""
))
cells.append(code(
    '''from IPython.display import IFrame

# Layer A tree + Layer B topics grouped by origin. Click a shelf on the left to
# see its topics on the right, split into Merged / Similarity / Relatedness
# (with a per-origin filter). Self-contained HTML, written to data/viz/.
out = fs.viz.layer_a_tree("foods").render(
    "tree", output=ROOT / "data" / "viz" / "layer_a_tree_foods.html"
)
print("wrote", out)

# IFrame src is relative to this notebook (in notebooks/), hence the ../.
IFrame(src="../data/viz/layer_a_tree_foods.html", width="100%", height=700)'''
))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "foodscholar", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH} with {len(cells)} cells")
