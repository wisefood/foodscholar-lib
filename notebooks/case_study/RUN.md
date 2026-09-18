# RUN — how to reproduce the case study (NB1 → NB4)

## Environment

```bash
conda activate foodscholar     # base env numpy is broken; this env is the gate
# notebooks import cs_common.py from this directory
cd notebooks/case_study
```

All four notebooks are runnable top-to-bottom with **Restart & Run All**, in
order. They import the shared module:

```python
import cs_common as cs
```

`SEED = 42` everywhere. Each notebook writes `artifacts/<nb>/env.json` recording
library versions, backend, seed, config hash (and `llm_model` for NB4).

## Ordering & shared state

The in-memory graph store is RAM-only, so **NB1 builds Layer A once and
persists it** to `artifacts/shared/` (graph JSON + chunk parquet). NB2–NB4
reload that snapshot in fresh kernels via `cs.load_graph(fs)` — no rebuild.

1. **NB1_setup_backbone** — build Layer A (backbone projection), resolve both
   anchors, M1 raw-vs-projected, persist the shared graph.
2. **NB2_themes** — Leiden vs BERTopic theme bake-off per anchor (M2). Re-saves
   the shared graph with the Leiden themes so NB3/NB4 can use them.
3. **NB3_retrieval** — flat vs graph-expanded retrieval per query (M3).
4. **NB4_cards** — Layer C cards, provenance, extract-vs-card (M4). **Needs a
   real LLM.**

## Data inputs (already on disk, repo `data/`)

| File | Role |
|------|------|
| `data/annotated_with_abstracts.parquet` | **34,359** NEL-annotated chunks (textbook 12,194 + guide 1,150 + abstract 21,015; no vectors in the parquet) |
| `data/cache/bge_base_emb_13767.npy` + `..._ids_..json` | real BGE-base vectors: 13,252 textbook/guide + 515 anchor-subtree abstracts (218 olive/legume + 297 fish) |
| `data/cache/MANIFEST.json` | cache fingerprint (id-count + sha) written by `build_corpus.py` |
| `data/foodon_cache.parquet` | cached FoodOn terms (39,278) |
| `data/foodon.owl` | FoodOn source (only read if the cache misses) |

**Rebuilding the corpus from raw NER/NEL** (one idempotent, offline command — no
Elastic/Neo4j): `python build_corpus.py` collapses the expanded abstracts NEL to
doc-level, filters to NEL-covered abstracts, stages `corpus_cs/`+`ner_cs/`,
ingests → `annotated_with_abstracts.parquet`, and embeds the anchor subtrees →
the BGE cache (with alignment guards). The notebooks consume only the two outputs
above; `build_corpus.py` is the source of truth for how they were made.

## LLM (NB2 labels, NB4 cards)

```bash
export GROQ_API_KEY=...     # Groq llama-3.3-70b-versatile
```

Without the key: NB1 and NB3 run fully (no LLM needed). NB2 runs with **keyword
labels** (LLM polish skipped). NB4 **aborts loudly** with a real-LLM guard
rather than emitting mock "Mock answer" text — its acceptance is marked PENDING
in `summary.md`.

## Deviations from the build spec (the full log)

These are the only places the implementation departs from `case_study_spec.md`
/ the build spec. Each is also noted in the relevant `summary.md`.

1. **Second anchor re-homed.** Spec P2 = "dietary fibre" in `nutrients`; the
   real Layer A is FoodOn-only so `nutrients` is a flat stub. Re-anchored to
   `legume food product` (FOODON:00001264) in `foods`. See `prereg.md`.
2. **Embeddings.** `data/annotated.parquet` carries no vectors (it came from the
   NEL/ingest path, which doesn't embed). The in-memory facade would otherwise
   use a dim-8 mock embedder. Instead `cs.load_real_embeddings(fs)` attaches the
   cached real BGE-base vectors (99.2 % coverage). Query vectors (NB3) use the
   same BGE-base HF embedder so kNN is in-space.
3. **PNG export.** No graphviz `dot` binary in this environment, so the
   `"graphviz"` PNG backend is unavailable. Interactive shelf/theme trees are
   emitted as HTML (pyvis / `"tree"` backend); all M1–M4 analytic figures are
   rendered with matplotlib instead.
4. **Layer location.** The case study lives under `notebooks/case_study/`
   (so `import cs_common` resolves) rather than a top-level `case_study/`.

## Artifacts

Everything lands under `artifacts/<nb>/` (+ `artifacts/shared/` for the
cross-notebook graph snapshot). Each notebook writes figures (`*.png`,
`*.html`), data (`*.json` / `*.csv`), and a `summary.md` listing what was
produced and any deviation/limitation.
