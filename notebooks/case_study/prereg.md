# Pre-registration — FoodScholar qualitative case study

> ## ⚠️ SUPERSEDED — reconciliation (read first)
>
> This file is the **original frozen intent**, kept verbatim below for the audit
> trail. The run evolved beyond it. What actually ran:
>
> | Item | Pre-registered (below) | What actually ran |
> |---|---|---|
> | Corpus | `annotated.parquet`, 13,344 chunks, no abstracts | `annotated_with_abstracts.parquet`, **34,359** chunks (textbook 12,194 + guide 1,150 + **abstract 21,015**) |
> | Anchors | 2 (olive_oil, legume) | **3** — `fish` (FOODON:00001248) added mid-run |
> | Embeddings cache | `bge_base_emb_13252.npy`, 99.2% coverage | `bge_base_emb_13767.npy`, **40%** coverage (textbook/guide + anchor-subtree abstracts only; rest BM25-only) |
> | Facets | FoodOn-only (`prefix_filter=["FOODON:"]`) → only `foods` populated | `prefix_filter=None` → **4 facets populated** (foods, nutrients, health, sustainability); dietary_patterns & allergies stay stubs |
> | Card register | not pre-registered | 1 textbook-sourced (olive_oil) + 2 abstract-sourced (legume, fish) — emerged from the theme picker |
>
> The four lenses/metrics and the rubric below still hold. The actual results are
> in each notebook's `summary.md`; deviations are flagged there and in
> `DELIVERABLE_6_results.md`. Nothing below this banner has been altered.

---

# Pre-registration — FoodScholar qualitative case study

Frozen **before** running NB1–NB4. The deliverable quotes this file as evidence
that anchors, queries, lenses and the rubric were fixed in advance, not chosen
to flatter the results. Edits after the first NB run must be logged in
`RUN.md` with a reason.

Config hash (over the `CONFIG` dict in `cs_common.py`) is recorded in every
notebook's `env.json`.

## Anchors (two paradigms)

| Key | Paradigm | Facet | Target label | FoodOn id |
|-----|----------|-------|--------------|-----------|
| `olive_oil` | P1 — a specific food | foods | olive oil | FOODON:03301826 |
| `legume`    | P2 — a food as a fibre source | foods | legume food product | FOODON:00001264 |

**Deviation from the original build spec (frozen here, not post-hoc).** The
spec's second anchor was *"dietary fibre"* in a `nutrients` facet. In the real
build, Layer A is FoodOn-only (`prefix_filter = ["FOODON:"]`), so the
`nutrients` facet is a degenerate flat stub — a single root shelf with no
hierarchy and no themes — and no fibre shelf exists. We therefore re-anchor P2
onto the canonical high-fibre **food** subtree, `legume food product`
(760 chunks, depth 3, with bean / lentil / soybean descendants), keeping both
anchors in the one facet Layer A actually populates. The intent of the P2
query (dietary fibre → blood sugar & cholesterol) is preserved; legumes are its
prototypical food carrier. This deviation is repeated in every `summary.md`.

Anchor resolution is by case-insensitive label match (exact preferred over
substring; ties broken by chunk_count). If the primary labels miss, the
pre-registered `fallback_labels` are tried and the fallback is logged loudly.

## Retrieval queries (verbatim, NB3)

- `olive_oil`: **"Is olive oil good for cardiovascular health?"**
- `legume`: **"How do legumes and dietary fibre affect blood sugar and cholesterol?"**

`retrieval_k = 10`. Flat baseline = hybrid BM25 + kNN (RRF) over chunks;
graph-expanded = flat hits enriched with theme-sibling + shelf-attached chunks,
re-ranked. The delta (graph − flat) is annotated per hit.

## Four lenses / metrics

1. **Lens 1 / M1 — Backbone.** Raw FoodOn ancestor chain & fan-out vs the
   projected Layer A backbone for each anchor (depth, sibling fan-out,
   single-child runs).
2. **Lens 2 / M2 — Themes.** Two theme sets per anchor shelf: Leiden two-pass
   vs BERTopic single-pass. Per method: n_themes, coverage, keyword terms,
   sample chunks.
3. **Lens 3 / M3 — Retrieval.** Flat vs graph-expanded result sets per query
   and the annotated delta (paraphrase / different-food-entity /
   different-document).
4. **Lens 4 / M4 — Cards.** One Layer C card per anchor theme; provenance
   trace (cited chunks resolved / total); Stage-1 extractive extract vs the
   LLM card (refinement / drift). **Requires a real LLM** (Groq
   llama-3.3-70b-versatile); never the mock.

## Rubric (pass/record per claim)

- Every figure has a matching machine-readable JSON/CSV so claims are checkable.
- Both anchors carried through all four lenses (or a fallback recorded once and
  propagated).
- M2 (Leiden vs BERTopic) and M3 (graph vs flat) complete for **both** anchors.
- Provenance resolution reported honestly (target 100 %; unresolved citations
  flagged, never hidden).
- A weaker-than-hoped result is reported as a finding/limitation, never
  massaged. Any step that cannot run is marked FAILED in `summary.md` with the
  reason.

## Environment facts pinned at pre-registration

- Corpus: `data/annotated.parquet` — 13,344 chunks (textbook 12,194 + guide
  1,150; abstracts already excluded), NEL-annotated, **no embeddings in the
  parquet**.
- Real vectors: `data/cache/bge_base_emb_13252.npy` (13,252 × 768, BGE-base),
  attached at load → 99.2 % chunk coverage. The in-memory facade's dim-8 mock
  embedder is **not** used for retrieval.
- Ontology: `data/foodon_cache.parquet` (FoodOn, 39,278 terms cached).
- Seed = 42 everywhere.
- LLM: Groq `llama-3.3-70b-versatile`. NB4 (and NB2 LLM labels) need
  `GROQ_API_KEY`; absent → those steps are marked PENDING, the notebooks run
  with keyword labels and fail loudly rather than fabricate card text.
