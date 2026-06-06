# Layer C — Theme Summarization (extractive → LLM)

**Date:** 2026-06-06
**Status:** Design approved; ready for implementation plan
**Scope:** A production Layer C that summarizes each Layer B theme into a `Card`, using a cheap
extractive Stage 1 (map-reduce) feeding an LLM Stage 2, plus an evaluation harness for choosing and
tuning the Stage-1 method.

---

## 1. Problem & objective

We need to turn a large collection of text chunks belonging to a single **Layer B theme/community**
(tens to several hundred chunks) into a concise, useful summary **Card**.

The constraint is **cost efficiency**: sending hundreds of chunks straight to an LLM is expensive. The
objective is to determine whether **classical extractive methods** can produce a good enough
intermediate representation ("building blocks") that an LLM then cheaply refines — and to ship that as
the production Layer C while keeping the means to evaluate and swap the extractive method.

Two related deliverables:

1. **Production Layer C** — `build_layer_c()` iterates themes, runs a config-pinned extractive method
   (map-reduce) → LLM refinement → one `Card` per theme, persisted via the existing
   `graph_store.upsert_cards` path.
2. **Evaluation harness** — a `sweep`-style read-only runner that executes *all* extractive methods over
   a theme's chunks and emits per-method JSON metrics for side-by-side qualitative comparison. Used to
   pick/tune the Stage-1 method that production pins in config.

### Out of scope (this baseline)
- Gensim TextRank (gensim ≥4 removed `gensim.summarization`; we use sumy's TextRank as *the* TextRank).
- T5 / abstractive baselines (interface leaves room; deferred).
- Per-claim / per-sentence grounding. `grounding_check` is honored only as a lightweight length/non-empty
  guard in this baseline; full grounding is future work.
- Any change to the `Card` model. We map outputs onto the existing fields.
- Summarizing shelves (Layer A nodes). Layer C summarizes **themes** only here.

---

## 2. Decisions (locked)

| Decision | Choice |
|---|---|
| Deliverable | Full production Layer C **and** an eval harness |
| Summary unit | Layer B **themes** → one `Card` per theme (`target_type="theme"`) |
| Stage-1 methods | Sumy **LexRank, LSA, Luhn, TextRank** + **NLTK frequency**. No gensim. T5 deferred. |
| Scale strategy | **Map-reduce** grouping in Stage 1 (group → extract → concat → extract → LLM) |
| Stage-2 output | Refine into the **existing `Card`** fields (no schema change) |
| Method wiring | **Name → factory registry**; single source of truth for builder + harness |
| Eval vs prod | Both: harness runs all methods; production runs one config-pinned method |
| Dependencies | New `[summarization]` extra (`sumy`, `nltk`), lazy-imported |

---

## 3. Architecture & module layout

New package `src/foodscholar/layer_c/`, mirroring `layer_b/` conventions:

```
src/foodscholar/layer_c/
  __init__.py          # thin exports
  base.py              # BaseSummarizer ABC: summarize(chunks: list[str]) -> str
  summarizers.py       # 5 extractive impls (Sumy×4 + NLTK frequency) + _ensure_nltk_data()
  registry.py          # name -> factory(cfg) -> BaseSummarizer; used by builder AND harness
  stage1.py            # map-reduce orchestration around a chosen BaseSummarizer
  stage2.py            # LLM refinement: Stage-1 extract -> Card fields (prompt + parse)
  builder.py           # build_layer_c(fs, *, facet, dry_run): iterate themes -> Cards
  persist.py           # persist_cards(cards, graph_store) -> upsert_cards
  benchmark.py         # eval harness: run ALL methods over a theme -> per-method JSON metrics
  models.py            # internal models: MethodResult, Stage1Output, LayerCReport
```

**Touched existing files (additive, small):**
- `config.py` — extend `LayerCConfig` (keep existing fields; add Stage-1/map-reduce/eval knobs).
- `facade.py` — replace deferred `build_layer_c()` with a real call; add `benchmark_layer_c(...)`.
- `pyproject.toml` — new `[summarization]` extra (`sumy`, `nltk`).
- `cli/main.py` — `layer-c` build command + `layer-c-bench` command (matching existing CLI patterns).

**Lazy imports:** `sumy` / `nltk` are imported inside `summarizers.py` functions (gated by the extra),
like `leidenalg` / `python-igraph` in Layer B — the core package imports without summarization deps
installed. NLTK data (`punkt`, `punkt_tab`, `stopwords`) is fetched on first use by
`_ensure_nltk_data()` (no network at import time).

---

## 4. The summarizer interface, methods, and registry

### `base.py`
```python
class BaseSummarizer(ABC):
    name: str                      # "lexrank" | "lsa" | "luhn" | "textrank" | "nltk_freq"
    def summarize(self, chunks: list[str]) -> str: ...
```
One contract: take a list of chunk texts, return a single extractive summary string. The sentence
budget (`n`) is passed at construction by the factory, so the same class can be tuned per call.

### `summarizers.py` — 5 implementations
- **Sumy ×4** (`SumyLexRankSummarizer`, `SumyLsaSummarizer`, `SumyLuhnSummarizer`,
  `SumyTextRankSummarizer`): each wraps the corresponding `sumy.summarizers.*` over a `PlaintextParser`
  + `Tokenizer("english")`, returns the top-`n` sentences joined. A tiny shared base does
  parse/tokenize/join; only the sumy algorithm class differs.
- **`NLTKFrequencySummarizer`**: hand-rolled — `nltk.sent_tokenize` → stopword-filtered, normalized
  word-frequency sentence scores → top-`n` sentences in original order. No sumy dependency.

**Degenerate handling:** fewer sentences than budget → return all; empty input → return `""`. Pure and
unit-testable with plain strings (no stores, no LLM).

### `registry.py`
```python
SUMMARIZERS: dict[str, Callable[[LayerCConfig], BaseSummarizer]] = {
    "lexrank":   lambda c: SumyLexRankSummarizer(n=c.stage1_sentences),
    "lsa":       lambda c: SumyLsaSummarizer(n=c.stage1_sentences),
    "luhn":      lambda c: SumyLuhnSummarizer(n=c.stage1_sentences),
    "textrank":  lambda c: SumyTextRankSummarizer(n=c.stage1_sentences),
    "nltk_freq": lambda c: NLTKFrequencySummarizer(n=c.stage1_sentences),
}
def build_summarizer(name: str, cfg) -> BaseSummarizer: ...
def all_summarizers(cfg) -> list[BaseSummarizer]: ...   # for the harness
```
Production: `build_summarizer(cfg.stage1_method, cfg)`. Harness: `all_summarizers(cfg)`. One source of
truth — no drift between builder and harness.

---

## 5. Two-stage pipeline

### Stage 1 — map-reduce extractive (`stage1.py`)
```
chunks (list[str])
  ├─ if total sentences ≤ map_reduce_threshold:
  │     single pass → summarizer.summarize(chunks)
  └─ else:
        MAP:    partition chunks into groups by char budget (group_char_budget)
                → summarize each group independently → group_summaries
        REDUCE: summarizer.summarize(group_summaries) → one extract
```
- Threshold + budget are config knobs. Map-reduce drops nothing: every chunk flows into a group's
  extract, then the reduce pass distills across groups.
- The same registry-built summarizer is used for both map and reduce passes (one method to tune).
- Output: `Stage1Output{ text, n_input_chunks, n_input_chars, strategy: "single"|"mapreduce",
  n_groups }` — provenance the builder records.

### Stage 2 — LLM refinement into the existing `Card` (`stage2.py`)
- Prompt takes **only the Stage-1 extract** (the cost win) plus the theme's `label` / `keyword_terms`
  for context, and asks `fs.llm.generate_json` for: `title`, `summary` (narrative, internally organized
  as key messages / main claims / insights), `tip`, `evidence_quality`, optional `controversy_note` /
  `confidence_note`.
- Mapped onto the existing `Card`:
  - `summary` = the narrative body; `title` / `tip` / `evidence_quality` from the LLM.
  - `cited_chunk_ids` = the theme's member chunk ids (provenance = chunks that fed Stage 1; the LLM does
    not see them individually, so this is theme-level provenance, not per-sentence grounding).
  - `llm_model`, `prompt_version` from config; `generated_at` stamped at build.
- **Grounding:** `grounding_check` honored as a lightweight post-check — `strict` requires the summary
  non-empty and ≤ `max_summary_chars`, else flag/skip. (Full per-claim grounding = future work.)
- **Safety:** themes whose facet ∈ `safety_sensitive_facets` set `Card.safety_flagged=True` for
  downstream review (flag only; no behavioral gating in this baseline).

**Failure handling:** LLM error/empty after the fallback chain → record the theme as failed (Stage-1
extract retained in logs), continue; build reports succeeded/failed/skipped counts (Layer B
quality-report style).

---

## 6. Builder, persistence, harness

### `builder.py`
```python
def build_layer_c(fs, *, facet="foods", dry_run=False) -> LayerCReport:
    themes = fs.graph.themes()                      # optionally filtered by facet via shelf_ids
    cards, skipped, failed, strat = [], 0, 0, {}
    for theme in themes:
        ids   = fs.graph_store.get_chunks_for_theme(theme.theme_id)
        texts = [c.text for c in fs.chunk_store.get_many(list(ids))]
        if not texts: skipped += 1; continue
        s1 = run_stage1(texts, summarizer, cfg)     # map-reduce
        try:
            card = run_stage2(fs.llm, s1, theme, cfg)
        except Exception: failed += 1; continue
        cards.append(card); strat[s1.strategy] = strat.get(s1.strategy, 0) + 1
    if not dry_run: persist_cards(cards, fs.graph_store)
    return LayerCReport(n_themes=len(themes), n_cards=len(cards),
                        n_skipped=skipped, n_failed=failed, strategy_counts=strat)
```
- `dry_run=True` runs both stages but skips persistence (safe inspection) — like `build_layer_b`.
- One `Card` per theme; batched chunk fetch via `get_many`.

### `persist.py`
```python
def persist_cards(cards, graph_store) -> None:
    graph_store.upsert_cards(cards)   # Card carries target_id/target_type="theme" for routing
```
Single step (cards have no denorm). Mirrors `layer_b/persist.py`.

### `benchmark.py` — evaluation harness (read-only; no LLM, no persistence by default)
`benchmark_theme(fs, theme_id, cfg) -> list[MethodResult]` runs **every** registry method over one
theme's chunks (single-pass — comparing raw method quality) and emits per method:
```json
{ "method": "lexrank", "summary": "...", "input_chunks": 243,
  "input_chars": 184532, "execution_time_ms": 412, "summary_length_chars": 1840 }
```
`benchmark_facet(fs, facet, cfg, *, themes=N, out=PATH)` runs across the N largest themes, writes a
combined JSON under `benchmark_out_dir`, and prints a small console table (like `sweep_layer_b`).
Optional flag adds the configured LLM Stage 2 on the single best method; default is extractive-only.

### Facade + CLI
- `fs.build_layer_c(facet=…, dry_run=…)` → real impl; included in `fs.build()`.
- `fs.benchmark_layer_c(facet=…, themes=N)` → harness.
- CLI: `foodscholar layer-c [--facet --dry-run]`; `foodscholar layer-c-bench [--facet --themes N --out PATH]`.

---

## 7. Config

`LayerCConfig` keeps its existing fields and adds Stage-1 / map-reduce / eval knobs:
```python
class LayerCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # existing
    llm_model: str = "claude-sonnet-4-6"
    prompt_version: str = "v1"
    sample_size: int = 12
    grounding_check: Literal["strict", "lenient", "off"] = "strict"
    safety_sensitive_facets: list[Facet] = Field(default_factory=lambda: ["allergies"])
    # new
    stage1_method: Literal["lexrank","lsa","luhn","textrank","nltk_freq"] = "lexrank"
    stage1_sentences: int = 8           # sentence budget per extractive pass
    map_reduce_threshold: int = 400     # total sentences above which map-reduce kicks in
    group_char_budget: int = 20_000     # chars per map group
    max_summary_chars: int = 4000       # Stage-2 length guard (grounding=strict)
    benchmark_out_dir: str = "data/layer_c_bench"
```

---

## 8. Evaluation criteria (how we judge a method)

A Stage-1 method is "good enough" if its extract gives the LLM useful building blocks:
- Good coverage of the dominant themes.
- Representative sentences / claims.
- Minimal redundancy.
- Reasonable coherence.
- Very low compute / zero API cost.

The harness surfaces cost (`execution_time_ms`, char counts) objectively; coverage/redundancy/coherence
are judged qualitatively from the side-by-side JSON. LexRank / LSA / TextRank are the expected
front-runners for the pinned Stage-1 method.

---

## 9. Testing

Mirror `tests/unit/` Layer B conventions (pytest; in-memory stores; mock LLM).

- `test_layer_c_summarizers.py` — each summarizer over plain strings: returns ≤ `n` sentences, handles
  empty / single-sentence / fewer-than-budget input; deterministic where applicable. NLTK-data-dependent
  tests guarded so they skip cleanly if data download is unavailable.
- `test_layer_c_registry.py` — names resolve to the right classes; `all_summarizers` returns all 5;
  unknown name raises.
- `test_layer_c_stage1.py` — single-pass below threshold; map-reduce above threshold (group count, no
  dropped input); `Stage1Output` provenance fields.
- `test_layer_c_stage2.py` — with a mock `LLMClient` (per `test_llm.py` `_OKClient` / `_FailClient`):
  Card fields populated, `cited_chunk_ids` = theme members, grounding guard flags over-length, failure
  path counted.
- `test_layer_c_builder.py` — in-memory stores + mock LLM: themes → cards, skip on empty, failed count
  on LLM error, `dry_run` persists nothing, report counts correct.
- `test_layer_c_config.py` — defaults; `extra="forbid"`; literal validation of `stage1_method`.
- Heavy bits (real sumy/nltk downloads) marked `@pytest.mark.slow`; no real ES/Neo4j/LLM in unit tests.

The project pytest suite is the gate.

---

## 10. Risks & mitigations

- **NLTK data download** at first run (network). Mitigated by `_ensure_nltk_data()` (lazy, cached) and
  slow-marked tests; document the one-time fetch.
- **Extractive O(S²) blowups** on huge themes. Mitigated by map-reduce + `group_char_budget` +
  `map_reduce_threshold`.
- **sumy/nltk version drift** with the existing ML stack. Mitigated by isolating them in the
  `[summarization]` extra and lazy imports; no pinned-EOL deps (gensim avoided).
- **Theme-level (not per-claim) citations** may overstate grounding. Explicitly documented as baseline
  behavior; full grounding is future work.
- **LLM cost still scales with theme count** (one Stage-2 call per theme). Acceptable: Stage 2 sees only
  the compact extract, not the raw chunks — the intended cost reduction.

---

## 11. Future work
- T5 / abstractive baseline behind `BaseSummarizer`.
- Per-claim grounding (cite the chunks behind each retained sentence).
- Shelf-level Cards (`target_type="shelf"`) reusing the same pipeline.
- Promote the harness's qualitative judgments into an automated coverage/redundancy metric.
