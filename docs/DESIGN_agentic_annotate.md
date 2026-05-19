# Design — Agentic NER+NEL annotation (M2 redesign)

**Status:** approved · **Date:** 2026-05-18 · last updated 2026-05-18

This document replaces the BERT-model-based annotation pipeline (SciFoodNER +
SapBERT dense tier) with an **agents-first** design: a tool-using LLM agent
does named-entity recognition *and* entity linking in one reasoning loop,
using the existing lexical linker tiers and an adapted
[OntoRAG](https://github.com/jan3657/onto_rag) as **tools**. It supersedes the
M2 annotate design where it conflicts.

## Decision log (resolved 2026-05-18)

The design owner (who also owns BRIEF.md) signed off on the following; the
remaining `[DECIDE]` markers below are minor implementation choices left to
build time.

- **BRIEF §15 deviation — approved.** Making annotation agentic departs from
  §15 ("deterministic only for v1"). The brief owner approved it; recorded as
  a deliberate deviation in BRIEF §3.5. Agentic NER already shipped.
- **`ontorag_resolve` = retrieval-only.** The tool exposes OntoRAG's tri-hybrid
  retriever (Whoosh + FAISS + RRF) returning ranked candidates; *our* agent
  selects. OntoRAG's own LLM selector / scorer / synonym-retry loop are NOT
  nested — that would be an LLM calling an LLM, and the retry loop is the part
  §15 most directly defers.
- **Annotation cache = SQLite.** Indexed point lookups + incremental upserts
  match the cache's access pattern; Parquet (immutable, bulk-scan) does not.
- **First piece shipped:** agentic NER as a standalone stage (`AgenticNER`,
  commit a9d4b4b).

---

## 1. Motivation

The shipped M2 pipeline (iterations 4.0–4.6) reached these limits:

- **KeywordNER** only matches verbatim FoodOn labels — misses compound /
  contextual phrasings ("fortified cereals", "healthy fats").
- **SapBERT dense tier** — measured: links morphological/scientific synonyms
  well (`ascorbate`/`vitamin C` ≈0.76) but *not* opaque abbreviations
  (`EVOO`/`olive oil` ≈0.46). Half the hard cases unsolved.
- **SciFoodNER** — a proprietary fine-tuned model; the project wants to drop
  dependence on bespoke ML models entirely.

Decision (from design discussion): **no SciFoodNER, no proprietary ML models.**
NER becomes agent-driven. The linker keeps its cheap deterministic tiers but
its expensive final tier becomes an adapted OntoRAG.

## 2. Core architecture

### 2.1 The shape — tiers become an agent's tools

The current linker is a fixed cascade `lexical_exact → lexical_fuzzy → dense →
llm`. The redesign **keeps the tier *logic* (cheap-first) but turns the tiers
into tools an agent calls**, rather than a hardcoded `if/elif` chain.

> **[DECIDE] — this resolves a conflict in the design discussion.** "Fused
> NER+NEL agent" and "keep the standalone 3-tier linker" cannot both be literal
> — a single extract-and-link agent has no separate cascade downstream. The
> resolution proposed here: the three tiers survive as the agent's **toolbox**
> (`lexical_exact_lookup`, `lexical_fuzzy_lookup`, `ontorag_resolve`), not as a
> pipeline after it. Cheap-first preference is the agent's instructed strategy,
> not a fixed control flow. Confirm this interpretation before build.

```
┌──────────────────────────────────────────────────────────────────┐
│  AnnotationAgent  (one tool-using LLM agent per chunk)             │
│                                                                    │
│   chunk text ──► agent reasons: "what food/health/nutrient         │
│                   entities are here? for each, which FoodOn id?"   │
│                                                                    │
│   tools available to the agent:                                   │
│     • lexical_exact_lookup(text)   → id | None      [free]         │
│     • lexical_fuzzy_lookup(text)   → id, score | None  [cheap]     │
│     • ontorag_resolve(text, ctx)   → ranked candidates [expensive] │
│     • ontology_neighbors(id)       → parents/children  [free]      │
│                                                                    │
│   output: list[EntityLink]  (mention spans + resolved ids)         │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
        ContentAddressedCache  (the reproducible artifact)
```

The agent extracts a candidate mention, tries `lexical_exact_lookup` first
(free, instant — resolves "olive oil" with zero LLM-tier cost), escalates to
fuzzy, and only calls `ontorag_resolve` for the genuinely hard mentions. It
decides when an answer is good enough. The agent also decides span boundaries
and which mentions are worth keeping — that *is* the NER.

### 2.2 Why this satisfies both stated goals

- *"Agents-first, fused NER+NEL"* — one agent does extraction + linking in a
  single reasoning loop.
- *"Don't abandon the 3-tier linker"* — exact / fuzzy / OntoRAG survive as
  tools; the cheap-first ordering survives as the agent's strategy. The
  deterministic tiers still do the bulk of the work and keep cost down.

### 2.3 OntoRAG as the `ontorag_resolve` tool

OntoRAG (studied from the repo) is itself a pipeline:
**tri-hybrid retrieval** (Whoosh BM25 + MiniLM FAISS + SapBERT FAISS, merged by
Reciprocal Rank Fusion) **→ LLM selector → LLM confidence scorer → synonym-gen
retry loop.**

Adapting it as our `ontorag_resolve` tool:

- **Keep:** tri-hybrid retrieval + RRF candidate merging. This is OntoRAG's
  strongest, most reusable part and strictly better than our single-embedder
  `DenseIndex`.
- **Keep:** the candidate object shape (`id`, `label`, `score`, `fusion_score`,
  `sources`) — maps onto our needs.
- **Drop / change:** OntoRAG's *own* LLM selector + scorer + retry loop. Our
  **agent** is already the LLM doing selection — running OntoRAG's selector
  inside a tool the agent calls would be an LLM calling an LLM. Instead,
  `ontorag_resolve` returns the **ranked candidate list** and the agent selects.
  This also sidesteps adopting OntoRAG's agentic retry loop (BRIEF §15 defers
  agentic multi-step retrieval — see §6).

> **[DECIDE]** Confirm: `ontorag_resolve` is **retrieval only** (tri-hybrid +
> RRF), returning candidates for *our* agent to choose from — we do NOT nest
> OntoRAG's selector/scorer/synonym loop inside it. Alternative: adopt
> OntoRAG's full pipeline as the tool (an LLM-in-a-tool), simpler to vendor but
> a double LLM hop and pulls in the retry loop.

## 3. NER: what the agent extracts

The fused agent replaces `NER` + `Linker` both. The `NER` protocol may be
retired, or kept with an `AgenticNER` implementation that the agent fulfils —
**[DECIDE]**. Recommended: keep the `Mention` and `EntityLink` Pydantic
contracts (they are the lingua franca per BRIEF §4) but the agent produces both
in one pass.

Entity types the agent is prompted to find: foods, nutrients, health concepts
(diseases/conditions), dietary patterns, allergens, population groups,
biomarkers (measurable outcomes), and processing/preparation qualifiers. The
first five align to BRIEF §1 facets; population/biomarker/processing were added
because a nutrition corpus is full of mentions like "children", "glycemic
control", and "fermentation" that the original taxonomy silently dropped. The
agent tags a span, classifies it, and links it — or explicitly rejects it
(e.g. "iron deficiency" → recognized as a *condition*, not a food → no FoodOn
link, which is the correct answer the old lexical linker got wrong). The linker
formalizes this: only food-like types (food, nutrient, dietary_pattern) are
resolved against FoodOn; the rest are kept as `Mention`s but never linked.

## 4. Reproducibility — the content-addressed cache

An agentic pipeline is non-deterministic; BRIEF §13 demands idempotent reruns.
Resolution (decided in discussion): **the cache is the artifact.**

- **Key:** `sha256(chunk_text + agent_model_id + prompt_version + ontology_hash)`.
- **Value:** the full `list[EntityLink]` the agent produced for that chunk.
- **Behavior:** `fs.annotate()` looks up each chunk; cache hit → reuse, zero
  LLM cost; miss → run the agent, store. Re-running annotate over an unchanged
  corpus with an unchanged model/prompt/ontology is therefore a pure cache
  replay — deterministic, free, idempotent.
- **The cache file is a versioned artifact** stamped with an `ArtifactMeta`
  (§13 config-hash). Changing the model, the prompt version, or the ontology
  changes every key → a full honest rebuild.
- **Storage:** Parquet or SQLite under `data/`. **[DECIDE]** which.

This is also what makes agentic annotation *affordable* at BRIEF §16 scale
(~500k–1M chunks): food mentions and whole chunks recur heavily; the cache
amortizes the agent cost dramatically. A secondary **mention-level** cache
(`mention_text → EntityLink`) can short-circuit repeated mentions within the
corpus even across distinct chunks — **[DECIDE]** whether to add that too.

## 5. Cost — the honest number

At ~1M chunks, one agent run per chunk with a tool-using loop is the dominant
cost. A fused tool-using agent may make 3–8 LLM round-trips per *uncached*
chunk (reason → tool call → reason → ...). Mitigations, in order of impact:

1. **The chunk cache** (§4) — the single biggest lever; repeated/unchanged
   chunks cost nothing on rerun.
2. **Cheap tiers as free tools** — `lexical_exact_lookup` resolves common
   mentions with no LLM round-trip inside the loop.
3. **Batching** — many chunks per agent context where the model allows.
4. **A cheap model for NER, escalate only hard mentions** — e.g. Groq Llama-70B
   for the bulk, a stronger model only when `ontorag_resolve` is invoked.
   **[DECIDE]** whether to support per-stage model tiering.

A pilot on a few hundred chunks should produce a real cost-per-1k-chunks figure
before committing to a full corpus run. **[DECIDE]** — pilot first.

## 6. Relationship to BRIEF.md — deviations

This is a substantial, deliberate departure from BRIEF §2. Recorded here for
the §3.5 deviations log:

| BRIEF says | This design | Deviation? |
|---|---|---|
| §2: NER = SciFoodNER | agentic LLM NER | **Yes** — by explicit project decision (drop proprietary ML) |
| §2: linker = lexical → dense (SapBERT) | lexical tiers + OntoRAG, orchestrated by an agent | **Yes** — extends §2; lexical tiers retained |
| §13: deterministic, idempotent reruns | non-deterministic agent; reproducibility via content-addressed cache | **Partial** — idempotency preserved at the *cache* boundary, not the model |
| §15: no agentic / multi-step retrieval in v1 | the annotation pipeline is now agentic | **Yes — the biggest one.** §15 explicitly defers this |

§15 is the load-bearing concern. The brief defers agentic retrieval to v2.
This design makes *annotation* agentic. §13 permits "treat [retrieval] as a v1
— measure, iterate, and document deviations" — but annotation is not
retrieval, and §15 is unambiguous. **[DECIDE] — this needs sign-off from the
brief's owner before build.** Note: OntoRAG's *own* synonym-gen retry loop is
NOT adopted (§2.3), which keeps the deviation to "agentic annotation" and not
"agentic multi-step retrieval with retry".

## 7. Dependencies

New `[ontorag]` extra: `whoosh`, `faiss-cpu`, `sentence-transformers`
(MiniLM + SapBERT for the tri-hybrid retriever). `rdflib` not needed — we
already load FoodOn via `pronto`. The agent itself uses the existing
`foodscholar.llm` provider layer (Groq / Ollama / etc.) — no new LLM deps.

Core install stays slim; `[ontorag]` is opt-in like `[annotate]` / `[llm]`.

## 8. Proposed module layout

```
src/foodscholar/annotate/
  agent.py          # AnnotationAgent — the tool-using NER+NEL agent
  tools.py          # tool fns: lexical_exact_lookup, fuzzy, ontorag_resolve, ...
  cache.py          # ContentAddressedCache (the reproducible artifact)
  ontorag/
    retriever.py    # adapted tri-hybrid retriever (Whoosh + FAISS×2 + RRF)
    index.py        # build/load Whoosh + FAISS indexes from FoodOnAPI
  runner.py         # (exists) — rewired: agent + cache instead of NER+linker
  linker.py         # (exists) — lexical tiers kept, exposed as tools
```

What is **retired**: `embedder.py`'s `SapBERTEmbedder` standalone dense-tier
use (SapBERT moves inside the OntoRAG retriever), the `dense` and `llm` linker
tiers as separate `ThreeTierLinker` stages, and likely `KeywordNER` /
`SciFoodNERAdapter`. The `dense_index.py` `DenseIndex` may be superseded by the
FAISS retriever — **[DECIDE]** keep as the no-FAISS fallback, or remove.

## 9. Open decisions — summary

1. **§2.1** — tiers-as-tools interpretation (vs literal separate linker).
2. **§2.3** — `ontorag_resolve` = retrieval-only (vs OntoRAG's full pipeline nested).
3. **§3** — retire the `NER` protocol, or keep it with an `AgenticNER` impl.
4. **§4** — cache store: Parquet vs SQLite; add a mention-level cache too?
5. **§5** — per-stage model tiering (cheap NER, strong escalation)?
6. **§5** — pilot on a few hundred chunks before any full run?
7. **§6** — **sign-off on the BRIEF §15 deviation (agentic annotation).**
8. **§8** — keep `DenseIndex` as a no-FAISS fallback, or remove it?

## 10. Suggested build order (once decisions are locked)

1. `ontorag/` — adapted tri-hybrid retriever + index builder. Pure, testable,
   no agent yet. Unit-test against the mini FoodOn fixture.
2. `cache.py` — content-addressed cache. Pure, trivially testable.
3. `tools.py` — wrap lexical tiers + the retriever as agent tools.
4. `agent.py` — the tool-using agent loop over `foodscholar.llm`.
5. Rewire `runner.py`; update facade `fs.annotate()`.
6. Notebook + BRIEF §3.5 deviation note + PROGRESS + README.
7. Pilot run → real cost numbers → decide on full-corpus run.
```
