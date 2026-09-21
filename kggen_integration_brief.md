# FoodScholar — KGGen Integration Brief

**Status**: Implementation brief. Hand to Claude Code alongside `AGENTS.md`, `BRIEF.md` and the existing repo.
**Source material**: `kggen/graph_code/` — colleague-authored (V. Pitsilou / S. Papadias, WiseFood). ~6k lines across `kggen_extended/`, `retrieval/`, `ner-nel/`, `build_graph/`, `chunking/`, `utils/`.
**Scope of this brief**: three workstreams — (1) the **NER/NEL method upgrade** (§3), (2) the **Layer 0 relation graph** (§4–§5), (3) **corpus chunking** (§6). They are independent after §0.1 and can proceed in any order or in parallel.
**Out of scope** (declared here, specified elsewhere): the hybrid retrieval implementation of `fs.query()`; Layer B pass-2 upgrade to typed relations; the abstracts chunking pipeline. §8 fixes the contracts those will consume so they can land without re-opening this work.
**Decisions already taken** (do not re-litigate): relations are **Layer 0, under the entity graph** — not a fourth layer beside C. The triple extractor runs on **foodscholar's `LLMClient`** — dspy and litellm do not enter the dependency set.

---

> **Note:** `kggen/` is no longer tracked in Git — it is a fork of an external
> project whose licence is not reproduced, so it is kept locally rather than
> redistributed. Paths below refer to that local reference snapshot; ask the
> WiseFood team if you need it.

## 0. Ground truth and conventions

`AGENTS.md` is authoritative for environment, naming, and the test gate. This brief specifies construction only.

- **Environment**: `conda activate foodscholar` (Python 3.11). The `base` env's NumPy is broken and produces misleading errors.
- **The gate**: `pytest tests/unit -q` and `ruff check src tests` must pass before any phase is called done. New behavior gets a unit test, in-memory-first.
- **Config is `extra="forbid"`.** Every knob below is a field on a Pydantic model in `config.py`, documented in `config.example.yaml`. No loose kwargs.
- **Determinism**: the extractor is LLM-driven and therefore *not* bit-reproducible. Everything downstream of extraction (grounding, dedup, persistence) **must** be deterministic given a fixed `ExtendedGraph`. Tests assert that property, not the LLM's output.
- **Nothing in `kggen/` is edited in place.** It is provenance. Code moves into `src/foodscholar/` rewritten to the library's idiom, or into `research/` / `scripts/` as archive. See §2.

### 0.1 Preflight — do this before the first commit

`kggen/` is currently untracked. Two blockers:

1. **Four hardcoded credentials, three distinct keys, in four files** — all now redacted from history, **none of the keys yet rotated**:

   | key | files | as committed |
   |---|---|---|
   | GPUStack API key | `build_graph/extract_triplets_from_chunks.py:64`, `build_graph/aggregate_graphs.py:78` | `os.environ.get(...)` default stripped to `""` |
   | Hugging Face user access token (`hf_…`) | `ner-nel/nel_ner_evaluation.ipynb` (cell 1, `classic_token = ...`) | literal → `<HF_TOKEN_REDACTED>` |
   | OpenRouter API key (`sk-or-v1-…`) | `ner-nel/ner_benchmark_cross_dataset.ipynb` (cell 25, beside the author's own `# ← paste key here`) | literal → `<OPENROUTER_API_KEY_REDACTED>` |

   **Rotate all three.** Redaction keeps them out of git; it does not un-expose them — the notebook ones reached GitHub's servers in a rejected push.

   *How this was found, recorded so the mistake is not repeated:* the first scan looked only for the GPUStack format it already knew about, and the claim "no secret in history" was made on that basis. GitHub push protection caught the other two. The scan that now gates commits (`secret_scan.py`, kept out of the repo) covers HF, OpenRouter, OpenAI, Anthropic, Groq, Google, GitHub, Slack, AWS, private keys, bearer tokens and URL-embedded credentials — with a leading word boundary on every pattern, because the first version matched the "sk-" inside *"the-risk-of…"* in a URL.

   *Status: history rewritten (`filter-branch` over the unpushed range), backup refs and reflog purged, object database swept clean. Rotation is the remaining human step.*
2. ~30 `__pycache__/*.pyc` files, `Sources_Catalogue.ods`, and generated logs (`guides_removed_pages_log.txt`, `.triage_cache.json`) are staged-adjacent. Add a scoped `.gitignore` or strip them.

---

## 1. What this integrates, and why it is worth doing

Two gaps in foodscholar that the colleague's work closes:

**Gap 1 — foodscholar has no relation concept.** A grep for triple/predicate/relation-as-data across `src/foodscholar/` returns nothing. Entities are FoodOn-linked nodes joined to chunks by `(:Chunk)-[:MENTIONS]->(:Entity)`. Layer B's pass 2 (`layer_b/relatedness_graph.py`) is *untyped co-mention*, weighted by inverse document frequency — it knows that olive oil and LDL cholesterol co-occur, never that one *lowers* the other. `kggen_extended` produces exactly the missing edge, and — this is the part worth preserving — **it keeps the passage each edge came from**.

**Gap 2 — `fs.query()` is a stub.** `facade.py:1207` raises `_deferred("query")`; `src/foodscholar/retrieval/` holds only the `Answer` model. `kggen/graph_code/retrieval/retrieval_core.py` is a working, cached, benchmarked hybrid retriever whose middle and heaviest branches (mean-triplet similarity, Personalized PageRank over the entity subgraph) are *only* implementable once Gap 1 is closed. Closing Gap 1 is therefore a prerequisite for the retrieval milestone, not a parallel nicety.

### 1.1 The alignment that makes this cheap

**`passage_id` ≡ `chunk_id`, byte for byte.** Both pipelines read the same corpus CSV shape (`chunk_id, chunk_text, type, chunk_metadata`); `build_graph/extract_triplets_from_chunks.py` sets `passage_id` to the chunk UUID. kggen's "passages" **are** foodscholar `Chunk`s. There is no ID mapping, no join table, no reconciliation step anywhere in this integration. Preserve that invariant — §4.1 makes it a contract.

Three bridges already exist in the library:

| Bridge | Where | State |
|---|---|---|
| kggen's NER/NEL output CSV → `Mention`/`EntityLink` | `corpus/nel_loader.py` | **Done.** Parses `chunk_id, chunk_entities_ner, chunk_uri_nel` — the exact output schema of `run_ner_nel_corpus_gliner2_sapbert.py`. `fs.ingest(corpus, nel_dir=...)` consumes their production run today. |
| SapBERT as a linking encoder | `config.py:LinkerConfig.nel_encoder`, `annotate/nel_index.py:36` | **Done.** `"sapbert"` is already a valid value. |
| SapBERT's CLS pooling | `annotate/embedder.py:131` | **Done.** The exact gotcha their README warns about is already handled. |

### 1.2 The one real semantic mismatch

**kggen entities are raw LLM surface strings. foodscholar entities are FoodOn IDs.**

```
kggen:       ("olive oil", "reduces", "LDL cholesterol")        # free text
foodscholar: Entity(ontology_id="FOODON:03301710", ...)         # grounded
```

Left alone, the repo would hold two disjoint entity universes — the worst available outcome. §4.2 is the fix, and it is the conceptual core of this integration: every triple endpoint is run through the **existing** `fs.linker` to obtain an `ontology_id`, so a relation becomes an edge between the same `Entity` records that Layer A projects and Layer B clusters. This is what converts "the colleague's knowledge graph" into "foodscholar's relation layer".

---

## 2. Disposition of every kggen file

| kggen path | Disposition | Target |
|---|---|---|
| `kggen_extended/models.py` (`ExtendedGraph`) | **Rewrite, demoted** | `relations/extracted.py` — extractor-internal type only. Does **not** become a library data contract; see §4.1.1. |
| `kggen_extended/steps/_1_get_entities.py`, `_2_get_relations.py` | **Port** | `relations/extract.py`, on `LLMClient.generate_json` (§5.4) |
| `kggen_extended/prompts/*.txt` | **Move verbatim** | `relations/prompts/` — the prompts are good; they are the benchmarked artifact |
| `kggen_extended/utils/deduplicate.py` (semhash) | **Port** | `relations/dedupe.py` (§5.5) |
| `kggen_extended/utils/llm_deduplicate.py` (KMeans + LM) | **Defer** | Not in phase 1. `semantic_consolidation/` already does LLM-judged merging for Layer A; revisit for reuse rather than porting a second implementation. |
| `kggen_extended/kg_gen_extended.py` (orchestrator) | **Replace** | `relations/builder.py` + `fs.build_relations()` (§5.7). Its `retrieve*` methods are dead code superseded by §8.1. |
| `kggen_extended/export.py`, `utils/visualize_kg.py` | **Drop** | `io/graphml.py` and `viz/` already cover this; `VizEdge.kind` is free-form so typed relations render with no viz change |
| `ner-nel/run_ner_nel_corpus_gliner2_sapbert.py` | **Port the defaults, not the script** | `annotate/gliner2_ner.py` + config (§3) |
| `ner-nel/*.ipynb` (2 benchmark notebooks) | **Archive** | `research/ner_nel_bakeoff/` — provenance for the selection in §3.1 |
| `ner-nel/run_scifoodner_inference.py` | **Archive** | `research/ner_nel_bakeoff/` — needs its own conda env, benchmark-only |
| `retrieval/retrieval_core.py`, `run_retrieval.py` | **Out of scope, contract fixed** | §8.1 states what this brief must deliver for it |
| `build_graph/*.py`, `*.sh` | **Drop** | foodscholar's CLI + phase model replaces the drivers, model catalogue and resume logic |
| `chunking/chunking_pipeline_{guides,textbooks}_overlap.ipynb` | **Port** | `corpus/chunker.py` — shared window + the docling producer (§6.4). The library's missing input producer. |
| `chunking/chunking_abstracts_split_check.ipynb` §1–6 | **Port (split only)** | same shared window over NLTK sentences — it is the *same algorithm*, not a different one (§6.5) |
| `chunking/chunking_abstracts_split_check.ipynb` §7–11 | **Move as-is** | `scripts/corpus/` — MiniBatchKMeans file partitioning, not chunking (§6.5) |
| `chunking/pdfs.filtered.txt` | **Move, consumed** | `data/` — the chunker reads it. **Covers guides only**; the textbooks' excluded pages are inline Python sets that must be exported to a manifest (§6.4). |
| `utils/pdf_page_triage.py` | **Move as-is** | `scripts/corpus/` — LLM page triage, runs once per document set (§6.5) |
| `utils/guides_and_guidelines.ipynb`, `Sources_Catalogue.ods` | **Leave in `kggen/`** | Source acquisition; belongs to the data pipeline, not the lib |

---

# WORKSTREAM 1 — NER/NEL

## 3. What changes and why

Your colleague ran a cross-dataset benchmark (`ner_benchmark_cross_dataset.ipynb`, `nel_ner_evaluation.ipynb`) over six datasets and picked defaults. foodscholar currently ships the *pre-benchmark* configuration.

### 3.1 The delta

| | foodscholar today | kggen selected | Note |
|---|---|---|---|
| NER model | `urchade/gliner_large_bio-v0.1` | `fastino/gliner2-large-v1` | evidence is **mixed** — §3.1.1 |
| NER threshold | `0.4` | `0.35` | |
| Label set | 13 bare strings (`config.py:_GLINER_DEFAULT_LABELS`) | **27 labels, each with a description** | GLiNER2 consumes `{label: description}`; the descriptions are load-bearing |
| NER API | `GLiNER.inference(...)` | `GLiNER2.batch_extract_entities(...)` | different package (`gliner2`), different call shape |
| Link encoder | `biolord` | `sapbert` | evidence is **mixed** — §3.1.2 |
| Link threshold | `0.70` | `0.70` | already aligned |

### 3.1.1 What the NER benchmarks actually say — read before trusting the table

The `graph_code` README states GLiNER2 has "best semantic F1 and best precision-with-usable-recall on both corpora." The saved notebook outputs are weaker than that, and the two notebooks **disagree**:

| notebook | scale | ground truth | winner |
|---|---|---|---|
| `nel_ner_evaluation.ipynb` cell 15 | **one passage**, 39 entities | human-curated | **GLiNER biomed-v0.1 — F1 0.909** (what foodscholar ships today); GLiNER2 is not in this table |
| `ner_benchmark_cross_dataset.ipynb` cell 33 | 6 datasets | **GPT-4o-mini proxy**, SapBERT cos ≥ 0.85 | **gliner2_custom — F1 0.729** |

Cross-dataset means, sorted by F1 (cell 33 output, verbatim):

| model | Recall | Precision | F1 | entities/Q |
|---|---|---|---|---|
| `gliner2_custom` | 0.709 | **0.610** | **0.729** | 6.06 |
| `gliner_custom_flat` ← *foodscholar today* | **0.806** | 0.435 | 0.630 | 8.45 |
| `gliner_custom_nested` | 0.905 | 0.397 | 0.618 | 10.97 |
| `en_core_sci_lg` | 0.764 | 0.319 | 0.526 | 14.68 |
| `scifoodner_cafeteria` | 0.146 | 0.753 | 0.511 | 1.39 |

So GLiNER2's win is **against an LLM proxy ground truth**, and it wins by **trading recall for precision**: −12% recall, +40% precision, and **~28% fewer entities per passage** (6.06 vs 8.45).

**That is not a free upgrade for foodscholar.** Layer A shelf support counts, `min_chunks_per_shelf`, the Layer B relatedness graph (`tau_strict`, `min_shared_ids`) and `Entity.chunk_count` all key off mention volume. A 28% mention drop could visibly prune the Layer A tree. §3.5's bake-off must measure that, not just link rate.

### 3.1.2 The SapBERT-vs-BioLORD claim does not survive the notebook output

`nel_ner_evaluation.ipynb` links the same ground-truth entity list with every encoder. Side by side (cells 25 and 28):

| entity | SapBERT | BioLORD |
|---|---|---|
| `carrots` | `FOODON_03316751` *carrot (quick frozen)* — **wrong specificity** | `FOODON_00001687` *carrot food product* — better |
| `HbA1c` | NIL — **correct** | `CHEBI_35143` *hemoglobin* — **wrong** |
| `LDL cholesterol` | `CHEBI_39026` @0.799 | `CHEBI_39026` @0.943 — same id, higher confidence |
| `calcium` | `CHEBI_22985` *calcium molecular entity* | `CHEBI_35156` *calcium salt* — both weak (MiniLM found `CDNO_0000015` *dietary calcium*) |
| `almonds` | `FOODON_00003523` @0.937 | `FOODON_00003523` @0.944 |

Mixed, not dominant. The selection criterion was "highest link rate + URI diversity" — but **link rate is not accuracy**: `carrots → carrot (quick frozen)` is a link *and* an error. That is exactly the failure `config.py:LinkBlocklistEntry` already exists to patch (`fish` → `FOODON:00002281`, aquarium fish food). Selecting on link rate re-creates that bug class at scale.

Two further caveats on this notebook: it ran **without CUDA**, so FoodSEM variants C/D were skipped; and the cross-encoder reranking variants are visibly broken (`Italy → HANCESTRO_0307 "Italian"`, `Japan → "Japanese in Tokyo, Japan (1KGP)"`, `DASH diet → "obsolete: ..."`). The selected config has no reranker, which the output supports.

**Consequence for this workstream**: GLiNER2 and SapBERT ship as **configurable options**. Neither becomes a default without foodscholar's own numbers (§3.5). Nothing above says the colleague's selection is wrong for *their* pipeline — it says the evidence does not transfer unexamined to a pipeline whose downstream layers are sensitive to mention volume.

### 3.1.3 The label set

The label-set change is the substantive one. GLiNER-bio's 13 labels are bare nouns; GLiNER2's 27 carry a sentence of description each (`"food additive": "Substances intentionally added to food for preservation, flavouring, ..."`). The benchmark that selected GLiNER2 used those exact strings — **copy them verbatim from `run_ner_nel_corpus_gliner2_sapbert.py:78-106`. Editing them invalidates the selection.**

### 3.2 Module: `annotate/gliner2_ner.py`

A sibling to `annotate/gliner_ner.py`, implementing the same `NER` protocol (`model_id`, `extract`, `extract_batch`). Do not modify `GLinerNER` — both ship, selected by config, so the old path stays available for comparison.

```python
class GLiner2NER:
    """GLiNER2 NER with a described label set. Lazy model load; batched is the fast path."""

    def __init__(self, *, model_id: str = "fastino/gliner2-large-v1",
                 threshold: float = 0.35, labels: dict[str, str],
                 batch_size: int = 16, quantize: bool = True) -> None: ...

    def extract(self, text: str) -> list[Mention]: ...
    def extract_batch(self, texts: list[str]) -> list[list[Mention]]: ...
```

Port these behaviors from the reference script, they are not incidental:

- **Batch-then-fallback**: `batch_extract_entities` in a `try`, per-text `extract_entities` in the `except`, empty list as the last resort. Mirrors `GLinerNER.extract_batch` and guards the same CUDA/OOM hiccups.
- **Per-chunk dedup on `(surface.lower())`, first occurrence wins**, minimum surface length 2. Reference: `run_ner_batch`.
- **Whitespace normalization** of the surface (`" ".join(s.split())`).
- **Quantize on CUDA only** (`GLiNER2.from_pretrained(..., quantize=(device == "cuda"))`).
- **Offsets**: request `include_spans=True`. Where GLiNER2 returns usable char offsets, populate `Mention.start`/`end` from them; fall back to `text.find(surface)` exactly as `GLinerNER._mentions_from_raw` does. Do **not** repeat `nel_loader`'s `0:len(surface)` placeholder — that loader has no offsets available, this path does.

### 3.3 The `EntityType` problem — read before coding

`io/chunk.py:EntityType` is a `Literal` whose values are *deliberately* GLiNER-bio's vocabulary, so the NER→`Mention` bridge is a no-op string copy. GLiNER2's 27 labels do not match it: `dietary supplement`, `dietary pattern`, `biomarker` and `food` line up; `vitamin`, `mineral`, `amino acid`, `lipid`, `enzyme`, `hormone`, `gene`, `genotype`, `microbe`, `symptom`, `organ or tissue`, `physiological process`, `exercise`, `life stage`, `drug`, `chemical`, `food additive` do not.

`GLinerNER` maps unknown labels to `"other"`. Doing that here **throws away the richest signal in the upgrade** — and `EntityType` feeds `facet_for_entity_type` (`layer_a/facet.py`, aliased as `_facet_hint_for_entity_type` in `facade.py`), which sets `Entity.facet_hint`, which is how the non-`foods` facets will eventually get populated. Silently collapsing 23 types to `other` would be a regression disguised as an upgrade.

**Required approach**: extend the `EntityType` literal to the union of both vocabularies, and add an explicit `_GLINER2_TO_ENTITY_TYPE` mapping table where the two overlap semantically under different spellings (e.g. GLiNER2 `disease` → existing `medical condition`). Anything genuinely new becomes a new literal member. Then extend **`ENTITY_TYPE_TO_FACET` in `layer_a/facet.py`** — the declared single source of truth for this mapping, which `build_entities` imports — so the new types route to facets (`vitamin`/`mineral`/`amino acid`/`lipid` → `nutrients`; `disease`/`symptom`/`biomarker` → `health`; `dietary pattern` → `dietary_patterns`). Note that file also holds `PREFIX_TO_FACET`, the OBO-prefix fallback that fires when `entity_type == "other"`; a richer type vocabulary means that noisier fallback fires less often, which is a second, unadvertised win. **This is an additive change to a widely-referenced contract — do it in its own commit with its own test, ahead of the `GLiner2NER` commit.**

### 3.4 Config

```python
class GLiner2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = "fastino/gliner2-large-v1"
    threshold: float = 0.35
    batch_size: int = 16
    quantize: bool = True
    labels: dict[str, str] = Field(default_factory=lambda: dict(_GLINER2_DEFAULT_LABELS))

class AnnotateConfig(BaseModel):
    ner: Literal["gliner", "gliner2"] = "gliner"   # flip default only after §3.5
    gliner: GLinerConfig = Field(default_factory=GLinerConfig)
    gliner2: GLiner2Config = Field(default_factory=GLiner2Config)
    ...
```

`facade._build_ner()` dispatches on `cfg.annotate.ner`. Add `gliner2` to the `[annotate]` extra in `pyproject.toml`.

### 3.5 Defaults are flipped by evidence, not by this brief

Both switches (`ner: gliner → gliner2`, `nel_encoder: biolord → sapbert`) are **staged behind a bake-off on the real corpus**, in `research/ner_nel_bakeoff/`, reusing `evaluation/linker.py`. The colleague's benchmark selected these on *their* corpus with *their* HNSW index; foodscholar builds its own index from the loaded FoodOn (`annotate/nel_index.py`) and — per §3.1.1–§3.1.2 — the published margin is thinner than the README suggests.

Four cells (2 NER × 2 encoder). **Report all of the following, not just the first group** — the upstream metrics can improve while the downstream ones regress, and that is the outcome this bake-off exists to catch:

*Annotation-level*
- mentions per chunk (mean, and the total); link rate; distinct linked URIs
- a **hand-scored precision sample** — 100 random (surface, linked URI) pairs per cell. Link rate alone selected `carrots → carrot (quick frozen)`.

*Downstream — the part the colleague's benchmark could not measure*
- **Layer A**: shelf count, count of shelves clearing `min_chunks_per_shelf`, max tree depth, and the diff of the active shelf set vs the current build. A 28% mention drop (§3.1.1) plausibly prunes shelves.
- **Layer B**: themes discovered per facet, and mean relatedness-graph edge count — Pass 2 keys off shared entity ids.
- **Entity store**: total `Entity` records, and how many carry a `facet_hint` (this should *rise* with the richer GLiNER2 vocabulary, per §3.3 — if it doesn't, the mapping table is wrong).

Flip a default only in a follow-up commit citing these numbers. If a switch wins upstream and loses downstream, **it ships as an option and the default stands** — that is a result, not a failure.

> Note on `nel_encoder`: changing it invalidates the cached HNSW index. `LinkerConfig.nel_index_path` is auto-derived from the encoder name (`foodon_hnsw_sapbert.bin` vs `..._biolord.bin`), so the two coexist on disk — but the first `sapbert` run pays a full index build. Say so in the bake-off notebook so nobody thinks it hung.

---

# WORKSTREAM 2 — LAYER 0, THE RELATION GRAPH

## 4. What Layer 0 is

**Layer 0 is a typed, corpus-grounded, ontology-anchored edge set over the entities that already exist in `fs.entity_store`.** It sits *under* Layer A, not beside Layer C:

```
Layer C   cards          (cited write-ups)
Layer B   themes         (per-shelf communities)
Layer A   shelves        (FoodOn backbone)
──────────────────────────────────────────────
Layer 0   relations      (:Entity)-[:RELATED {predicate}]->(:Entity)   ← this brief
          entities       (:Entity)  ← fs.build_entities(), exists
          chunks         (:Chunk)   ← fs.ingest(), exists
```

It is built by a new phase `fs.build_relations()` that runs **after `fs.build_entities()`** and **before or independently of `build_layer_a`** — it has no dependency on shelves or themes, and they have none on it. Consumers (§8) opt in.

Positioning consequences, all intended:
- No change to the three-layer narrative in `BRIEF.md` / `docs/concepts/architecture.md` beyond one new subsection under the entity graph.
- `fs.build()` does **not** call it in phase 1 (it costs an LLM pass over the whole corpus). It is opt-in until §10's cost numbers are real.
- Layer A/B/C keep passing their tests untouched.

### 4.1 Data contract — `io/relation.py`

Mirrors `io/entity.py` in structure, frozen, `extra="forbid"`.

```python
class Relation(BaseModel):
    """A typed, corpus-grounded edge between two entities.

    One record per distinct (subject_id, predicate, object_id). Endpoints are
    ontology ids when the extracted surface form linked (the common case), and
    a `NIL:<slug>` sentinel otherwise — always paired with the matching
    `*_linked` flag, so consumers never string-sniff the id prefix.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_id: str
    """Deterministic: see §4.4. Stable across runs for the same triple."""

    subject_id: str        # "FOODON:03301710" | "NIL:olive-oil-infusion"
    predicate: str         # normalized surface predicate, e.g. "reduces"
    object_id: str

    subject_surfaces: tuple[str, ...] = ()
    """Every extracted surface form that grounded to `subject_id` for this
    relation — the audit trail for the grounding step."""
    object_surfaces: tuple[str, ...] = ()
    predicate_surfaces: tuple[str, ...] = ()
    """Predicate variants collapsed into `predicate` by dedup (§5.5)."""

    chunk_ids: tuple[ChunkId, ...] = ()
    """PROVENANCE — the passages this relation was extracted from. Capped at
    RELATION_CHUNK_SAMPLE_CAP; `chunk_count` carries the true total. This is
    the field the whole integration exists to preserve."""
    chunk_count: int = 0
    mention_count: int = 0
    """Times the triple was extracted before dedup. A crude confidence proxy."""

    subject_linked: bool = True
    object_linked: bool = True
    """False when the endpoint is NIL. Lets consumers filter to the fully
    grounded subgraph in one predicate without string-sniffing the ids."""

    extractor_version: str
    """'kggen-relations-v1(<llm model_id>;<prompt_version>)'."""
    dedupe_version: str
    last_seen: datetime = Field(default_factory=_utcnow)


RELATION_CHUNK_SAMPLE_CAP = 50   # mirrors ENTITY_CHUNK_SAMPLE_CAP
```

**Invariant (test this):** every id in `Relation.chunk_ids` resolves via `chunk_store.get(...)`. This is the `passage_id ≡ chunk_id` alignment of §1.1, made enforceable.

#### 4.1.1 Why `ExtendedGraph` does not become a library model

It is tempting to adopt `kggen_extended/models.py` wholesale. Do not. Its `_sync_relations_with_entities_and_edges` model-validator is ~150 lines of **stale-provenance repair** — loose normalization, `SequenceMatcher` predicate fuzzy-matching, alias-cluster rewriting — that exists because saved chunk-graph JSONs drifted out of sync with their dedup clusters *on disk*. That is a file-format migration problem inherited from their savepoint/resume design. foodscholar has stores, not savepoint files, and `RelationStore` is authoritative. Importing that validator would import a class of bug the library does not have.

`ExtendedGraph` is kept, rewritten and slimmed, as `relations/extracted.py` — the **extractor's** working type, in memory, for one batch at a time. It never persists and never appears in a public signature.

### 4.2 The grounding step — the core of the integration

Per §1.2. In `relations/ground.py`:

```python
def ground_endpoints(
    graph: ExtractedGraph,
    *,
    linker: Linker,
    min_sim: float,      # cfg.relations.grounding.min_sim
) -> GroundingResult:
    """Map every entity surface in `graph` to an ontology_id via `linker`.

    One `link_many` call over the deduplicated surface set (NOT per triple —
    the same surface recurs across hundreds of triples). Returns the surface →
    ontology_id map plus the NIL set."""
```

Rules:

1. **Reuse `fs.linker`.** It is `HNSWLinker` over the same FoodOn index that produced every `EntityLink` in the corpus. Using anything else would re-introduce the two-universe problem one level down.
2. **Batch over the distinct surface set.** `HNSWLinker.link_many` does one encode + one kNN for the batch. A 4.6k-passage corpus yields tens of thousands of triples over a few thousand distinct surfaces — the difference is two orders of magnitude.
3. **Construct a `Mention` with `entity_type="food"`** for each surface, exactly as `HNSWLinker.dry_run` does, so the linker's semantic-type gate does not filter the candidate out.
4. **NIL endpoints are kept, not dropped.** `NIL:<slug(surface)>`, `subject_linked=False`. Dropping them would silently delete every relation touching a concept FoodOn lacks — which, for a nutrition corpus, includes most biomarkers, hormones and physiological processes. Consumers filter on `subject_linked`/`object_linked`; the data layer stays faithful. (Same principle as `Shelf.status="absent"` in Layer A.)
5. **A distinct `min_sim` from linking mentions.** `cfg.relations.grounding.min_sim` defaults to the same `0.70`, but it is a separate knob: an LLM-emitted entity string is a different distribution from a GLiNER span, and you will want to tune it without disturbing `fs.annotate()`.
6. **Emit a grounding report** — link rate, NIL count, top-20 NIL surfaces by frequency — through `ArtifactMeta` and the logger. The top NIL surfaces are the highest-value diagnostic this phase produces: they tell you what your ontology is missing.

### 4.3 `RelationStore` protocol

New protocol in `storage/protocols.py`, alongside `EntityStore`:

```python
@runtime_checkable
class RelationStore(Protocol):
    def init(self) -> None: ...
    def upsert(self, relations: Iterable[Relation]) -> None: ...
    def get(self, relation_id: str) -> Relation | None: ...
    def get_many(self, relation_ids: list[str]) -> list[Relation]: ...

    def for_entity(
        self, ontology_id: str, *, direction: Literal["out", "in", "both"] = "both",
        k: int = 100,
    ) -> list[Relation]:
        """Relations with `ontology_id` as subject / object / either."""

    def for_chunks(self, chunk_ids: list[ChunkId]) -> list[Relation]:
        """Every relation extracted from any of these chunks. The hot path for
        retrieval's triplet-similarity branch (§8.1) — implementations must do
        this in ONE round-trip (ES `terms` on chunk_ids)."""

    def by_predicate(self, predicate: str, *, k: int = 100) -> list[Relation]: ...
    def scan(self) -> list[Relation]: ...
    def iter_relations(self, batch_size: int = 1000) -> Iterable[list[Relation]]: ...

    def clear(self) -> None:
        """Drop every relation. Called at the start of build_relations() so a
        re-run with a changed extractor/config leaves no ghosts. Mirrors
        GraphStore.clear_layer_a / clear_themes."""
```

Three adapters, matching the existing pattern exactly:

- `storage/memory.py` → `InMemoryRelationStore` (dicts + a `chunk_id → relation_ids` inverted index for `for_chunks`)
- `storage/elastic_relations.py` → `ElasticRelationStore`, sibling of `elastic_entities.py`, own file, index `foodscholar_relations`:

  | field | type | note |
  |---|---|---|
  | `relation_id` | keyword | also the ES `_id` |
  | `subject_id`, `object_id`, `predicate` | keyword | + `predicate.text` subfield for BM25 |
  | `subject_surfaces`, `object_surfaces` | text | multivalue |
  | `chunk_ids` | keyword[] | **the `terms` filter behind `for_chunks`** |
  | `subject_linked`, `object_linked` | boolean | |
  | `chunk_count`, `mention_count` | integer | |
  | `last_seen` | date | |

  No `dense_vector` here. Relation embeddings for the triplet-similarity branch are a retrieval-side concern (§8.1) — and per the ES 9.4 `_source` behavior already documented in `AGENTS.md`, adding one now buys a read-back problem for no current consumer.

- `storage/neo4j.py` → `(:Entity)-[:RELATED]->(:Entity)` with `predicate`, `relation_id`, `chunk_count`, `mention_count` on the edge, plus `upsert_relations()` and `clear_relations()` on `GraphStore`. Constraint: `relation_id` unique. **`MERGE` on `(subject_id, predicate, object_id)`** so re-runs are idempotent — the existing `upsert_entities` batching pattern applies directly.

  NIL endpoints become `(:Entity {ontology_id: "NIL:...", prefix: "NIL"})` nodes so the graph stays traversable. Filter them in Cypher with `WHERE NOT e.ontology_id STARTS WITH 'NIL:'`.

### 4.4 `relation_id` scheme

Deterministic, content-addressed, stable across runs:

```
rel:<sha1(subject_id + "\x1f" + predicate + "\x1f" + object_id)[:16]>
```

Not a UUID, not an incrementing counter — a re-run over an unchanged corpus must produce identical ids so `upsert` is a true upsert and cross-store parity audits work. Same spirit as Layer B's theme-id slugs (`layer_b_construction_brief.md` §8.3).

---

## 5. Construction

### 5.1 Module layout

```
src/foodscholar/
  io/relation.py              Relation contract + RELATION_CHUNK_SAMPLE_CAP
  relations/
    __init__.py               build_relations() entry
    extracted.py              ExtractedGraph — extractor-internal, slimmed ExtendedGraph
    extract.py                steps 1+2 on LLMClient.generate_json
    prompts/
      entities.txt            verbatim from kggen_extended/prompts/
      relations.txt           verbatim
      __init__.py             loader + PROMPT_VERSION
    ground.py                 surface -> ontology_id via fs.linker (§4.2)
    dedupe.py                 semhash entity/predicate dedup, provenance-preserving
    persist.py                writes to RelationStore + GraphStore
    builder.py                orchestration
  storage/elastic_relations.py
```

### 5.2 Pipeline

```
chunk_store.iter_chunks(batch_size)
   │
   ├─▶ extract.py      per chunk: entities → relations (2 LLM calls)      §5.4
   │                   emits ExtractedGraph with triplet→{chunk_id}
   ├─▶ dedupe.py       semhash over surfaces + predicates, provenance kept §5.5
   ├─▶ ground.py       surfaces → ontology_id via fs.linker (batched)      §4.2
   ├─▶ (aggregate)     collapse to one Relation per (subj, pred, obj)      §5.6
   └─▶ persist.py      RelationStore.upsert + GraphStore.upsert_relations  §4.3
```

**Dedup before grounding**, deliberately: dedup shrinks the distinct surface set the linker must encode, and grounding then collapses further (two distinct surfaces often share an `ontology_id`). Reversing the order does more encoder work for the same result.

### 5.3 Batching and resume

The reference implementation resumes by writing one JSON savepoint per chunk (`chunk_graphs/<chunk_id>.json`). **Do not port that.** The store is the savepoint:

- Process `cfg.relations.batch_size` chunks per LLM round, upsert each batch's relations before starting the next.
- On restart, skip chunks already covered — query `RelationStore` for the set of `chunk_ids` seen, or (cheaper, preferred) record a `relations_version` stamp on the chunk via the existing per-chunk update path.
- `fs.build_relations(chunk_ids=[...])` for targeted re-runs; `force=True` to ignore the resume set.

### 5.4 The extractor on `LLMClient`

The port is narrow because their steps are already structured-output calls wearing a dspy costume. `_get_entities_litellm` builds a JSON schema, sends system+user messages, and validates the response — which is precisely `LLMClient.generate_json(prompt, schema)`.

```python
_ENTITIES_SCHEMA = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
    "required": ["entities"], "additionalProperties": False,
}
def _relations_schema(entities: list[str]) -> dict:
    """Subject and object are constrained to an ENUM of the step-1 entities.

    This is not a detail — see the note below. Mirrors
    `_create_relations_model`, which builds `Literal[tuple(entities)]`.
    """
    entity_enum = {"type": "string", "enum": entities}
    return {
        "type": "object",
        "properties": {"relations": {"type": "array", "items": {
            "type": "object",
            "properties": {"subject": entity_enum,
                           "predicate": {"type": "string"},
                           "object": entity_enum},
            "required": ["subject", "predicate", "object"],
            "additionalProperties": False,
        }}},
        "required": ["relations"], "additionalProperties": False,
    }

def extract_graph(text: str, chunk_id: str, *, llm: LLMClient, cfg) -> ExtractedGraph:
    ents = llm.generate_json(_entities_prompt(text), _ENTITIES_SCHEMA, max_tokens=cfg.max_tokens)
    entities = ents["entities"]
    rels = llm.generate_json(
        _relations_prompt(text, entities),
        _relations_schema(entities),           # per-call, entity-constrained
        max_tokens=cfg.max_tokens,
    )
    triples = _filter_to_entities(rels["relations"], entities)   # belt and braces
    ...
```

**The enum constraint is load-bearing — do not simplify it to `{"type": "string"}`.** `steps/_2_get_relations.py:_create_relations_model` builds a dynamic Pydantic model whose `subject` and `object` are `Literal[tuple(entities)]`, so the provider's structured-output mode forces the model to pick endpoints from the step-1 entity list rather than inventing them. That is why their triples align with their entities. A plain string schema would push the mismatch downstream into grounding (§4.2), where an invented endpoint becomes a NIL node and silently pollutes the graph.

Mirror their two-tier recovery as well (`parse_relations_response`): try strict validation first; on `ValidationError`, fall back to raw JSON and **drop any triple whose subject or object is not in the entity set**. Both tiers are needed — a local model behind an OpenAI-compatible endpoint may ignore the enum entirely.

Two operational caveats:
- **Enum size.** A chunk yielding 40 entities produces a 40-value enum, twice per item. Some providers cap enum length or degrade on large ones; log and fall back to the unconstrained schema plus the filter when the call is rejected.
- `ExtendedGraph`'s validator re-adds missing relation endpoints to `entities` precisely because "some relation extractors can emit subject/object strings that were not returned by the entity extractor" — so even with the enum, expect leakage and keep the filter.

What this buys, and why it was the chosen route:

- **dspy and litellm stay out of the dependency set.** The `[relations]` extra is then just `semhash` (+ its `model2vec`/`numpy` tail).
- **Provider fallback for free** — `FallbackLLMClient` already chains providers. An LLM pass over a full corpus *will* hit a rate limit or a 500; today's reference script dies and resumes from savepoints.
- **`${ENV}` config, no hardcoded keys** — directly retires the §0.1 secret.
- **The Groq lesson applies automatically.** `AGENTS.md` records that `openai/gpt-oss-*` returns empty via `GroqClient.generate` (reasoning models hide content in `reasoning_content`). The reference `MODEL_CATALOG` lists `gpt-oss:20b`. Going through the library's providers means that guardrail covers this path too — worth an explicit note in the docs page.

**Required addition — an OpenAI-compatible provider.** The colleague's production endpoint is **GPUStack** (`http://test6.magellan2.imsi.athenarc.gr/v1`), reached as `openai/<model>` against a custom base URL.

Most of this already exists: `ProviderConfig.host` is present and `llm/factory.py:40-45` already forwards it as `base_url=` for OpenRouter (the field is documented as overloading to "ollama daemon URL / openrouter base_url"). What is missing is only that `OpenAIClient.__init__` does not accept or pass a base URL. So:

- add `"openai_compatible"` to the `LLMProvider` literal;
- add `OpenAICompatibleClient` in `llm/providers.py` — a thin variant of `OpenAIClient` passing `base_url=`, with its key read from a distinct env var so it does not collide with `OPENAI_API_KEY`;
- register it in `llm/factory.py:PROVIDERS` and extend the existing `spec.host` branch to route to it.

No new config field is needed — `host` already carries the base URL. This also covers self-hosted vLLM and Ollama's OpenAI shim; it is generally useful, not a one-off.

**Chunk-size handling.** The reference passes `KG_CHUNK_SIZE = 5000` and splits on sentence boundaries via NLTK when the model reports a context-length error. foodscholar chunks are ≤512 tokens by construction, so this path should be dead for the normal corpus — **do not port `utils/chunk_text.py`**. Instead: catch the context-length error, log `relations.chunk_too_long` with the chunk id, skip. If it fires, the corpus has a problem worth seeing rather than papering over.

> Until Workstream 3 lands, "by construction" means *by a notebook outside this repository*, which the library cannot verify — see §6. If phases 10–11 (§6.2) run first, this assumption becomes a measured number instead of an inherited belief, and the skip-and-log path above becomes a real diagnostic rather than dead code.

### 5.5 Dedup (`relations/dedupe.py`)

Port `utils/deduplicate.py`'s semhash pass at `similarity_threshold=0.95` (their production value). The essential property to preserve — **verified in the source**, `run_semhash_deduplication` unions both `entity_passages` and `triplet_passages` on collapse: **when two surfaces merge, their provenance sets union rather than overwrite.** A triple extracted from three passages that dedup merges with a variant from two more has five `chunk_ids`. Losing that would discard the one thing this integration is for.

**Port the normalization too — it is doing real work and is undocumented in their README:**
1. `unicodedata.normalize("NFKC", text)`
2. **Per-token singularization via `inflect.singular_noun`** — "whole grains" → "whole grain". This is where much of the dedup power comes from, and `inflect` is an unlisted dependency; add it to the `[relations]` extra.
3. Only then semhash.

> **Determinism bug — fix it in the port, do not reproduce it.** `DeduplicateList.deduplicate` iterates `graph.entities`, a `set[str]`, and assigns `items_map[singular] = item` (last write wins). It then calls `SemHash.from_records(list(normalized_items))` on a set as well. Python randomizes string hashing per process (`PYTHONHASHSEED`), so whenever two distinct surfaces share a singular form — "vitamins"/"vitamin" — **the canonical pick varies between runs in different processes**. §10's determinism criterion would pass in-process and fail in CI. The fix is one line: `sorted()` the input before normalizing and before handing records to semhash. Add a regression test that runs the dedup in a subprocess with two different `PYTHONHASHSEED` values and asserts identical output.

Two scopes:
- **Entity surfaces** — `"olive oil"` / `"Olive oil"` / `"olive-oil"` → one canonical.
- **Predicates** — `"reduces"` / `"lowers"` / `"decreases"`. Keep every variant in `predicate_surfaces`; the canonical goes in `predicate`. Predicates are where an open extractor sprawls, and the retrieval PPR branch treats distinct predicates as parallel edges (extra PageRank votes) — so under-merging predicates silently biases the ranking. Worth its own tuning knob: `cfg.relations.dedupe.predicate_threshold`, defaulting to the entity threshold.

`semhash` goes in a new `[relations]` extra, lazy-imported. With it absent, `dedupe` degrades to exact-match normalization and logs a warning — the phase still runs (the in-memory/unit path depends on this).

Skip `llm_deduplicate.py` for now (§2).

### 5.6 Aggregation

Group by `(subject_id, predicate, object_id)` after grounding; union `chunk_ids` (cap at `RELATION_CHUNK_SAMPLE_CAP`, true total in `chunk_count`), sum `mention_count`, union the surface tuples, sort every tuple for determinism. Same shape as `fs.build_entities()`'s aggregation loop — read it first and match its idiom.

### 5.7 Facade phase

```python
def build_relations(
    self, *, chunk_ids: list[ChunkId] | None = None,
    force: bool = False, dry_run: bool = False,
) -> ArtifactMeta:
    """Extract typed relations from chunk text and persist them as Layer 0.

    Requires: chunks in the chunk store, a real LLM (not the mock), and a
    linker. Runs after build_entities(); independent of Layer A/B/C.

    Idempotent: relation_ids are content-addressed (§4.4), so a re-run over an
    unchanged corpus rewrites identical records. `force=True` re-extracts
    chunks already covered.
    """
```

- **Refuse to run under `_MockLLM`** unless `dry_run=True`, with a clear error. `in_memory()` wires a mock LLM whose output is meaningless — the same trap `AGENTS.md` already flags for theme labels and cards. Better a loud failure than a store full of garbage triples.
- `dry_run=True` → extract + ground + report, no writes. The notebook path.
- `fs.relations` read namespace mirroring `fs.entities` (`_EntityView` in `facade.py` is the template): `.for_entity(id)`, `.for_chunks(ids)`, `.by_predicate(p)`, `__len__`.
- CLI: `foodscholar build-relations --config config.yaml [--dry-run]`.
- **Not** added to `fs.build()` in phase 1.

### 5.8 Config

```yaml
relations:
  enabled: false                 # opt-in; build() skips it while false
  batch_size: 16
  max_tokens: 4096
  llm: null                      # null → inherit cfg.llm; else a ProviderConfig
                                 # (extraction wants a cheap local model, while
                                 #  Layer C wants a strong one — hence the override)
  grounding:
    min_sim: 0.70
    keep_nil: true
  dedupe:
    enabled: true
    similarity_threshold: 0.95
    predicate_threshold: 0.95
  store:
    backend: memory              # memory | elastic
    es_index: foodscholar_relations
```

Document every field in `config.example.yaml` with its default, per convention.

---

# WORKSTREAM 3 — CORPUS CHUNKING

## 6. The gap this closes

**foodscholar cannot currently rebuild its own primary input.** There is no chunker anywhere in the repository — not in `src/`, not in `scripts/`, not in `notebooks/`. `fs.ingest(path)` reads *pre-chunked* CSVs through `corpus/csv_reader.py`. The only text-splitting in the library is `layer_c/base.py:split_sentences` (extractive summarization) and the HF tokenizer in `annotate/embedder.py` (truncation for SapBERT pooling); neither produces chunks.

The only chunker in this repository is `kggen/graph_code/chunking/` — three Docling/NLTK notebooks. Two consequences, both verified:

1. **Nothing validates chunk size at ingest.** `csv_reader.py` and `io/chunk.py` check nothing. §5.4 argues against porting `chunk_text.py` on the grounds that foodscholar chunks are ≤512 tokens *by construction* — but that construction happens outside the library, so the library can neither enforce nor verify its own core assumption.
2. **The `source_metadata` schema is defined nowhere in the library.** `file`, `urn`, `country`, `title`, `audience`, `pages`, `heading`, `page_number`, `pdf_name` return **zero grep hits** across `src/`. `Chunk.source_metadata` is a free-form `dict[str, object]`. Retrieval needs `file`/`page_number` for citations; Layer C cards cite chunks. The library depends on a contract it does not state.

Net: every `chunk_id` in Elasticsearch traces to a notebook in a folder this brief otherwise archives, and adding one new guide PDF has no `fs.` path.

### 6.1 Read this first — `chunk_id` is a fresh UUID

Both PDF notebooks assign `chunk_id = str(uuid.uuid4())` at chunking time. There is **no content-addressed id**. Consequences, and they are severe:

- **Re-chunking a document produces entirely new ids.** Every `Relation.chunk_ids`, `Entity.chunk_ids`, `(:Chunk)-[:ATTACHED_TO]->(:Shelf)` edge, theme membership and `Card.cited_chunk_ids` that referenced the old ids is orphaned. Layer A/B/C and Layer 0 all silently lose their provenance.
- **§1.1's `passage_id ≡ chunk_id` invariant holds only because kggen read the same CSV files.** It is an artifact of a shared input file, not a property of the system. Re-chunking breaks it.
- **The §6.6 parity check cannot diff on `chunk_id`** — it must diff on normalized chunk *text*.

Required handling in `corpus/chunker.py`:

1. `chunk_documents()` **refuses by default** to write into a corpus directory whose CSVs are already ingested, and says why. `force=True` overrides, with a log line naming what will be orphaned.
2. Offer `chunk_id_strategy: Literal["uuid4", "content_hash"] = "uuid4"`. `content_hash` = `sha1(source_doc_id + "\x1f" + normalized_text)[:16]`, which makes re-chunking an unchanged document idempotent and turns the parity check into an id comparison. **`uuid4` stays the default** so the existing corpus remains reproducible bit-for-bit; `content_hash` is the recommended setting for any new corpus.
3. Document in `docs/concepts/corpus-input.md` that re-chunking an ingested corpus is a destructive operation requiring a full rebuild.

This is the single most important finding in Workstream 3. It was not visible from the READMEs.

### 6.2 Data contract — `io/chunk.py`

Promote `source_metadata` from a free dict to a stated shape. It must be permissive, because **the three producers emit three different key sets** — verified by reading all three notebooks:

| source_type | keys emitted | citation capability |
|---|---|---|
| `guide` | `file`, `urn`, `pdf_name`, `country`, `title`, `audience`, `pages`, `heading`, `page_number` | full — title + country + page |
| `textbook` | `file`, `heading`, `page_number` **only** | filename + page; **no title, no country** |
| `abstract` | `title`, `year`, paper identifiers (`csv_reader.py:_source_doc_id` probes `DOI`, `doi`, `title`) | bibliographic; **no page number** |

```python
class ChunkProvenance(BaseModel):
    """Stated shape of Chunk.source_metadata. Extra keys are preserved — the
    three producers emit three different key sets, and this documents the
    fields the library READS rather than constraining what a producer writes."""

    model_config = ConfigDict(extra="allow")

    file: str | None = None          # guide/textbook — citation anchor
    page_number: int | None = None   # guide/textbook — FIRST page only, see below
    heading: str | None = None       # guide/textbook
    pdf_name: str | None = None      # guide only
    urn: str | None = None           # guide only
    country: str | None = None       # guide only
    title: str | None = None         # guide + abstract
    audience: str | None = None      # guide only
    pages: int | None = None         # guide only
    doi: str | None = None           # abstract — see the casing trap
    year: int | None = None          # abstract
```

`Chunk.source_metadata` stays `dict[str, object]` on the wire — changing it would break every stored chunk and the parquet snapshots. `ChunkProvenance` is a **reader-side view**: `chunk.provenance` as a cached property. Consumers needing citations use it instead of `.get("page_number")` with a guess.

Two traps that must be handled, not assumed away:

> **Key casing.** `csv_reader.py:_source_doc_id` already probes `("DOI", "doi", "title")` in that order — the corpus contains **both** casings. Normalize on construction; do not pick one and silently drop the other.

> **`page_number` is the chunk's FIRST page, not its span.** Both notebooks write `oc["page_numbers"][0]` while the chunk may span several pages. A citation pointing into a chunk's tail points at the wrong page. Either add `page_numbers: tuple[int, ...]` to the contract and have the chunker emit the full list (a producer change, so new corpora only), or document the imprecision where §8.1 consumes it. **Do not leave it undocumented** — it is the kind of thing that surfaces as a user-visible citation bug.

### 6.3 Ingest-time validation

Add to `corpus/csv_reader.py`, config-driven, **warn by default and never raise**: a corpus that fails validation is still a corpus, and hard-failing `fs.ingest` on a long chunk is worse than a noisy log.

```python
class CorpusValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    max_chunk_tokens: int = 512      # must match ChunkerConfig.max_tokens
    token_estimate: Literal["chars", "tokenizer"] = "chars"
    """`chars` = len(text)/4, zero-dependency, the default. `tokenizer` loads
    the CHUNKER's tokenizer (bge-large, not the embedder's bge-base) for an
    exact count — accurate, but pulls transformers into the ingest path."""
    on_violation: Literal["warn", "raise"] = "warn"
```

Log `corpus.chunk_oversized` with `chunk_id`, estimated tokens and `source_doc_id`; emit a one-line summary at the end of ingest (`n_oversized / n_total`). This is the first time the 512-token assumption becomes observable.

### 6.4 Module: `corpus/chunker.py`

The decisive finding from reading all three notebooks: **the sliding window is one algorithm, shared by all three pipelines.** The abstracts notebook's `build_overlapping_chunk_texts` is the same cumsum / expand-end / advance-for-overlap logic as the PDF notebooks' `build_overlapping_chunks`, with the same 512/64 — reimplemented over NLTK sentences instead of Docling fine-chunks. Only the *fine-unit producer* differs.

So the port is **one window + two producers**, not two pipelines:

```python
# the shared core — source-agnostic, pure, unit-testable with no models
def build_overlapping_chunks(
    fine_units: list[FineUnit],        # .text, .group_key, .page_numbers
    *, max_tokens: int = 512, overlap: int = 64,
    count_tokens: Callable[[str], int],
) -> list[MergedChunk]: ...

# producer A — PDFs (docling), gated by the [chunking] extra
def _fine_units_from_pdf(path, cfg) -> list[FineUnit]: ...
# producer B — plain text (nltk sentences), no docling needed
def _fine_units_from_text(text, cfg) -> list[FineUnit]: ...

def chunk_documents(
    pdf_dir: str | Path, *, out_dir: str | Path, source_type: SourceType,
    cfg: ChunkerConfig, metadata_csv: str | Path | None = None,
    excluded_pages: dict[str, set[int]] | str | Path | None = None,
) -> list[Path]:
    """PDFs -> one corpus CSV per document, in the shape fs.ingest() reads."""
```

Window semantics, stated precisely because the brief's earlier "512/64 sliding window" was too loose to port faithfully:

- Group fine units by `group_key` using **`itertools.groupby`**, which groups only *consecutive* units. A dict-grouping port produces different chunks. `group_key` = top-level heading (`chunk.meta.headings[0]`, `""` when absent) for PDFs; a single constant group for abstracts.
- If everything remaining in a group fits in `max_tokens`, emit it as one chunk and stop — no tail-only chunks.
- Otherwise expand `end` while `cumsum[end+1] - cumsum[start] <= max_tokens`.
- **`overlap` is a minimum tail, not a target**: advance `start` to the *furthest* `k` whose remaining tail `cumsum[end] - cumsum[k]` is still `>= overlap`, then `start = max(new_start, start + 1)` to guarantee progress. Actual overlap is therefore ≥ 64 and frequently more.
- A single oversized fine unit is emitted alone and `start` advances by one.

Values that must be preserved verbatim — they define the existing corpus, and drifting from them makes new chunks incomparable to old ones:

| Stage | Setting | Source |
|---|---|---|
| Docling convert | `do_ocr=False`, `generate_picture_images=False`, accelerator device configurable (both notebooks hardcode `cuda:1`) | guides cell 5 / textbooks cell 4 |
| Fine chunking (PDF) | `HybridChunker`, tokenizer **`BAAI/bge-large-en-v1.5` with `use_fast=False`**, `FINE_MAX_TOKENS=80`, `merge_peers=True`, `repeat_table_header=False`, `always_emit_headings=False`, `omit_header_on_overflow=False`, tables → Markdown via `MarkdownTableSerializer`, images dropped | guides cell 7 / textbooks cell 6 |
| Fine chunking (text) | `nltk.sent_tokenize`, same tokenizer for counting | abstracts cell 7 |
| Window | 512 / 64, per the semantics above | all three |
| Filtering | drop chunks failing `has_meaningful_text` (`^[\W_]+$` = punctuation-only) and chunks touching an excluded page | guides cell 11 / textbooks cell 10 |

> **`use_fast=False` is load-bearing.** The slow and fast bge tokenizers can differ by a token on some inputs; the corpus was counted with the slow one. Pin it.

> **Stale docstring warning.** `process_one_textbook`'s docstring says "HybridChunker (max 256 tokens)" while the code uses `FINE_MAX_TOKENS = 80`. Do not copy the comment.

**Page exclusion differs per source and both mechanisms are needed:**

- **Guides** parse `pdfs.filtered.txt` (`filename: X.pdf | removed_pages: [1, 2, …]`), produced by `utils/pdf_page_triage.py`, matched to the PDF by filename *or* stem.
- **Textbooks** have the excluded pages **hardcoded as six inline Python sets** in notebook cell 0 — hundreds of page numbers, not in any manifest. Porting textbook chunking requires either exporting those sets to a manifest file (recommended — do it once, commit it to `data/`) or accepting an explicit `excluded_pages` dict.

> **Exclusion drops the whole chunk.** `if any(p in excluded_pages for p in oc["page_numbers"]): continue` — a 512-token chunk spanning pages 4–5, where page 4 is a cover, is discarded *entirely*, including the good page-5 content. This is the existing behavior and the port must reproduce it for parity; whether to improve it is a separate decision, recorded in §11.

Metadata assembly is per-source (§6.2 table). `_get_guide_metadata` **raises `KeyError`** when a PDF is absent from `guide_metadata.csv` — so that CSV (in `data/original/guides/`, which this checkout does not have) is a hard required input for guides, and the port should fail with a clear message rather than a bare KeyError.

Config:

```yaml
chunking:
  max_tokens: 512
  overlap: 64
  fine_max_tokens: 80
  tokenizer: BAAI/bge-large-en-v1.5
  use_fast_tokenizer: false
  device: cuda:1          # docling accelerator; "cpu" works, slowly
  do_ocr: false
  chunk_id_strategy: uuid4   # uuid4 | content_hash — see §6.1
  excluded_pages_manifest: null
```

New `[chunking]` extra: `docling`, `docling-core`, `nltk`. Note the reference environment pins **`docling-slim==2.117.0`** rather than `docling` — the slim distribution omits the default ML model deps and is what actually produced the corpus. `docling` is the safe superset; swap to `docling-slim` if install weight matters. Heavy either way (torch + PDF parsing + layout models) — **optional, lazy-imported inside the producer**, exactly like `gliner` and `leidenalg`. The shared window and the text producer must work without it, so the unit suite can exercise the algorithm with a stub tokenizer and no models.

### 6.5 Scope boundaries

**In**: the shared window (§6.4), both fine-unit producers, the provenance contract (§6.2), ingest validation (§6.3), the `chunk_id` guard (§6.1).

**Out, and why** — note this corrects an earlier draft of this brief, which scoped out the abstracts pipeline wholesale on the strength of its README:

- **Abstract *splitting* is IN** — it is the same window over NLTK sentences, and only abstracts over 512 tokens are split at all (`needs_split = token_count > MAX_TOKENS`; `INCLUDE_UNSPLIT = True` passes the rest through unchanged). It falls out of the shared core for free.
- **Abstract *clustering* is OUT.** Notebook §7–11 is a separate stage that takes the already-chunked CSV and partitions it into 20 files with `MiniBatchKMeans` over normalized bge-large embeddings (`random_state=42`, memory-mapped, sha1-signature-validated cache). It is **file organization for disk manageability**, not chunking — it changes which file a chunk lands in, never the chunk. Leave it in `scripts/corpus/`. Its embedding-cache validation (model + row count + dim + sha1 over ordered ids) is good prior art for `data/cache/bge_base_*`, and its final assertion block (every input row in exactly one output file, no dupes, no missing) is worth imitating in the parity check.
- **`pdf_page_triage.py`** — LLM page triage producing the guides manifest. Runs once per document set, cached to `.triage_cache.json`. Goes to `scripts/corpus/` as-is; the chunker *consumes* its manifest but does not run it. If ever promoted, it routes through `LLMClient` like §5.4 — it currently calls Ollama directly.
- **`guides_and_guidelines.ipynb`** — source acquisition via the `wisefood` client. Stays in `kggen/`.

### 6.6 What this does *not* fix

The existing corpus was chunked by the notebooks, not by `corpus/chunker.py`. Porting does not retroactively validate what is already in Elasticsearch. Phase 12 (§7) therefore includes a **parity check**: re-chunk two or three already-ingested documents and compare against their stored chunks.

Because ids are UUIDs (§6.1), **compare on normalized text, not id**: for each document, assert the multiset of `" ".join(text.split())` values matches, and report count, min/mean/max token length, and any text present on one side only. Borrow the abstracts notebook's assertion style — count parity, no duplicates, no missing, no unknown. If the port drifted, that is where it surfaces, rather than six months later when someone adds a PDF and the new chunks behave differently from the old.

## 7. Ordering, and what it costs

| Phase | Work | Gate |
|---|---|---|
| **0** | §0.1 preflight: rotate the key, clean `kggen/`, commit it as provenance | `git status` clean, no secret in history |
| **1** | `EntityType` extension + facet-hint routing (§3.3) | unit tests; existing suite green |
| **2** | `GLiner2NER` + config + `[annotate]` extra | unit test with a stubbed model; `fs.annotate()` unchanged under `ner: gliner` |
| **3** | `research/ner_nel_bakeoff/` — 2×2 on the real corpus (§3.5) | numbers reported; defaults flipped or explicitly not |
| **4** | `io/relation.py` + `RelationStore` protocol + `InMemoryRelationStore` | contract tests, in-memory round-trip |
| **5** | `relations/extract.py` + prompts + `OpenAICompatibleClient` | unit tests on a stub `LLMClient`; **no network in the gate** |
| **6** | `ground.py` + `dedupe.py` + aggregation | determinism test: fixed `ExtractedGraph` → identical `Relation` set over 2 runs |
| **7** | `persist.py`, `ElasticRelationStore`, Neo4j `:RELATED` | integration tests (docker-compose), marked `integration` |
| **8** | `fs.build_relations()` + CLI + `fs.relations` view + docs | end-to-end on `FoodScholar.in_memory()` with a stub LLM |
| **9** | Real corpus run + a `viz` relation view + cost/latency numbers | §10 success criteria |
| **10** | `ChunkProvenance` + `chunk.provenance` reader view (§6.2), all three source shapes | unit tests incl. the `DOI`/`doi` casing trap and textbook-only-3-fields |
| **11** | Ingest validation + `CorpusValidationConfig` (§6.3) | oversized-chunk count reported on the real corpus |
| **12a** | Shared `build_overlapping_chunks` + the text producer (§6.4) — **no docling** | window semantics unit-tested with a stub tokenizer, no models, in the gate |
| **12b** | Docling producer + `[chunking]` extra + `fs.chunk_documents()` + the §6.1 `chunk_id` guard | **parity check** on 2–3 already-ingested documents, diffed on normalized text (§6.6) |
| **12c** | Export the textbooks' inline excluded-page sets to a manifest in `data/` | textbook chunking reproducible outside the notebook |

The three workstreams — phases 1–3 (NER/NEL), 4–9 (relations), 10–12 (chunking) — are independent after phase 0 and can proceed in any order or in parallel. Within chunking, phases 10–11 are cheap and worth doing early regardless: they make the corpus contract observable before anyone relies on it. Phase 12a is the whole algorithm and needs no heavy dependency — do it before 12b.

**Docs are part of each phase, not a phase 13.** `AGENTS.md` documents the Sphinx build with `-W`, so a stale API reference fails the build rather than degrading quietly. Budget the `docs/reference/*` update inside the phase that changes the contract; the concept pages (`architecture.md`, `glossary.md`, `corpus-input.md`, `worked-example.md`) and the two new pages (`concepts/layer-0-relations.md`, `guides/chunking-a-corpus.md`) land with the phase that makes them true. A brief that ships code in twelve phases and documentation in one is how `docs/` ends up with a `DESIGN_agentic_annotate.md` describing a removed feature.

**Cost — unknown, and phase 9 exists to find out.** Two LLM calls per chunk over a ~13–14k-chunk corpus (the `data/cache/bge_base_*` artifacts suggest that scale) is ~27k calls. On a local Ollama/GPUStack model that is hours of GPU time and zero euros; on a hosted API it is a real invoice. **Measure on 200 chunks and extrapolate before running the full corpus**, and report tokens/chunk, seconds/chunk and triples/chunk in the phase-9 notes. The reference `UsageTracker` (a litellm callback that accumulates prompt/completion tokens) is the right idea in the wrong place — implement the equivalent as an optional counter on the builder.

---

## 8. Contracts for the deferred consumers

Not built here. Fixed here so they need no rework.

### 8.1 `fs.query()` — hybrid retrieval

`retrieval_core.py` scores `w_text·textSim + w_triplet·meanTripletSim + w_ppr·PPR` at `0.3/0.3/0.4`. Once Layer 0 exists, each branch has a store-backed source:

| Branch | kggen source | foodscholar source |
|---|---|---|
| text similarity | `EmbeddingBundle.passage_embeddings` (`.npy` cache) | `chunk_store.knn_search_chunks(qvec, k=...)` — BGE + ES HNSW, **already built** |
| mean triplet similarity | `triplet_embeddings` + `passage_to_triples` | `RelationStore.for_chunks(candidate_ids)` + a relation-text embedding cache |
| PPR | `IndexBundle.relation_graph` (MultiDiGraph from GraphML) | relations → `nx.MultiDiGraph`, or Cypher over `:RELATED` |
| passage metadata | `passages.json` / GraphML node attrs | `Chunk.source_metadata` via `chunk.provenance` (§6.2) — **already there** |

Two citation caveats inherited from the chunker, both from §6.2: `page_number` is the chunk's **first** page, not its span, so a citation into a chunk's tail can point one page early; and **textbook chunks carry only `file`/`heading`/`page_number`** — no title, no country — so a citation renderer must degrade per `source_type` rather than assume the guide shape.

`RelationStore.for_chunks` in one round-trip (§4.3) exists precisely for branch 2. Keep distinct predicates as parallel edges in the PPR graph — that is deliberate in the reference (`_build_indexes_from_graph` step 4): multiple predicates between the same pair count as multiple votes. §5.5's predicate-dedup threshold therefore has a direct effect on ranking, which is why it is its own knob.

**The harmonization dividend, stated so it is not lost:** seeding PPR from *all* entities is what kggen can do. foodscholar can seed from **the entities of a shelf or theme** — "answer this within the `oils` shelf" — because Layer A/B partition the same entity set. kggen has no shelf concept and structurally cannot do this. `Answer` already carries `activated_shelves`, `activated_themes` and `cited_cards`. Scoped retrieval is the payoff that justifies binding the two systems rather than running them side by side.

### 8.2 Layer B pass 2

`layer_b/relatedness_graph.py` weights chunk-chunk edges by shared FoodOn ids, `1/log(1+doc_freq)`. With Layer 0, two chunks can instead be related by *participating in the same relation*, or by entities one hop apart in the relation graph. Strictly opt-in behind a `RelatednessConfig` knob, never a default — Layer B's production settings are tuned and audited, and this brief does not disturb them.

### 8.3 Viz

`VizEdge.kind` is a free-form string and `viz/builder.py:entity_neighborhood` already walks entity→chunk edges. A `relation_neighborhood(entity_id, hops=n)` builder needs no model change — edges carry `kind=predicate`, and the pyvis/cytoscape renderers map kind→color today.

---

## 9. Testing

**Unit (`tests/unit/`, the gate — no network, no GPU, no real models):**

- `test_relation_model.py` — `relation_id` determinism; frozen/`extra=forbid`; NIL round-trip.
- `test_relation_store_memory.py` — full `RelationStore` protocol against `InMemoryRelationStore`, including `for_chunks` and `clear`.
- `test_relations_extract.py` — a stub `LLMClient` returning canned JSON; asserts schema handling, malformed-JSON recovery, and that `chunk_id` lands in provenance.
- `test_relations_ground.py` — a stub `Linker`; asserts batching (one `link_many` call per surface set, not per triple), NIL handling, and the `min_sim` gate.
- `test_relations_dedupe.py` — **provenance union on merge** is the assertion that matters; plus graceful degradation with `semhash` absent.
- `test_relations_determinism.py` — fixed `ExtractedGraph` → byte-identical `Relation` set across two runs.
- `test_chunk_window.py` — the shared window (§6.4) with a stub `count_tokens`, **no models**: groupby groups only consecutive units; overlap is a minimum tail; a single oversized unit is emitted alone and advances by one; everything-fits emits one chunk and stops; progress is always made (no infinite loop).
- `test_chunk_provenance.py` — all three source shapes; `DOI`/`doi` normalization; textbook chunks exposing only `file`/`heading`/`page_number`; unknown keys preserved.
- `test_corpus_validation.py` — oversized chunk warns and does not raise; `on_violation="raise"` raises; counts reported.
- `test_chunk_id_guard.py` — `chunk_documents()` refuses to write over an ingested corpus without `force=True`; `content_hash` strategy is stable across two runs on the same text.
- `test_gliner2_ner.py` — stubbed model; batch→per-text fallback, dedup, offset handling, label→`EntityType` mapping.
- `test_entity_type_mapping.py` — every GLiNER2 label maps to a valid `EntityType`; every new type has a facet hint or an explicit `None`.
- `test_build_relations_e2e.py` — `FoodScholar.in_memory()` + stub LLM + stub linker, through `fs.build_relations()` to `fs.relations`.

**Integration (`tests/integration/`, marked, docker-compose):** `ElasticRelationStore` round-trip incl. the `for_chunks` `terms` filter; Neo4j `:RELATED` idempotence under repeated `MERGE`; cross-store parity (every ES relation has a Neo4j edge and vice versa) in the style of `evaluation/audit.py`.

**Research (`research/`, not the gate):** the NER/NEL bake-off (§3.5), and a relation-quality sample — 100 random relations hand-scored for faithfulness to their cited chunk. Extraction quality is the risk this brief cannot design away; only a human read of the output closes it.

---

## 10. Success criteria

**Workstream 1 is done when:** `GLiner2NER` ships behind `annotate.ner: "gliner2"`, the described label set is verbatim from the benchmark, GLiNER2's labels reach `Mention.entity_type` without collapsing to `other`, the 2×2 bake-off numbers are recorded in `research/`, and the defaults are flipped — or explicitly not flipped, with the numbers saying why.

**Workstream 2 is done when:**
1. `fs.build_relations()` runs end-to-end on the real corpus and writes to ES + Neo4j.
2. Every `Relation.chunk_ids` entry resolves in the chunk store — the `passage_id ≡ chunk_id` invariant holds on real data.
3. Relation endpoints are the **same** `ontology_id`s as `fs.entities`; `(:Entity)-[:RELATED]->(:Entity)` traverses to the same nodes `MENTIONS` points at. No second entity universe.
4. Grounding link rate is reported, with the top-20 NIL surfaces — the ontology-gap diagnostic.
5. A re-run over an unchanged corpus produces an identical relation set (determinism below the LLM).
6. Cost and latency per 1k chunks are recorded.
7. `pytest tests/unit` and `ruff check src tests` pass; Layer A/B/C behavior is unchanged.

**Workstream 3 is done when:**
1. `ChunkProvenance` states all three source shapes (§6.2), handles the `DOI`/`doi` casing split, and consumers degrade correctly for textbook chunks that carry only three fields.
2. `fs.ingest` reports an oversized-chunk count on the real corpus — whatever that number is, knowing it is the point.
3. `build_overlapping_chunks` is unit-tested against the window semantics in §6.4 (groupby-consecutive, overlap-as-minimum, single-oversized-unit, fits-in-one-chunk) **with no models loaded**.
4. `fs.chunk_documents(pdf_dir)` produces CSVs that `fs.ingest` reads unmodified, for guides, textbooks and abstracts.
5. The §6.6 parity check on 2–3 already-ingested documents matches on normalized text.
6. Re-chunking an already-ingested corpus is **refused by default** with an explanatory error (§6.1).

The library can then rebuild its own primary input.

---

## 11. Open decisions for the implementer

1. **Should page exclusion still drop the whole chunk?** §6.4: a chunk spanning pages 4–5 is discarded entirely when page 4 is excluded, losing the page-5 content. The port must reproduce this for parity, but it is plausibly costing real corpus. Measure how many chunks are dropped this way during phase 12b, then decide whether to trim the excluded pages' fine units before windowing instead. A change here produces a different corpus, so it is a new-corpus-only decision.
2. **Does `page_numbers` (the full span) join the contract?** §6.2. Adding it fixes citation precision but requires a producer change, so it only benefits corpora chunked after the change. Worth doing at the same time as any `chunk_id_strategy: content_hash` migration, since both are new-corpus-only.
3. **Predicate vocabulary.** Open surface predicates (this brief) or a closed set the LLM must choose from? Open is faithful to the text and is what the benchmarked prompt does; closed makes the graph queryable and PPR better-behaved. Ship open, measure predicate cardinality after the first real run, and revisit — a corpus that yields 8k distinct predicates is telling you to close the set.
4. **Do relations get embeddings, and where?** §8.1's triplet branch needs them. Options: an ES `dense_vector` on the relation index (fights the ES 9.4 `_source` behavior in `AGENTS.md`), a `.npy` cache like `data/cache/bge_base_*` (matches existing practice), or compute-on-demand. **Defer to the retrieval brief** — do not add the field speculatively.
5. **Is `mention_count` a usable confidence signal?** A triple extracted from 12 passages is probably more reliable than one seen once. Worth checking against the phase-9 hand-scored sample before any consumer filters on it.
6. **NIL entity promotion.** Recurrent high-frequency NIL surfaces are candidate ontology gaps. Do they get promoted to first-class `Entity` records with `prefix="NIL"`, or stay relation-local? Phase 1 keeps them relation-local; the grounding report is what makes the question answerable.

---

## Appendix A — kggen → foodscholar name map

| kggen | foodscholar | Note |
|---|---|---|
| `passage_id` | `Chunk.chunk_id` | **identical values**, §1.1 |
| `passages[pid]["text"]` | `Chunk.text` | |
| `passages[pid]["metadata"]` | `Chunk.source_metadata` | same dict, same producer |
| `ExtendedGraph.entities` (strings) | `Entity.ontology_id` after grounding | §4.2 — the semantic bridge |
| `ExtendedGraph.relations` (tuples) | `Relation` | §4.1 |
| `triplet_passages[key]` | `Relation.chunk_ids` | the field this integration exists for |
| `entity_passages[e]` | `Entity.chunk_ids` + `(:Chunk)-[:MENTIONS]->(:Entity)` | already built |
| `entity_clusters` | `Relation.subject_surfaces` / `object_surfaces` | dedup aliases, flattened |
| `edge_clusters` | `Relation.predicate_surfaces` | |
| `DeduplicateMethod.SEMHASH` | `cfg.relations.dedupe` | §5.5 |
| `ExtendedKGGen(model=..., api_base=...)` | `cfg.relations.llm` → `build_llm()` | §5.4 |
| `IndexBundle.relation_graph` | `RelationStore` → `nx.MultiDiGraph` | §8.1 |
| `RetrievalConfig.w_text/w_triplet/w_ppr` | `cfg.retrieval.*` | deferred, §8.1 |
| `Retriever.retrieve()` | `fs.query() -> Answer` | deferred, §8.1 |

## Appendix B — scan coverage and implementation status

**Read in full.** `kggen/graph_code/`: `README.md` + all six sub-READMEs;
`kggen_extended/{__init__,models,kg_gen_extended,export}.py`,
`steps/{_1_get_entities,_2_get_relations,_3_deduplicate}.py`,
`utils/{deduplicate,llm_deduplicate}.py` (llm_deduplicate: structure only),
`prompts/{entities,relations}.txt`;
`ner-nel/run_ner_nel_corpus_gliner2_sapbert.py`;
`build_graph/{extract_triplets_from_chunks,aggregate_graphs}.py`;
all three `chunking/*.ipynb`; both `requirements.txt` files. Saved outputs of
`ner-nel/{nel_ner_evaluation,ner_benchmark_cross_dataset}.ipynb` (§3.1.1–§3.1.2).

**Still not read** — claims touching these remain inference:

| File | Lines | Affects |
|---|---|---|
| `retrieval/retrieval_core.py` | 1133 | ~40% read (config, index build, scoring). Caching/embedding paths unread — §8.1, deferred |
| `retrieval/run_retrieval.py` | 325 | §8.1, deferred |
| `utils/pdf_page_triage.py` | 595 | moved as-is to `scripts/corpus/` |
| `ner-nel/run_scifoodner_inference.py` | 193 | archived; needs its own conda env |
| `kggen_extended/utils/{chunk_text,visualize_kg}.py` | 267 | dropped in §2 |

**Resolved since the first draft:**

- `aggregate_graphs.py` — read. Confirms §5.6: aggregation is `kg.aggregate()`
  (a union-merge of passage-aware fields) then semhash dedup then export. No
  semantics beyond what `relations/aggregate.py` reproduces.
- `export.py` — read. GraphML with passage nodes and `Source` edges, mirroring
  AutoSchemaKG. Those `Source` edges are what the retrieval PPR branch walks to
  reach passages; `Relation.chunk_ids` + `RelationStore.for_chunks` carry the
  same information, so dropping the exporter loses nothing.
- `llm_deduplicate.py` — KMeans + intra-cluster LM dedup, as described.
  Deferring it in favour of reusing `layer_a/semantic_consolidation/` stands.
- **Dependency conflicts: assessed, none found.** The reference env pins
  `numpy==2.5.1`, `pandas==3.0.5`, `torch==2.7.0`, `transformers==4.57.6`,
  `pydantic==2.13.4`, `scikit-learn==1.9.0`, `networkx==3.6.1`,
  `semhash==0.3.3`, `inflect==7.5.0`, `nltk==3.9.1`, `docling-slim==2.117.0`,
  `gliner==0.2.26`, `gliner2==1.3.1`. foodscholar upper-bounds none of these,
  so the extras compose. See the `docling-slim` note in §6.4.

**Missing from the folder entirely:** `src.nel` — the package
`run_ner_nel_corpus_gliner2_sapbert.py` imports `HNSWNELLinker` from. The
production NER/NEL script cannot run as delivered. Also no `data/` directory,
though the READMEs reference `data/README.md`; corpus-scale figures here are
inferred from `data/cache/` filenames, not measured.

**Method note.** An earlier draft of this appendix claimed files as read that
were not. Anything asserted about an unread file above is a disposition
decision, not a finding.

## Appendix C — `kggen_extended` usage, before and after

The migration map. Every "before" snippet is the real call shape from `build_graph/extract_triplets_from_chunks.py` and the `kggen_extended` README.

### C.1 Build the graph over a corpus

**Before** — a shell orchestrator per corpus, a Python driver per CSV, savepoint files for resume, exports to disk:

```bash
cd graph_code/build_graph
./run_all_kggen.sh --model "mistral-small3.2:24b-instruct-2506-q8_0" --skip-existing
# which, per document, runs:
conda run -n python3.12_venv python extract_triplets_from_chunks.py \
    --csv ../data/chunks/guides/chunks_guide_ie-key-messages.csv \
    --out-dir ../data/graph/guides/ie-key-messages \
    --model "mistral-small3.2:24b-instruct-2506-q8_0"
# then, separately:
python aggregate_graphs.py --graphs-dir ../data/graph/guides --out-dir ../data/graph/_aggregated_all
```

```python
# inside the driver — provider routing hardcoded at module level
kg = ExtendedKGGen(
    model="openai/mistral-small-3.2-24b-instruct-2506",   # litellm-prefixed
    api_base=GPUSTACK_API_BASE,                            # module constant
    api_key=GPUSTACK_API_KEY,                              # module constant (!)
    max_tokens=8192, temperature=0.0, disable_cache=True,
)

chunk_graphs = []
for _, row in df.iterrows():
    chunk_id = str(row["chunk_id"])
    ckpt = chunks_dir / f"{chunk_id}.json"
    if chunk_id in processed_ids:                  # resume = one JSON per chunk
        chunk_graphs.append(ExtendedGraph.from_file(str(ckpt)))
        continue
    graph = kg.generate(
        input_data=str(row["chunk_text"]),
        chunk_size=KG_CHUNK_SIZE,                  # 5000
        passage_id=chunk_id,
        passage_text=str(row["chunk_text"]),
        passage_metadata=_parse_metadata(row["chunk_metadata"]),
    )
    graph.to_file(str(ckpt))                       # savepoint
    chunk_graphs.append(graph)

combined = kg.aggregate(chunk_graphs)
deduped  = kg.deduplicate(combined, method=DeduplicateMethod.SEMHASH,
                          semhash_similarity_threshold=0.95)
export_all(deduped, out_dir)   # graphml + triples.csv/jsonl + passages.csv/json + metrics.json
```

**After** — one phase, config-driven, the store is the savepoint:

```python
from foodscholar import FoodScholar

fs = FoodScholar.from_config("config.yaml")
fs.build_relations()                       # extract -> dedupe -> ground -> persist
```

```bash
foodscholar build-relations --config config.yaml
foodscholar build-relations --config config.yaml --dry-run     # report, no writes
```

```yaml
# config.yaml — replaces MODEL_CATALOG, the hardcoded keys and the shell flags
relations:
  enabled: true
  batch_size: 16
  llm:
    provider: openai_compatible
    model: mistral-small-3.2-24b-instruct-2506
    host: ${GPUSTACK_API_BASE}
    api_key: ${GPUSTACK_API_KEY}
  dedupe: { similarity_threshold: 0.95 }
  grounding: { min_sim: 0.70 }
```

What moved, and where it went:

| Before | After | Why |
|---|---|---|
| `--csv` per file + `--graphs-dir` aggregate pass | one phase over `chunk_store` | aggregation is `GROUP BY (subj, pred, obj)` (§5.6), not a second CLI run |
| `chunk_graphs/<id>.json` savepoints | `RelationStore` + a per-chunk version stamp (§5.3) | the store already is durable; savepoint files were a second, driftable source of truth (§4.1.1) |
| `MODEL_CATALOG` + `_resolve_kggen_runtime()` + env vars | `cfg.relations.llm` → `build_llm()` | provider routing, fallback chain and `${ENV}` already exist in the library |
| module-level `GPUSTACK_API_KEY = "gpustack_4a54..."` | `${GPUSTACK_API_KEY}` | §0.1 — this is the secret to rotate |
| `export_all()` → GraphML/CSV/JSON | ES + Neo4j (+ `fs.export_graphml()` if a file is wanted) | the exports existed because there was no store |
| `--skip-existing`, `--dry-run`, `--verbose`, `--log-dir` | `force=`, `dry_run=`, `structlog` | library conventions |

### C.2 Extract from one passage

**Before** — the README's minimal example:

```python
from kggen_extended import ExtendedKGGen, DeduplicateMethod

kg = ExtendedKGGen(model="ollama_chat/mistral-small3.2:24b-instruct-2506-q8_0", temperature=0.0)
graph = kg.extract(text="…", passage_ids=["chunk-1"])
combined = kg.aggregate([graph])
deduped = kg.deduplicate(combined, method=DeduplicateMethod.SEMHASH,
                         semhash_similarity_threshold=0.95)

deduped.relations                      # {("olive oil", "reduces", "LDL cholesterol"), …}
deduped.get_triplet_pids(("olive oil", "reduces", "LDL cholesterol"))   # {"chunk-1"}
```

**After** — same shape, but the endpoints come back grounded:

```python
fs.build_relations(chunk_ids=["chunk-1"], dry_run=True)   # returns ArtifactMeta + report
rels = fs.relations.for_chunks(["chunk-1"])

rels[0].subject_id        # "FOODON:03301710"        <- was the string "olive oil"
rels[0].predicate         # "reduces"
rels[0].object_id         # "CHEBI:47775"            <- was "LDL cholesterol"
rels[0].subject_surfaces  # ("olive oil", "Olive oil")   <- what collapsed into it
rels[0].chunk_ids         # ("chunk-1",)             <- was get_triplet_pids(...)
rels[0].object_linked     # True  (False -> object_id is "NIL:<slug>")
```

The one behavioral difference that matters: **`subject_id`/`object_id` are ontology ids, not surface strings** (§4.2). The surfaces are not lost — they move to `subject_surfaces`/`object_surfaces` — but any code that pattern-matched on entity text must now match on ids, and gains the ability to join against `fs.entities`, Layer A shelves and Layer B themes, which the string form never could.

### C.3 Provenance lookups

| Before | After |
|---|---|
| `graph.get_triplet_pids(("a","rel","b"))` | `Relation.chunk_ids` |
| `graph.iter_triplet_pids()` | `fs.relations.scan()` / `iter_relations()` |
| `graph.triple_has_multiple_passages(t)` | `relation.chunk_count > 1` |
| `graph.entity_passages[e]` | `Entity.chunk_ids` + `(:Chunk)-[:MENTIONS]->(:Entity)` — **already exists** |
| `graph.passages[pid]["text"]` | `fs.chunk_store.get(chunk_id).text` |
| `graph.passages[pid]["metadata"]` | `chunk.provenance` (§6.2) |
| `graph.entity_clusters[canonical]` | `Relation.subject_surfaces` / `object_surfaces` |
| `graph.edge_clusters[canonical]` | `Relation.predicate_surfaces` |
| `graph.stats()` | `fs.info()` + the `relations.done` log line |

### C.4 Downstream use

**Before** — a second, disconnected pipeline reading the exported GraphML:

```python
from retrieval_core import RetrievalConfig, Retriever

retriever = Retriever(RetrievalConfig(
    graph_path=Path("../data/graph/_aggregated_all/aggregated_graph.graphml"),
    passages_path=Path("../data/graph/_aggregated_all/passages.json"),
    cache_dir=Path("../data/cache"), gpu_index=1,
)).warm_up()
result = retriever.retrieve("Is olive oil heart-healthy?")
result.passage_ids, result.score_breakdown
```

**After** — deferred to the retrieval brief (§8.1), but the call site is already fixed by the existing `Answer` contract:

```python
answer = fs.query("Is olive oil heart-healthy?")
answer.cited_chunks        # was result.passage_ids — SAME ids (§1.1)
answer.activated_shelves   # no equivalent before — the Layer A/B dividend
answer.activated_themes
```

```python
# and the scoped query kggen structurally cannot express:
fs.graph.shelf("shelf:oils").query("Is olive oil heart-healthy?")
```

### C.5 What has no "after", by design

| Before | Status |
|---|---|
| `ExtendedKGGen.retrieve()` / `retrieve_relevant_nodes()` / `generate_embeddings()` | **Dropped.** Vestigial single-branch cosine retrieval on the orchestrator, superseded by §8.1. |
| `ExtendedKGGen.cluster()` | **Dropped.** Already `@deprecated` upstream in favour of `deduplicate()`. |
| `export_graphml` / `to_nx_extended` / `visualize_kg` | **Dropped** — but see Appendix B: `export.py` was not read. Confirm before the phase that touches it. |
| `utils/chunk_text.py` | **Dropped.** Chunks are ≤512 tokens — and after §6.2 that is *verified*, not assumed. |
| `utils/llm_deduplicate.py` | **Deferred.** `layer_a/semantic_consolidation/` already does LLM-judged merging; reuse it rather than port a second implementation (§2). |

---

## Appendix D — implementation status

All three workstreams are implemented. Deviations from the brief as written:

| Brief item | Status |
|---|---|
| §3 GLiNER2 + `EntityType` extension | **Done.** 31 members, 27/27 labels resolve, 21 route to a facet. Defaults unchanged. |
| §3.5 bake-off | **Harness written** (`research/ner_nel_bakeoff/`), not yet run — needs the real corpus. |
| §4–§5 Layer 0 | **Done**, including `openai_compatible` provider. |
| §5.4 enum schema | **Done**, with the two-tier recovery and a >120-entity fallback. |
| §5.5 determinism | **Done.** Sorted before normalize and before semhash; regression test runs three `PYTHONHASHSEED` values in subprocesses. |
| §6 chunking | **Done.** Shared window + both producers + guard + validation. |
| §7 phase 12c (textbook page manifest) | **Not done** — the inline sets still need exporting. Noted in `scripts/corpus/README.md`. |
| §8.1 retrieval | **Deferred by design.** `RelationStore.for_chunks` is in place for it. |
| §9 tests | **Done.** 142 new unit tests, 13 integration tests (verified against live ES 9.4.2 + Neo4j). |
| §10 docs | **Done.** 2 new pages, 10 edited, Sphinx `-W` clean. |

Two defects were found *by* the new tests and fixed:

1. `make_relation_id` was separator-injectable — `("A\x1fp", "x", "B")` and
   `("A", "p\x1fx", "B")` hashed identically. Fields are now length-prefixed.
2. The upstream dedup's canonical pick varied with `PYTHONHASHSEED` (it
   iterated a `set` with last-write-wins). Fixed in the port, not reproduced.
