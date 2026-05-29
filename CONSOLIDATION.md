# FoodScholar — Semantic Consolidation Module (Brief)

**Status**: Implementation brief, ready to hand to Claude Code.
**Prerequisites**: Rule-based shelf consolidation already in place (lexical canonicalization + synonym index + canonical IRI lookup).
**Goal**: Catch semantic-duplicate shelves that lexical methods miss, using embedding similarity + LLM verification.

---

## 1. Why this exists

After the rule-based consolidation pass:

- `(raw)`, `(canned)`, ` food product` suffixes are stripped
- EFSA numeric prefixes are stripped
- Variants are clustered via the FoodOn synonym index

That handles ~90–95% of duplicates. The remaining ~5% are pairs that **share semantic meaning but not lexical stem** — and they're invisible to regex.

Examples the rules cannot catch:

- "fermented dairy product" ↔ "yogurt food product"
- "ground cattle meat food product" ↔ "beef (ground)"
- "cured pork belly" ↔ "bacon (raw)" (if FoodOn happened to use both)
- "sucrose-sweetened beverage" ↔ "sugar-sweetened drink"

This module catches them by:

1. **Embedding** every surviving shelf's `label + definition + synonyms` into a vector space.
2. **Finding candidate pairs** within a cosine-distance threshold.
3. **Verifying with an LLM judge** that the pair is a true merge, with the chunk attachments used as context.

The same module is reusable for MONDO (health) and ChEBI (nutrients) in v2 — only the ontology source changes.

---

## 2. Where it fits in the projection pipeline

```
NER + NEL on chunks
   │
   ▼
[ Layer A construction ]
   │
   ├── support counting (ancestor propagation)
   ├── blacklist + threshold + depth cap + single-child collapse
   ├── rule-based consolidation (lexical + synonym index)
   ├── synthetic root injection
   │
   ▼
[ semantic consolidation ]  ← THIS MODULE
   │
   ├── embed each shelf
   ├── candidate generation
   ├── LLM verification
   ├── apply confirmed merges
   │
   ▼
Persist to Neo4j + denormalize shelf_ids to Elastic
```

The module runs **last** in Layer A, before persistence. Output is the final shelf set + an audit log of every merge decision (confirmed and rejected).

---

## 3. Module layout

Add under your existing package:

```
foodscholar/
  projection/
    semantic_consolidation/
      __init__.py          # public API
      models.py            # Pydantic data contracts
      embed.py             # embedding step
      candidates.py        # candidate pair generation
      judge.py             # LLM-as-judge
      apply.py             # merge application
      prompts.py           # judge prompt templates
      config.py            # ConsolidationConfig
      cache.py             # decision cache (by content hash)
    semantic_consolidation_cli.py   # `foodscholar sem-consolidate --config ...`
```

Wire into your existing `foodscholar/projection/pipeline.py` as an optional step controlled by config.

---

## 4. Data contracts (Pydantic)

In `foodscholar/projection/semantic_consolidation/models.py`:

```python
from pydantic import BaseModel, Field
from typing import Literal

ShelfId = str  # matches your existing ShelfId type

class ShelfEmbedding(BaseModel):
    shelf_id: ShelfId
    text: str                  # the concatenated text that got embedded
    embedding: list[float]     # dimension matches embedder
    embedder_id: str           # e.g. "BAAI/bge-large-en-v1.5@2024-10"

class CandidatePair(BaseModel):
    shelf_a: ShelfId
    shelf_b: ShelfId
    cosine_similarity: float
    rule_filtered_reason: str | None = None    # why a rule excluded it, if any
    needs_llm: bool                            # true if rules can't decide

class MergeDecision(BaseModel):
    shelf_a: ShelfId
    shelf_b: ShelfId
    decision: Literal["merge", "keep_separate", "uncertain"]
    confidence: float                          # 0..1
    rationale: str                             # one or two sentences from the LLM
    llm_id: str                                # e.g. "claude-haiku-4-5-20251001"
    prompt_version: str                        # e.g. "v1.0"
    decided_at: str                            # ISO timestamp

class ConsolidationArtifact(BaseModel):
    """The full audit log of one consolidation run."""
    run_id: str
    config_hash: str
    embedder_id: str
    llm_id: str
    candidate_count: int
    confirmed_merges: list[MergeDecision]
    rejected_pairs: list[MergeDecision]
    uncertain_pairs: list[MergeDecision]       # require human review
    started_at: str
    finished_at: str
```

These get persisted alongside the shelf graph with the same config-hash stamping you use elsewhere.

---

## 5. Config additions

Extend your existing projection config:

```yaml
projection:
  facets:
    foods:
      # ... existing rules ...
      semantic_consolidation:
        enabled: true
        embedder:
          model: BAAI/bge-large-en-v1.5
          device: cuda          # or cpu
          batch_size: 64
        candidates:
          cosine_threshold: 0.88        # tunable; see Phase 2
          max_candidates_per_shelf: 5   # cap on pairwise comparisons
          exclude_already_clustered: true
          exclude_subtype_patterns:     # never merge across these word-prefixes
            - Canadian
            - turkey
            - beef
            - imitation
            - red
            - white
            - green
            - silken
            - extra firm
            - soft
            - firm
        judge:
          enabled: true
          model: claude-haiku-4-5
          temperature: 0.0
          batch_size: 10
          prompt_version: v1.0
          auto_merge_confidence: 0.85   # merges below this go to uncertain
        cache:
          enabled: true
          path: ./cache/sem_consolidation.sqlite
```

The `exclude_subtype_patterns` is the key safety net: even if embeddings say `Canadian bacon` and `turkey bacon` are close to `bacon`, they're never merged because one of them starts with a subtype prefix.

---

## 6. Algorithm — three phases

### Phase A — Embed

```python
def embed_shelves(shelves: list[Shelf], embedder, cfg) -> list[ShelfEmbedding]:
    texts = []
    for s in shelves:
        # Concatenate the signal we want the embedder to capture
        parts = [s.label]
        if s.definition:
            parts.append(s.definition)
        parts.extend(s.synonyms[:5])      # cap to avoid label bloat
        texts.append(" | ".join(parts))

    vectors = embedder.encode_batch(texts, batch_size=cfg.batch_size)
    return [
        ShelfEmbedding(shelf_id=s.id, text=t, embedding=v.tolist(),
                       embedder_id=cfg.embedder.model)
        for s, t, v in zip(shelves, texts, vectors)
    ]
```

Use a single sentence-encoder backbone; BGE-large-en-v1.5 is a strong default. Don't average across the text components; concatenate so the model sees label + definition + synonyms together. Cache embeddings by `(shelf_id, embedder_id, text_hash)` so re-runs are cheap.

### Phase B — Candidate generation

```python
def find_candidates(embeddings: list[ShelfEmbedding], cfg) -> list[CandidatePair]:
    # 1) Compute all pairs above cosine threshold via faiss or numpy
    matrix = np.stack([e.embedding for e in embeddings])
    sims = matrix @ matrix.T                      # cosine, since normalized
    np.fill_diagonal(sims, 0.0)

    pairs = []
    for i, j in np.argwhere(sims > cfg.cosine_threshold):
        if i >= j: continue                       # dedup symmetric pairs
        a, b = embeddings[i].shelf_id, embeddings[j].shelf_id
        pairs.append(CandidatePair(shelf_a=a, shelf_b=b,
                                   cosine_similarity=float(sims[i, j]),
                                   needs_llm=True))

    # 2) Apply rule-based filters BEFORE sending to LLM (save cost)
    pairs = [p for p in pairs if not _is_subtype_collision(p, cfg)]
    pairs = [p for p in pairs if not _is_compound_food(p, cfg)]
    pairs = [p for p in pairs if not _already_merged(p)]

    # 3) Cap candidates per shelf to avoid runaway costs
    pairs = _cap_per_shelf(pairs, cfg.max_candidates_per_shelf)

    return pairs
```

`_is_subtype_collision` checks each label for the configured subtype prefixes (`Canadian`, `turkey`, `red`, `silken`, etc.). If either label starts with one and the other doesn't, the pair is excluded — they're parallel siblings, not duplicates.

`_is_compound_food` is the simplest version of the compound-food rule: if one label contains the other as a substring AND the longer label has additional words that aren't qualifiers (`cream cheese` vs `cream`, `tuna salad` vs `tuna`), exclude.

### Phase C — LLM judge

```python
def judge_candidates(pairs: list[CandidatePair],
                     shelves: dict[ShelfId, Shelf],
                     chunks_by_shelf: dict[ShelfId, list[Chunk]],
                     llm: LLMClient, cfg) -> list[MergeDecision]:
    decisions = []
    for batch in _chunks_of(pairs, cfg.judge.batch_size):
        prompts = [build_judge_prompt(p, shelves, chunks_by_shelf) for p in batch]
        responses = llm.batch_complete(prompts, temperature=cfg.judge.temperature)
        for pair, resp in zip(batch, responses):
            decisions.append(parse_decision(pair, resp, cfg))
    return decisions
```

The judge prompt (in `prompts.py`):

```python
JUDGE_PROMPT_V1 = """You are a domain expert helping consolidate a nutrition research knowledge graph.

Two shelves in our graph appear semantically similar. Decide whether they should be merged into a single shelf for navigation, or kept as distinct categories.

Shelf A: "{label_a}"
  definition: {def_a}
  synonyms: {syns_a}
  sample chunks: {samples_a}

Shelf B: "{label_b}"
  definition: {def_b}
  synonyms: {syns_b}
  sample chunks: {samples_b}

Decide:
- MERGE if a nutrition researcher would expect these to be the same browsing destination
- KEEP_SEPARATE if they're distinct concepts that just happen to share words
- UNCERTAIN if the choice depends on context not given

Output JSON only:
{{"decision": "merge"|"keep_separate"|"uncertain",
  "confidence": 0.0-1.0,
  "rationale": "one sentence"}}
"""
```

Pass 2-3 sample chunks per shelf. The chunks ground the decision — labels alone can be misleading.

### Apply

```python
def apply_merges(shelves: list[Shelf],
                 decisions: list[MergeDecision],
                 cfg) -> list[Shelf]:
    confirmed = [d for d in decisions
                 if d.decision == "merge"
                 and d.confidence >= cfg.judge.auto_merge_confidence]

    by_id = {s.id: s for s in shelves}
    for d in confirmed:
        a, b = by_id.get(d.shelf_a), by_id.get(d.shelf_b)
        if not (a and b): continue   # one was already merged in this batch
        canonical = _pick_canonical(a, b)   # prefer the one with cleaner label
        variant = b if canonical is a else a
        canonical.chunk_ids |= variant.chunk_ids
        canonical.see_also.append(variant.iri)
        canonical.support_lifted += variant.support_lifted
        canonical.support_direct += variant.support_direct
        # IMPORTANT: deduplicate chunk_ids before recomputing support
        canonical.support_with_desc = len(canonical.chunk_ids)
        del by_id[variant.id]

    return list(by_id.values())
```

Decisions with `confidence < auto_merge_confidence` get logged as `uncertain` and exported to a CSV for human review. Don't auto-merge on borderline calls.

---

## 7. Implementation order (4 phases, ~2 weeks total)

**Phase 1 — Scaffolding + mock LLM (2 days)**
- Create the module structure, Pydantic models, config additions.
- Write unit tests with a mock LLM (deterministic decisions) and a tiny ontology fixture (~20 classes with known semantic duplicates).
- Wire into the pipeline behind `semantic_consolidation.enabled` flag — defaults to off.

**Phase 2 — Real embedder + candidate generation (3 days)**
- Integrate sentence-transformers / BGE-large.
- Tune `cosine_threshold` on the fixture: should catch known duplicates without false positives.
- Add the rule-based pre-filters (subtype patterns, compound foods, already-merged).
- Print the candidate pair list. Human-review before LLM step.

**Phase 3 — LLM judge + caching (3 days)**
- Implement the judge prompt, batch inference, decision parsing.
- Add the SQLite decision cache (key: hash of label+def+synonyms+samples; value: MergeDecision).
- Build the audit log writer.

**Phase 4 — Integration + production validation (2 days)**
- Wire into your existing pipeline runner.
- Run end-to-end on your 230-shelf foods facet.
- Compare before/after shelf counts and the canonical vocabulary check.
- Audit the `uncertain` decisions by hand; refine the prompt or threshold if needed.

---

## 8. Cost & latency estimates

For your 230-shelf foods facet:

- **Embedding pass**: 230 shelves, BGE-large on CPU: ~10 seconds. On GPU: ~1 second. Embedding cost is negligible.
- **Candidate generation**: O(N²) cosine, ~50k pair comparisons, ~100 ms with numpy. After filters, expect ~50–200 LLM-candidate pairs.
- **LLM judge**: ~200 pairs × ~300 tokens prompt + ~30 tokens response = ~66k tokens total. With Haiku 4.5 at current pricing: under $1 per full run.
- **Total run time**: under 5 minutes end-to-end (excluding any human-review step).

For the full FoodOn (~9k classes) when you bring in MONDO and ChEBI: still tractable, ~$5–15 per consolidation run. Cache reuses across runs.

---

## 9. Reproducibility requirements

Every `ConsolidationArtifact` stamps:

- `config_hash` (deterministic hash of the full config block)
- `embedder_id` (model name + version date)
- `llm_id` (model name + version date)
- `prompt_version` (manual semver; bump on prompt edits)

Two consolidation runs with the same inputs and config must produce identical outputs given the cache. Without the cache, only embedder output is fully deterministic — the LLM at `temperature=0.0` is mostly deterministic but not guaranteed; that's why the cache exists.

When you change the prompt: bump `prompt_version`, which invalidates the cache. When the LLM updates: bump `llm_id`, same effect. The cache key includes both.

---

## 10. Testing strategy

**Unit tests** (`tests/projection/semantic_consolidation/`):
- `test_embed.py`: embedder output dimensions, batching, caching
- `test_candidates.py`: cosine threshold, subtype filter, compound food filter
- `test_judge.py`: prompt construction, response parsing, mock LLM
- `test_apply.py`: merge logic, chunk deduplication, support recomputation

**Integration test** (`tests/projection/test_semantic_pipeline_e2e.py`):
- Fixture: ~30 shelves with 3-4 known semantic duplicates inserted
- Run end-to-end with a mock LLM returning predetermined decisions
- Assert the known duplicates merge, the distinct concepts don't

**Eval set** (`evals/semantic_consolidation_gold.yaml`):
- 50 hand-curated pairs from your real corpus: `merge` / `keep_separate` / `uncertain`
- Run the real LLM judge against these
- Track precision/recall on the merge decisions
- Run after every prompt or model change

Aim for ≥90% precision on `merge` decisions (don't incorrectly merge distinct foods) and ≥75% recall (catch most real duplicates).

---

## 11. Reusability for MONDO and ChEBI

The module takes an abstract input: list of `Shelf` objects with `label`, `definition`, `synonyms`, `chunk_ids`. It doesn't care what ontology those shelves came from.

For v2 with MONDO (health) and ChEBI (nutrients):

1. Load the new ontology via the same `OntologyLoader` interface.
2. Run the same rule-based consolidation (the lexical patterns are mostly ontology-agnostic — `food product` becomes irrelevant for diseases, but the parenthetical / EFSA-prefix patterns may still apply, and adding a few patterns per ontology is cheap).
3. Run this semantic consolidation module unchanged.

Each ontology gets its own subtype-prefix list in config (`disease X` doesn't have `Canadian` / `turkey` / `red` collisions, but might have `acute`, `chronic`, `idiopathic` as subtype indicators).

---

## 12. Open decisions for the implementer

These should be made by the engineer building it, not pre-decided here:

- **Embedder choice**: BGE-large-en-v1.5 is a strong default. For domain specificity, consider PubMedBERT or SciBERT — but they're often weaker general-purpose encoders. Start with BGE; switch if eval suggests it's needed.
- **Cosine threshold default**: 0.88 is a guess. Tune on the fixture in Phase 2 and the eval set in Phase 4. Expect to end up between 0.85 and 0.92.
- **Whether to chain multiple LLM judges**: a single Haiku call per pair is cheap and probably enough. If precision is below target on the eval set, consider running 3 calls per pair and taking majority vote (3x cost, ~2-3% precision improvement typical).
- **Human-review UI**: out of scope for v1. Export `uncertain` decisions to CSV; review in spreadsheet. Build proper UI only if review volume justifies it.
- **When to re-run**: at minimum, every time the ontology updates or the embedder changes. Optionally on a schedule (monthly) to catch decisions that may have drifted as the corpus grew.

---

## 13. What success looks like

After running this module on your current state (230 shelves):

- **5–15 additional merges** beyond the rule-based consolidation (semantic duplicates that lexical methods miss)
- **Zero false-positive merges** when measured against the eval gold set (precision = 1.0 on confirmed)
- **Audit log present and reproducible** — re-running with the cache produces the same `confirmed_merges` list
- **Cleanly drops into the existing pipeline** without touching unrelated code
- **Re-usable** when MONDO / ChEBI come online (no rewrite, just config additions)

Once those hold, Layer A is truly done. Move to Layer B clustering on the busy shelves with the consolidated set as input.

---

## Appendix: snippets you'll need

### Reading definitions/synonyms from pronto

```python
def shelf_inputs_for_embedding(iri: str, ont) -> tuple[str, str, list[str]]:
    term = ont[iri]
    label = term.name or ""
    definition = term.definition or ""
    synonyms = [s.description for s in term.synonyms if s.scope == "EXACT"]
    return label, definition, synonyms
```

### Sentence-transformers embedder wrapper

```python
class BgeEmbedder:
    def __init__(self, model_name: str, device: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.model_id = f"{model_name}@{self.model.last_modified}"

    def encode_batch(self, texts: list[str], batch_size: int = 64):
        return self.model.encode(texts, batch_size=batch_size,
                                  normalize_embeddings=True,
                                  show_progress_bar=False)
```

### LLM client wrapper (Anthropic SDK)

```python
class HaikuJudge:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        from anthropic import Anthropic
        self.client = Anthropic()
        self.model_id = model

    def batch_complete(self, prompts: list[str], temperature: float = 0.0):
        # Sequential for now; switch to async batching if latency matters
        out = []
        for p in prompts:
            resp = self.client.messages.create(
                model=self.model_id,
                max_tokens=256,
                temperature=temperature,
                messages=[{"role": "user", "content": p}],
            )
            out.append(resp.content[0].text)
        return out
```

### Cache key

```python
import hashlib, json

def decision_cache_key(label_a, def_a, syns_a, label_b, def_b, syns_b,
                       llm_id, prompt_version) -> str:
    payload = json.dumps({
        "a": [label_a, def_a, sorted(syns_a)],
        "b": [label_b, def_b, sorted(syns_b)],
        "llm": llm_id,
        "prompt": prompt_version,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
```

---

End of brief. Hand to Claude Code; ~2 weeks engineering work; cost per consolidation run under $1 at current scale.