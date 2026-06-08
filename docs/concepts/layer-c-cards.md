# Layer C — Cards

Layer C writes the prose a reader actually sees: a short summary **card** for each Layer B
theme. Where Layers A and B are structure, Layer C is the human-facing explanation — produced
by a deliberately **cheap two-stage pipeline** and stored as a **vector-searchable** artifact.

## The cost problem, and the two-stage answer

A single theme can hold **hundreds** of chunks (tens of thousands of words). Sending all of
that to an LLM, per theme, across a whole facet, is expensive. Layer C's core idea is to do
the bulk compression with a **near-zero-cost extractive** method first, and only hand the LLM
a small, already-distilled extract.

```{mermaid}
flowchart LR
    T["theme<br/>(N chunks, ~150 KB)"] --> S1["Stage 1 — extractive<br/>(LexRank, free)"]
    S1 --> X["extract<br/>(~1–2 KB)"]
    X --> S2["Stage 2 — LLM refine<br/>(Llama 8B)"]
    S2 --> CARD["Card<br/>(title, summary, …)"]
    CARD --> EMB["embed title+summary"]
    EMB --> ES[("cards index<br/>dense_vector")]
    CARD --> NEO[("Neo4j<br/>(:Card) node")]
```

The LLM sees only the **extract**, never the raw chunks — that is the whole cost saving.

(stage-1-extractive)=
## Stage 1 — cheap extractive compression

Stage 1 reduces a theme's chunks to a compact extract using a classical, **zero-API-cost**
summarizer chosen by `config.layer_c.stage1_method`. Five are available behind one
`BaseSummarizer` interface:

```{list-table}
:header-rows: 1

* - Method
  - Backend
  - Notes
* - `lexrank` *(default)*
  - sumy
  - Graph-centrality sentence ranking. **Won the benchmark** — cleanest prose.
* - `lsa`
  - sumy
  - Latent-semantic ranking. The other front-runner.
* - `luhn`
  - sumy
  - Classic word-frequency significance.
* - `textrank`
  - sumy
  - PageRank over a sentence-similarity graph.
* - `nltk_freq`
  - nltk
  - Hand-rolled normalized word-frequency scorer.
```

### Map-reduce for big themes

Extractive methods like LexRank build an `S × S` sentence-similarity matrix — `O(S²)`. A
theme with hundreds of chunks → thousands of sentences → an expensive matrix. So Stage 1
**escalates to map-reduce** once the input exceeds `map_reduce_threshold` sentences (default
400): partition the chunks into character-budgeted groups, summarize each, concatenate the
group summaries, and summarize **once more**. Nothing is dropped.

```{mermaid}
flowchart TD
    C["theme chunks"] --> Q{"> map_reduce_threshold<br/>sentences?"}
    Q -->|no| ONE["single pass → extract"]
    Q -->|yes| MAP["MAP: group by char budget<br/>→ summarize each group"]
    MAP --> GS["group summaries"]
    GS --> RED["REDUCE: summarize the<br/>concatenated summaries"]
    RED --> EX["extract"]
    ONE --> EX
```

```{admonition} Tables defeat naïve sentence splitting
:class: warning
On a real corpus full of markdown nutrition **tables** (`| whole | 150 | 275 |`), a naïve
"split on `.!?`" treats an entire table as one giant *sentence* — so a "top-8 sentences"
extract becomes a 17 KB table wall. Layer C's splitter strips pipe-delimited table runs, drops
non-prose fragments, and clamps over-long pseudo-sentences before any method runs. This was
found by the benchmark (below) and is why LexRank/LSA — which rank prose above tables — looked
clean while the others didn't until the fix.
```

(stage-2-llm)=
## Stage 2 — LLM refinement into a Card

The extract (plus the theme's label and keyword terms for context) goes to the LLM
(`config.layer_c.llm_model`, default **`llama-3.1-8b-instant`** on Groq), which reorganizes it
into the `Card` fields: a `title`, a flowing `summary` (key messages / claims / insights as
prose), a `tip`, and an `evidence_quality` grade. One LLM call per theme.

```{admonition} Grounding in this baseline
:class: note
`config.layer_c.grounding_check` is honored as a **lightweight guard** here: `strict` (default)
requires the summary to be non-empty and within `max_summary_chars` (default 4000), else the
theme is flagged/skipped. Because the LLM sees the *extract* and not individual chunks, a card's
`cited_chunk_ids` carries **theme-level** provenance (the chunks that fed Stage 1), not
per-sentence citations. Per-claim grounding is noted as future work.
```

```python
class Card(BaseModel):
    card_id: CardId
    target_id: str                  # the theme_id this card describes
    target_type: Literal["shelf", "theme"]   # Layer C builds "theme" cards
    title: str
    summary: str
    tip: str | None
    evidence_quality: EvidenceQuality      # high | medium | low | debated | unclear
    controversy_note: str | None           # when sources disagree
    confidence_note: str | None
    cited_chunk_ids: list[str]              # theme-level provenance
    llm_model: str                          # e.g. "llama-3.1-8b-instant"
    prompt_version: str
    safety_flagged: bool                    # facet ∈ safety_sensitive_facets
    generated_at: datetime
    embedding: list[float] | None           # title+summary vector (Stage 3)
    embedding_model: str | None
```

A populated card (the `mammalian milk product` shelf's theme), from an 8B run:

> **Milk and Dairy Nutrition Basics** — *evidence_quality: high*
> Fortified plant-based beverages such as soy milk can match milk's calcium, vitamin D and
> B12. About 30% of calcium is absorbed from dairy; people with limited sun exposure may need
> extra vitamin D. Lactose-intolerant readers can choose low-lactose or plant-based
> alternatives rich in calcium.
> *cited_chunk_ids:* 180 chunks

## Honest about evidence

A card carries metadata that keeps it honest rather than authoritative:

- **`evidence_quality`** grades the overall strength of the material: `high` → `medium` →
  `low`, plus `debated` (sources genuinely conflict) and `unclear` (too thin to judge).
- **`controversy_note`** is the *specifics* of any disagreement; `confidence_note` flags thin
  coverage.
- **`safety_flagged`** is set when the theme's facet is in
  `config.layer_c.safety_sensitive_facets` (e.g. `allergies`), so such cards can be reviewed
  or withheld.

(stage-3-embeddings)=
## Stage 3 — embed + vector-search

Every card is **embedded** (`title + summary`, via the same BGE embedder used for chunks) and
written to **two** stores: the **Neo4j** `(:Card)` node (as before) and a dedicated Elastic
**`foodscholar_cards`** index (`dense_vector`, HNSW, cosine). That makes cards retrievable by
meaning, not just by graph lookup.

```{mermaid}
flowchart LR
    Q["query text"] --> E["embed (BGE)"]
    E --> KNN["knn over foodscholar_cards"]
    KNN --> H["nearest card ids + scores"]
    H --> R["fetch Card records"]
```

```python
fs.build_layer_c(facet="foods")            # build, embed, persist (Neo4j + cards index)
fs.search_cards("calcium and vitamin D for bones", k=5)   # vector search → nearest Cards
fs.graph.theme("...").card().summary       # graph lookup of a theme's card
```

```{note}
`fs.search_cards` is a thin retrieval helper (embed → kNN → fetch). The full `fs.query()` with
answer synthesis remains deferred; Stage 3 provides the searchable card store it will build on.
```

## Where Layer C sits

```{mermaid}
flowchart LR
    LA["Layer A<br/>shelves"] --> LB["Layer B<br/>themes"]
    LB --> LC["Layer C<br/>cards"]
    LC --> NEO[("Neo4j<br/>(:Card)")]
    LC --> CARDS[("cards index<br/>(vectors)")]
```

`fs.build()` runs the whole arc — `embed → build_entities → build_layer_a → attach →
build_layer_b → build_layer_c`. Because every card traces back through its theme's chunks to
their source documents, the trail **card → theme → chunk → source** stays auditable end to end.

```{seealso}
The Layer B & C methods brief (`docs/methods_layer_b_c_brief.md`) covers the two-stage internals,
the table-aware splitter, the benchmark findings, and the full Layer C config matrix.
```
