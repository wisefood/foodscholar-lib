# Extended KG-Gen

**Extended KG-Gen is the method FoodScholar uses to construct its knowledge
graph from raw documents, and to retrieve over it.** It is a pipeline developed
by colleagues on the WiseFood project: an extension of KGGen that keeps the
passage every triple came from, plus a hybrid retriever over the resulting
graph, an NER/NEL benchmark, and the Docling chunking that produces the corpus.

FoodScholar implements it end to end. This page is the method; the
[Layer 0 page](layer-0-relations.md) is the extraction half in depth and
[Retrieval](retrieval.md) is the query half.

## The method, end to end

```{mermaid}
flowchart TD
    PDF[Source PDFs] -->|Docling chunker| CH[Passages<br/>overlapping, heading-aware]
    CH -->|GLiNER / GLiNER2| M[Mentions]
    M -->|HNSW + BioLORD / SapBERT| EN[Entities<br/>FoodOn ids]
    CH -->|LLM step 1| E1[Entity list per passage]
    E1 -->|LLM step 2, endpoints constrained| TR[Triples<br/>subject, predicate, object]
    TR -->|NFKC + singularize + semantic hash| DD[Deduplicated triples]
    DD -->|grounding — FoodScholar's addition| R[Relations<br/>ontology-anchored, passage-carrying]
    EN --> R
    R --> Q[Hybrid retrieval<br/>text + triples + PageRank]
    CH --> Q
```

**1 — Chunk.** Docling turns PDFs into overlapping, heading-aware passages. The
passage is the unit of provenance for everything downstream: whatever a triple
claims, the passage it came from can be shown.

**2 — Link.** NER finds mentions; a dense HNSW index over FoodOn resolves each
to an ontology id. This runs independently of extraction and produces the
entity graph.

**3 — Extract, in two constrained LLM steps.** Step 1 asks for the entities in
a passage. Step 2 asks for relations **whose subject and object must come from
step 1's list** — an enum constraint, not a suggestion. That constraint is the
load-bearing idea: an unconstrained extractor invents endpoints that match
nothing else in the graph, and the triples then sit in their own disconnected
world. The two prompts are the benchmarked artifact and are used verbatim.

**4 — Deduplicate.** Surfaces and predicates are NFKC-normalized,
singularized per token, and collapsed by semantic hash. Predicates are where an
open extractor sprawls (`reduces` / `lowers` / `decreases`), and under-merging
them biases the PageRank branch later, because distinct predicates become
parallel edges and extra votes.

**5 — Ground.** *FoodScholar's addition, and the join between the two systems.*
KG-Gen's endpoints are free text. FoodScholar resolves them through the same
linker that produced the entity graph, so a KG-Gen triple becomes an edge
between two **ontology-anchored** entities rather than between two strings.
Endpoints that do not resolve are kept as `NIL:` sentinels rather than dropped
— discarding them would delete every relation touching a concept FoodOn lacks,
which for a nutrition corpus is most biomarkers, hormones and processes.

**6 — Retrieve.** The hybrid scores a passage on how it reads (0.3), on what
its triples assert (0.3), and on where it sits in the entity graph by
Personalized PageRank (0.4). The graph branches are what make this more than
vector search.

Steps 1–5 are `fs.chunk_documents()` → `fs.annotate()` → `fs.build_relations()`;
step 6 is `fs.retrieve()`.

## What FoodScholar adds

KG-Gen produces a graph of strings. FoodScholar anchors it:

- **Ontology grounding** (step 5) — endpoints become FoodOn/CHEBI ids, so a
  triple joins the same entity graph the annotation pipeline built.
- **The hierarchy** — Layers A/B/C sit above the relations, so retrieval can be
  scoped to a shelf or theme. KG-Gen has no shelf concept: it seeds PageRank
  from all entities, where FoodScholar can seed from the entities of one shelf.
- **Store-backed everything** — the reference kept a GraphML file and a ~520MB
  embedding cache on disk. Here each stage reads and writes the configured
  stores, so a rebuild is visible immediately.

## Two things are called "kggen"

The name is overloaded, and the distinction matters for licensing:

- **KGGen** — an external, third-party project for LLM knowledge-graph
  extraction.
- **`kggen_extended`** — WiseFood's *fork* of that project, adding
  passage-level provenance to every triple.
- **The WiseFood pipeline** — the surrounding chunking, benchmark and
  retrieval code written by WiseFood colleagues, which uses the fork.

When these docs say "the kggen pipeline" they mean the third: WiseFood's own
code. Where a specific piece descends from the external project's fork, the
table below says so.

## Where the original lives

The source pipeline is **not distributed with FoodScholar**. It is not in the
Git repository, not in the sdist, and not in the wheel — it is kept locally by
the team as a reference snapshot, and nothing in the package imports from it.

That is deliberate. `kggen_extended` is a fork of an external project whose
licence text is not reproduced alongside it, so redistributing it inside an
Apache-2.0 package would put FoodScholar in an unclear position. FoodScholar
ships its own re-implementation (Apache-2.0, like the rest of the library);
the fork stays upstream with its authors.

If you need the original, ask the WiseFood team, and observe the external
project's own licence for anything derived from `kggen_extended`.

## What FoodScholar took

| in FoodScholar | from kggen | what changed |
|---|---|---|
| The two extraction prompts (`relations/prompts/*.txt`) | `kggen_extended/prompts/` | **Copied verbatim.** They are the benchmarked artifact; treat them as fixtures. |
| Entity and relation extraction (`relations/extract.py`) | `kggen_extended/steps/_1_get_entities.py`, `_2_get_relations.py` | Ported from dspy/LiteLLM onto FoodScholar's `LLMClient`. The enum constraint on relation endpoints — subject and object must come from the step-1 entity list — is kept; it is what keeps triples aligned with entities. |
| Passage provenance (`relations/extracted.py`) | `kggen_extended/models.py` (`ExtendedGraph`) | Slim rewrite, extractor-internal only. The property that matters — a triple keeps every passage it was extracted from, and collapsing triples **union** their passages — is preserved. The upstream validator's stale-key repair (for savepoint files this library does not have) is not. |
| Surface and predicate dedup (`relations/dedupe.py`) | `kggen_extended/utils/deduplicate.py` | Ported: NFKC normalization, per-token singularization, semantic hashing. Made deterministic — the original iterated a `set`, so its canonical pick varied with `PYTHONHASHSEED`. |
| The chunking window (`corpus/window.py`) and PDF producer | `chunking/chunking_pipeline_{guides,textbooks}_overlap.ipynb` | Three notebooks turned out to share one algorithm; it is written once, dependency-free, and the parameters that produced the existing corpus are pinned. The abstracts notebook's sentence splitting is the same window over NLTK sentences. |
| GLiNER2 and its 27 described labels (`annotate/gliner2_ner.py`) | `ner-nel/run_ner_nel_corpus_gliner2_sapbert.py` and the two benchmark notebooks | Label set copied verbatim; shipped as an **option**, not the default — see below. |

**Grounding** (`relations/ground.py`) is the one piece with no kggen
counterpart, and it is the join between the two systems. kggen's entities are
free-text strings; FoodScholar's are ontology ids. Every relation endpoint is
resolved through FoodScholar's own linker, so a kggen-style triple becomes an
edge between the same `Entity` records Layer A projects and Layer B clusters.
Without it there would be two disjoint entity universes in one graph.

## What was not adopted as a default

The kggen README presents GLiNER2 and the SapBERT linking encoder as its
selected production defaults. Read from the notebooks' saved outputs, the
evidence is mixed: on the single-passage human-scored evaluation the model
FoodScholar already ships (GLiNER-bio) scores highest; GLiNER2's win is on a
cross-dataset benchmark scored against a GPT-4o-mini proxy, where it trades
about 12% recall for about 40% precision and yields roughly 28% fewer mentions
per passage. FoodScholar's Layer A support counts and Layer B relatedness graph
key off mention volume, so that trade is not free here. Both ship as
configurable options, and `research/ner_nel_bakeoff/` in the repository holds
the harness — and the published numbers, verbatim — that gates any change to
the defaults.

## Not ported

- **Hybrid retrieval** (`retrieval/retrieval_core.py`: text similarity + mean
  triplet similarity + Personalized PageRank over the entity subgraph).
  **Ported** — as `fs.retrieve()` / `foodscholar.retrieval.kggen`, with the
  same 0.3/0.3/0.4 scoring read from the stores instead of a GraphML file and
  a 520MB embedding cache. See [Retrieval](retrieval.md).
- **Abstracts semantic clustering** (`chunking_abstracts_split_check.ipynb`,
  sections 7–11) — partitions the already-chunked corpus into files; disk
  organization rather than chunking.
- **LLM page triage** (`utils/pdf_page_triage.py`) — produces the
  excluded-pages manifest the chunker consumes; stays a corpus-prep script.
- `export.py`, `visualize_kg.py`, `chunk_text.py` — covered by
  `io/graphml.py`, `viz/` and the ≤512-token chunk contract respectively.
- `utils/llm_deduplicate.py` — deferred in favour of reusing
  `layer_a/semantic_consolidation/`.

## Crediting and licence

FoodScholar's implementation is Apache-2.0, like the rest of the library. It is
an independent re-implementation, not a redistribution: the external project's
code is not included in this package.

Two pieces are **copied verbatim** rather than rewritten, and are the WiseFood
team's work — the two extraction prompts (`relations/prompts/*.txt`) and the
27-label GLiNER2 label set in `GLiner2Config`. Both are benchmarked artifacts;
they are treated as fixtures, and editing them invalidates the published
numbers.

If you build on Layer 0, the chunker or the GLiNER2 configuration, credit the
WiseFood kggen pipeline alongside FoodScholar. Anything you derive from the
`kggen_extended` fork is additionally subject to the **external KGGen
project's own licence** — check it with the WiseFood team before
redistributing.

Each ported module's docstring names the source file it came from.
