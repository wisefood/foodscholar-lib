# Architecture

FoodScholar turns a pile of nutrition text into a **navigable, citable knowledge
graph**. The design is three layers built on top of each other, persisted across
two specialized stores, and held consistent by audit invariants.

The through-line for this page is one realistic query:

> *"Is olive oil good for cardiovascular disease?"*

## Three layers over two stores

```{mermaid}
flowchart TB
    subgraph Ingest[Ingestion]
      C[Corpus: textbooks, guides, abstracts] -->|chunker| K[Chunks]
      K -->|BGE-base 768d| E[Embeddings]
      K -->|NER| Mn[Mentions]
      Mn -->|dense linker| L[Entity links → FoodOn IDs]
    end
    subgraph LayerA[Layer A — Shelves]
      L -->|project onto FoodOn| SA[Shelf nodes]
      SA -.attach.-> KA[Chunks ↔ shelves]
    end
    subgraph LayerB[Layer B — Themes]
      KA -->|Pass 1: kNN + Leiden| T1[Similarity communities]
      KA -->|Pass 2: entity bridges + Leiden| T2[Relatedness communities]
      T1 --> TM[Themes]
      T2 --> TM
    end
    subgraph LayerC[Layer C — Cards]
      TM -.LLM-summarized.-> CD[Cards: title, summary, citations]
    end
    subgraph Storage[Storage]
      ES[(Elasticsearch<br/>BM25 + HNSW)]
      N4[(Neo4j<br/>graph)]
    end
    K --> ES
    E --> ES
    SA --> N4
    TM --> N4
    TM -.theme_ids denorm.-> ES
    SA -.shelf_ids denorm.-> ES
```

- **[Layer A — Backbone](layer-a-backbone.md)** anchors every chunk to the
  [FoodOn](https://foodon.org) ontology and projects a navigable hierarchy of
  *shelves* — the coarse menu a user browses.
- **[Layer B — Themes](layer-b-themes.md)** finds fine-grained topic communities
  *within each shelf* using two complementary signals, then merges them.
- **[Layer C — Cards](layer-c-cards.md)** writes a short, cited summary for each
  shelf and theme, with every claim traced back to source chunks.

Every layer builds on the previous one, and every layer is queryable on its own.

## Layer 0 — relations under the entity graph

The three layers are about *organizing* the corpus. **Layer 0** is about what
the corpus *asserts*: typed edges between the entities the annotate phase
already links.

```
Layer C   cards          cited write-ups
Layer B   themes         per-shelf topic communities
Layer A   shelves        the FoodOn backbone
──────────────────────────────────────────────────────────────
Layer 0   relations      (:Entity)-[:RELATED {predicate}]->(:Entity)
          entities       (:Entity)
          chunks         (:Chunk)
```

It is deliberately *under* the entity graph rather than a fourth layer beside
C: it depends on nothing above `build_entities()`, and nothing in A/B/C depends
on it. Both endpoints of every relation are ontology ids produced by the same
linker that annotates chunks, so a relation connects the same `Entity` records
Layer A projects and Layer B clusters — one entity universe, not two.

Layer 0 is **opt-in** (`relations.enabled`), because extraction costs two LLM
calls per chunk. See [Layer 0 — Relations](layer-0-relations.md). It is adapted
from the WiseFood **kggen** pipeline, as is the corpus chunker — see
[Extended KG-Gen](extended-kg-gen.md).

## Two stores, one truth

Retrieval and graph navigation have different ideal databases, so FoodScholar uses
both and keeps them in lockstep. Elasticsearch can't cheaply do multi-hop traversals
(chunk → theme → sibling chunks → shelf); Neo4j can't do good hybrid retrieval. So each
store owns what it's best at:

| Elasticsearch owns **retrieval** | Neo4j owns **navigation** |
|---|---|
| BM25 keyword search | `(:Shelf)-[:PARENT_OF]->(:Shelf)` hierarchy |
| HNSW kNN (768-d `dense_vector`, cosine) | `(:Chunk)-[:ATTACHED_TO]->(:Shelf)` |
| hybrid via [reciprocal-rank fusion](glossary.md) (RRF) | `(:Chunk)-[:THEME_OF {primary, weight}]->(:Theme)` |
| filter by `shelf_ids` / `theme_ids` | graph traversals (theme expansion, shelf walks) |

On the `THEME_OF` edge, `primary` marks a chunk's single best theme on a shelf (a chunk
can belong to several), and `weight` is its membership strength — both used when ranking.

A chunk's `shelf_ids` and `theme_ids` are **denormalized** onto its Elasticsearch
document so retrieval can filter by shelf/theme without round-tripping to Neo4j. The
two stores can drift, so an **audit** runs cross-store invariants after every Layer A
or Layer B build:

```text
[CRITICAL] shelf_ids (Elastic) ↔ ATTACHED_TO (Neo4j) parity = 1.0
[CRITICAL] no empty themes; no dangling theme_ids
[CRITICAL] no cycles in PARENT_OF
```

A failing critical invariant fails the build — drift is never allowed to ship.

```{note}
The `memory` backend implements the same protocols in-process, so the whole
pipeline (and the test suite) runs with zero services. See
[](../getting-started/configuration.md).
```

## One query, end to end

Tracing *"Is olive oil good for cardiovascular disease?"* shows why the structure
earns its keep:

1. **Candidate generation (Elasticsearch).** kNN over `embedding`
   (query vector = `BGE_base("olive oil cardiovascular")`) returns
   `retrieval.candidate_k` chunks. This is the only branch that touches the whole
   corpus; the other two re-rank what it returns.
2. **What the passages assert (Layer 0).** Pull the triples extracted from those
   chunks and score each chunk by the mean similarity of its triples to the query.
   A passage that *states* `olive oil --reduces--> LDL cholesterol` scores here even
   if its wording is nothing like the question.
3. **Where they sit in the graph (Personalized PageRank).** Seed PageRank at the
   entities nearest the query, walk the relation graph, and propagate the mass back
   onto the passages that evidence those entities. This is what surfaces
   Mediterranean-diet passages and MUFA biochemistry — reached through the graph,
   not through phrasing.

The three are min-max normalized and summed at 0.3 / 0.3 / 0.4. The output is a small
set of chunks with full provenance — source doc, section, FoodOn IDs — plus the branch
scores that ranked them, ready for a caller to ground an answer on. The library stops
there; see [Retrieval](retrieval.md).

Pure kNN returns near-duplicates and misses paraphrase; the graph branches are what fix
that, and they need Layer 0 — without it, retrieval falls back to branch 1 alone. The
provenance trail **chunk → shelf → theme → source doc** is what makes a downstream
answer auditable.

## Design lessons baked into the pipeline

A few hard-won rules the codebase now enforces:

- **Validate a read round-trip, not just the index.** Elasticsearch 9.x strips
  `dense_vector` from `_source` even when the mapping is correct; embeddings are
  read back via the `fields` API and merged.
- **Downstream IR is only as clean as ingestion.** OCR artifacts (font-PUA glyphs
  like `h18567`) once dominated c-TF-IDF theme labels; the keyword labeler now
  filters digit-bearing, sub-3-char, and id-like tokens.
- **Defaults should fail loud, not plausibly.** `in_memory()` uses a *mock* LLM;
  wire a real provider into `fs` at construction so a forgotten setup doesn't ship
  `"Mock answer citing [CHUNK]"` as a theme label.
