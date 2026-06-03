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
      Mn -->|tiered linker| L[Entity links → FoodOn IDs]
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

## Two stores, one truth

Retrieval and graph navigation have different ideal databases, so FoodScholar uses
both and keeps them in lockstep.

| Elasticsearch owns **retrieval** | Neo4j owns **navigation** |
|---|---|
| BM25 keyword search | `(:Shelf)-[:PARENT_OF]->(:Shelf)` hierarchy |
| HNSW kNN (768-d `dense_vector`, cosine) | `(:Chunk)-[:ATTACHED_TO]->(:Shelf)` |
| hybrid via reciprocal-rank fusion | `(:Chunk)-[:THEME_OF {primary, weight}]->(:Theme)` |
| filter by `shelf_ids` / `theme_ids` | graph traversals (theme expansion, shelf walks) |

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

1. **Hybrid retrieval (Elasticsearch).** BM25 over `text` + kNN over `embedding`
   (query vector = `BGE_base("olive oil cardiovascular")`), fused by reciprocal
   rank, filtered to `shelf_ids ∋ olive_oil` (optionally also a health shelf).
   Returns the top-k chunks.
2. **Theme expansion (Neo4j → Elasticsearch).** Take those chunks' `theme_ids`;
   for each theme, pull sibling chunks via the `THEME_OF` edge. This adds
   *complementary* evidence — Mediterranean-diet passages, MUFA biochemistry —
   that pure kNN missed because the **phrasing** differed.
3. **Re-rank & present.** Combine retrieval score, theme-membership weight, and
   (from Layer C) evidence quality. The output is a small set of chunks with full
   provenance — source doc, section, FoodOn IDs, theme labels — ready for an LLM to
   ground an answer on.

Pure kNN returns near-duplicates; pure BM25 misses paraphrase; filtering by shelf
gives scope; hopping by theme gives complementary evidence. The provenance trail
**chunk → shelf → theme → source doc** is what makes a downstream answer auditable.

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
