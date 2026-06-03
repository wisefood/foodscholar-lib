# FoodScholar

**A hierarchical knowledge graph over a corpus of nutrition literature — built for grounded, citable answers.**

FoodScholar ingests dietary guides, textbooks, and scientific abstracts, then
builds a **three-layer hierarchical graph** over the chunked corpus and serves a
retrieval API on top. Every layer is anchored to real evidence, so an answer can
always be traced back to the source chunks that support it.

```{mermaid}
flowchart LR
    Corpus[Chunked corpus] --> A
    subgraph Graph
      A[Layer A — Backbone<br/>FoodOn shelves] --> B[Layer B — Themes<br/>per-shelf communities]
      B --> C[Layer C — Cards<br/>cited write-ups]
    end
    A --> Q[Retrieval API]
    B --> Q
    C --> Q
```

- **Layer A — Backbone.** A curated, multi-facet semantic menu projected from the
  [FoodOn](https://foodon.org) ontology (foods, health, nutrients, dietary
  patterns, allergies, sustainability).
- **Layer B — Themes.** Fine-grained topic communities discovered **per shelf** by
  two complementary passes — embedding similarity and entity relatedness — then merged.
- **Layer C — Write-ups.** LLM-generated cards attached to every shelf and theme,
  with **every claim cited back** to the source chunks.

New here? Start with [](getting-started/quickstart.md).

```{note}
These docs are being built out. The **Getting started** section below is complete;
**Concepts**, **Guides**, and the **API reference** are being filled in next.
```

```{toctree}
:caption: Getting started
:maxdepth: 2

getting-started/installation
getting-started/quickstart
getting-started/configuration
```
