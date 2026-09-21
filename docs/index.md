# FoodScholar

**A hierarchical knowledge graph over a corpus of nutrition literature — built for grounded, citable answers.**

FoodScholar chunks dietary guides, textbooks, and scientific abstracts, builds a
**three-layer hierarchical graph** over that corpus, and serves a retrieval API on
top. Every layer is anchored to real evidence, so an answer can always be traced
back to the source chunks that support it.

```{mermaid}
flowchart LR
    PDFs[Source PDFs / abstracts] -->|chunker| Corpus[Chunked corpus]
    Corpus --> E[Entities<br/>FoodOn-linked]
    E -.->|opt-in| L0[Layer 0 — Relations<br/>typed, grounded edges]
    subgraph Graph
      A[Layer A — Backbone<br/>FoodOn shelves] --> B[Layer B — Themes<br/>per-shelf communities]
      B --> C[Layer C — Cards<br/>cited write-ups]
    end
    E --> A
    A --> Q[Retrieval API]
    B --> Q
    C --> Q
    L0 -.-> Q
```

- **[Layer A — Backbone](concepts/layer-a-backbone.md).** A curated, multi-facet
  semantic menu projected from the [FoodOn](https://foodon.org) ontology (foods,
  health, nutrients, dietary patterns, allergies, sustainability).
- **[Layer B — Themes](concepts/layer-b-themes.md).** Fine-grained topic communities
  discovered **per shelf** by two complementary passes — embedding similarity and
  entity relatedness — then merged.
- **[Layer C — Cards](concepts/layer-c-cards.md).** LLM-generated write-ups attached to
  every shelf and theme, with **every claim cited back** to the source chunks.

Underneath the three sits **[Layer 0 — Relations](concepts/layer-0-relations.md)**:
typed, ontology-grounded edges (`olive oil --reduces--> LDL cholesterol`) extracted
from chunk text, each carrying the passages it came from. Opt-in, because extraction
costs an LLM pass over the corpus. And upstream of everything, the library can now
[produce its own corpus](guides/chunking-a-corpus.md) from source PDFs.

New here? Start with [](getting-started/quickstart.md), then read
[](concepts/architecture.md) for the whole picture.

```{toctree}
:caption: Getting started
:maxdepth: 2

getting-started/installation
getting-started/quickstart
getting-started/configuration
```

```{toctree}
:caption: Concepts
:maxdepth: 2

concepts/architecture
concepts/worked-example
concepts/corpus-input
concepts/annotation
concepts/ontology
concepts/layer-0-relations
concepts/layer-a-backbone
concepts/layer-b-themes
concepts/layer-c-cards
concepts/retrieval
concepts/kggen-provenance
concepts/glossary
```

```{toctree}
:caption: Guides
:maxdepth: 2

guides/chunking-a-corpus
guides/building-the-graph
guides/exploring-the-graph
guides/visualization
guides/tuning-layer-b
guides/cli
```

```{toctree}
:caption: API reference
:maxdepth: 2

reference/index
```
