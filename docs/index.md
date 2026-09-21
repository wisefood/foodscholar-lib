# FoodScholar

**The whole pipeline from nutrition PDFs to ranked, citable evidence — chunking, ontology linking, knowledge-graph construction, and hybrid retrieval.**

FoodScholar takes dietary guides, textbooks and scientific abstracts and runs the
whole pipeline: **chunk** the PDFs, **link** every mention to the
[FoodOn](https://foodon.org) ontology, **construct** a three-layer hierarchical graph
over the result, and **retrieve** over it. Every layer is anchored to real evidence,
so a hit can always be traced back to the source chunk it came from.

The library retrieves and stops there — ranked, scored evidence. Answer synthesis
belongs to the service asking the question, which owns the model, the prompts and the
editorial policy the library has no view of.

```{mermaid}
flowchart LR
    PDFs[Source PDFs / abstracts] -->|chunker| Corpus[Chunked corpus]
    Corpus --> E[Entities<br/>FoodOn-linked]
    E -.->|opt-in| L0[Layer 0 — Relations<br/>typed, grounded edges]
    subgraph Graph[Navigation — browse the corpus]
      A[Layer A — Backbone<br/>FoodOn shelves] --> B[Layer B — Themes<br/>per-shelf communities]
      B --> C[Layer C — Cards<br/>cited write-ups]
    end
    E --> A
    Corpus --> Q[fs.retrieve<br/>text 0.3 + triples 0.3 + PageRank 0.4]
    L0 -.-> Q
    Q --> H[Ranked passages<br/>+ branch scores]
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

**[Retrieval](concepts/retrieval.md)** runs on the corpus and Layer 0 — not on Layers
A/B/C, which are the navigation surface. `fs.retrieve()` scores a passage on how it
reads, on what its triples assert, and on where it sits in the entity graph, so a
passage can surface because the graph connects it to the question even when the
wording does not match.

## The whole pipeline, not just the graph

FoodScholar is not only the hierarchy. It implements every stage between a
directory of PDFs and a ranked set of passages, and each one is usable on its
own:

| stage | what it does | entry point |
|---|---|---|
| [Chunking](guides/chunking-a-corpus.md) | PDFs → overlapping, heading-aware passages (Docling) | `fs.chunk_documents()` |
| [Annotation](concepts/annotation.md) | NER (GLiNER / GLiNER2) + dense linking to FoodOn | `fs.annotate()` |
| [Entities](concepts/annotation.md) | mentions deduped into first-class ontology-anchored entities | `fs.build_entities()` |
| [Layer 0 — Relations](concepts/layer-0-relations.md) | LLM triple extraction, deduped and ontology-grounded | `fs.build_relations()` |
| [Layers A/B/C](concepts/architecture.md) | backbone shelves, per-shelf themes, cited cards | `fs.build_layer_a()` … |
| [Retrieval](concepts/retrieval.md) | hybrid scoring over the corpus and Layer 0 | `fs.retrieve()` |

The construction and retrieval stages implement
**[Extended KG-Gen](concepts/extended-kg-gen.md)** — read that first if you
want the method rather than the API.

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
concepts/extended-kg-gen
concepts/worked-example
concepts/corpus-input
concepts/annotation
concepts/ontology
concepts/layer-0-relations
concepts/layer-a-backbone
concepts/layer-b-themes
concepts/layer-c-cards
concepts/retrieval
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
