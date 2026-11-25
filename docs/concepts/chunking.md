# Chunking

`foodscholar.chunking` turns source documents into chunks that keep their
context: the section they belong to, the pages and regions they came from, the
figures and tables they discuss, and their neighbours. It is part of the
library and usable on its own — importing it loads no store, ontology, LLM
client or facade, and running it needs no services. Its output feeds
`fs.ingest` unchanged.

```{note}
Two pipelines live side by side. The **legacy** pipeline (a Docling sliding
window, `fs.chunk_documents()`) produced the stored corpus and is kept
bit-for-bit. The **v2** pipeline described here is the structure-first
assembler. Today v2 extracts journal abstracts itself (the record head) and
assembles any document already in its intermediate representation; the heads
for textbook and guide PDFs are not implemented yet. See
[Chunking a corpus](../guides/chunking-a-corpus.md) for both.
```

## Why a second pipeline

Measured on the stored corpus with the real `bge-base` tokenizer:

| What the legacy window does | Measured effect |
|---|---|
| Groups by the top-level heading only | chunks never know their sub-section; the gold section matches a chunk's section 33 % of the time on the Irish guides |
| Records a chunk's first page only | a citation into a chunk's tail points one page early |
| Copies a 64-token tail into the next chunk | ≥ 14 % more tokens, and the same sentence becomes citable from two chunk ids |
| Fixes the window at 512 tokens of *content* | 20 % of the stored abstract chunks exceed 512 tokens and are silently truncated by the embedder |
| Discards every image | picture-based guides lose most of their content; guide chunks are fragments (median 55 tokens on the Irish guides, 81 % under 120) |

The numbers come from the evaluation harness; see
[Evaluating a chunker](../guides/evaluating-a-chunker.md).

## The pipeline

```{mermaid}
flowchart LR
    F[Source file] -->|head, chosen by genre| IR[DocumentIR<br/>typed elements + regions]
    R[(Source registry)] -.->|title, licence, pages| IR
    IR --> A[Assembler<br/>structure-first packing]
    A --> I[Identity + adjacency]
    I --> CSV[Corpus CSVs<br/>fs.ingest]
    I --> J[chunks.jsonl<br/>evaluation]
    S[(Chunk store)] -.->|expand: window, section,<br/>page, figure| Q[Retrieval / Layer C]
    CSV --> S
```

## Genres, heads and profiles

Heads are organised by **document genre**, not by domain:

| Genre | What it is | Nutrition source type | Elsewhere |
|---|---|---|---|
| `record` | a short structured record | journal abstract | an earnings release |
| `long_document` | a long structured PDF with an outline | textbook | an annual report |
| `leaflet` | a layout-heavy leaflet, poster or infographic | dietary guide | an investor one-pager |

`ChunkingConfig.sources` maps each source type onto a genre and its assembly
settings. A new domain is a new mapping, not new code. Nothing below the
profile knows about nutrition: the assembler never calls a domain model, and
annotation (GLiNER, the FoodOn linker) happens downstream as before.

## The document IR

Every head emits a {py:class}`~foodscholar.chunking.ir.DocumentIR`: document
metadata plus elements in reading order — `Heading`, `Paragraph`, `ListItem`,
`Table`, `Figure`, `Caption`, `Box`, `Footnote` and `Furniture` (running
headers and page numbers, kept for audit, never chunked).

A **page is a unit of provenance, not of content.** An element that continues
across a page break is one element with one `Region` per page it touches. Each
region states its page, its bounding box and the box's coordinate space, since
extractors disagree on where the origin is.

Tables carry cells, header rows, units and footnotes as data. Figures carry a
domain-free kind (`photo`, `diagram`, `chart`, `table_image`, `infographic`,
`logo`) that the imagery path will dispatch on.

## The assembler

Six rules, applied in order:

1. **Sections from headings.** A heading stack gives every element its *full*
   section path, not only the top-level heading.
2. **Atomic units.** A paragraph, a whole list, a table with its caption, a
   figure with its caption and a box are each one unit.
3. **Pack within a section, never across, in balanced parts.** A section that
   fits stays whole unless it is well over `target_tokens`; one that does not is
   cut into the fewest near-equal parts the budget allows, so no part is a
   stray tail. The budget, `max_tokens`, counts the context header, the content
   and the tokenizer's special tokens — exactly what the embedder sees.
4. **Split only what does not fit**: a paragraph at sentences, a list between
   items, a table between rows with its header repeated, a single over-long
   sentence by words as the last resort.
5. **Merge fragments.** A section's short tail rejoins the previous chunk;
   adjacent small chunks that share a parent section merge, each keeping its
   own subsection title in the text.
6. **No overlap by default.** See below.

Each chunk carries four kinds of text, never mixed: the faithful **source
text** a citation displays, a deterministic **context header** (document short
name, then the section path), the **retrieval text** (header + source text)
that is embedded and indexed, and — later — versioned generated text such as
translations or figure descriptions.

## Adjacency instead of overlap

Overlap copies a fixed tail of the previous chunk into the next one. Whether
that tail holds the sentence a reader needs is luck; it cannot reach two chunks
back, and it duplicates evidence. Every v2 chunk instead carries `ordinal`,
`prev_chunk_id`, `next_chunk_id` and `parent_id`, and both chunk stores expose:

```python
fs.chunk_store.expand(chunk_id, "window", n=1)   # the chunk and n neighbours either side
fs.chunk_store.expand(chunk_id, "section")       # every chunk of its section, in order
fs.chunk_store.expand(chunk_id, "page", page=37) # every chunk touching a page
fs.chunk_store.expand(chunk_id, "figure")        # every chunk citing the same figure or table
```

This is what iterative or agentic retrieval uses to follow a hit, and what
Layer C can use to gather evidence. One sentence of overlap remains available
as an ablation (`overlap_sentences`). Legacy chunks carry no adjacency keys and
expand to themselves.

## Identity

Two ids, because one cannot do both jobs:

- `chunk_version_id` is a content hash of the source id, the section path and
  the normalised text. It changes whenever the evidence changes.
- `chunk_id` is what relations, attachments and cards cite. On a first run it
  equals the version id. On a re-run, an unchanged chunk keeps its old id; a
  changed one gets a new id and a `supersedes` link to the chunk it most
  resembles in the same section; a vanished one is reported as retired.

An old citation therefore keeps resolving to the evidence it was made against,
and is never silently redirected to merely similar text.

## Provenance

Everything v2 adds travels in `chunk_metadata`, which Elasticsearch maps as
`flattened`, so no index change is needed to filter on it. The v1 keys (`file`,
`heading`, `page_number`) are still written. `Chunk.provenance` reads all of
them; see [Corpus input](corpus-input.md).

| Key | Meaning |
|---|---|
| `source_id` | the registry's persistent document id; also the chunk's `source_doc_id` |
| `section_path` | the full heading path |
| `page_start`, `page_end`, `pages` | the physical pages the chunk touches |
| `regions` | one region per page: page, bbox, coordinate space |
| `printed_page_start`, `printed_page_end` | printed labels, only when the PDF or the registry provides them — never invented |
| `language`, `licence` | per chunk, from the element or the registry |
| `context_header` | the header embedded with the text |
| `ordinal`, `prev_chunk_id`, `next_chunk_id`, `parent_id` | adjacency |
| `chunk_version_id`, `supersedes` | identity |
| `figure_ids`, `table_ids`, `element_ids` | what the chunk is made of |

## Records: journal abstracts

The record head reads a JSON list (or JSONL) of records and makes each one a
document: its DOI is the `source_id`, its title is the context header, and its
journal's catalogue id is kept as `collection_id`. A structured abstract
becomes one paragraph per labelled section, in any of the three shapes the
journals use — `Purpose of review: …`, `Background Two previous …` and the
glued `Purpose of reviewThe first …`. Section text is rendered `Label: text`;
only separators change, never words. An abstract counts as structured only
when it starts with a known label and has at least two.

An abstract stays one chunk whenever it fits the budget. When it does not, it
is cut at section boundaries into balanced parts that share one `parent_id`,
so `expand(chunk_id, "section")` returns the whole abstract. A part that is
exactly one section records its type (`methods`, `results`, …) as the chunk's
`section_type`.

On the 937 non-empty abstracts in `sources/` (measured 15 October 2026, exact
`bge-base` counts):

| What | Result |
|---|---|
| Abstracts detected as structured | 322 |
| Chunks over 512 embedded tokens | 0 (the stored corpus: 20 % of abstract chunks) |
| Abstracts whose chunks reconstruct their text exactly | 937 of 937 |
| Abstracts split | 98, smallest part 116 tokens |

Field names are configurable (`RecordHeadOptions`), so another domain's
records — an earnings release, a product sheet — need no new code.

## The source registry

Document-level facts come from one catalogue instead of per-run CSVs:
{py:class}`~foodscholar.chunking.registry.SourceRegistry` reads the FSKG
workbook (`FSKG_Sources_Extracted.xlsx`) or any CSV/JSON with the same core
columns. It parses page notes such as *"book content on PDF pages 43-1190;
printed page = PDF page - 42"* into a content range and a printed-page offset,
splits bilingual edition pairs by language, and maps gold-statement ids to
their source. A file that is not catalogued itself but sits beside a
catalogued one (a journal's records JSON next to its manifest) resolves
through its directory.

## See also

- [Chunking a corpus](../guides/chunking-a-corpus.md) — running it
- [Evaluating a chunker](../guides/evaluating-a-chunker.md) — scoring it, and the baseline
- [Chunking reference](../reference/chunking.md) — the API
