# Chunking a corpus

FoodScholar reads **pre-chunked** corpus CSVs. This guide covers producing
them from source PDFs and raw text, so the library can rebuild its own input
rather than depending on a notebook elsewhere.

```{warning}
**Re-chunking an already-ingested corpus is destructive.** Under the default
`chunk_id_strategy: uuid4`, chunking assigns a fresh UUID to every chunk. New
ids orphan every `Relation.chunk_ids`, `Entity.chunk_ids`, shelf attachment,
theme membership and `Card.cited_chunk_ids` that pointed at the old ones — the
graph survives, but its provenance no longer resolves. `chunk_documents()`
refuses to write into a non-empty output directory unless `force=True`.
See [Chunk ids](#chunk-ids) for the idempotent alternative.
```

## The corpus format

One CSV per document, four columns:

| column | meaning |
|---|---|
| `chunk_id` | unique id; also the join key for every downstream layer |
| `chunk_text` | the passage |
| `type` | `guide`, `textbook` or `abstract` |
| `chunk_metadata` | a Python-literal dict — see below |

`chunk_metadata` differs **by source type**, which is why
`ChunkProvenance` makes every field optional:

| source | keys |
|---|---|
| guide | `file`, `urn`, `pdf_name`, `country`, `title`, `audience`, `pages`, `heading`, `page_number` |
| textbook | `file`, `heading`, `page_number` — **only these three** |
| abstract | `title`, `year`, paper identifiers (`doi` / `paperId`) |

A citation renderer must therefore degrade per source type: a textbook chunk
has no title and no country.

```{note}
`page_number` is the chunk's **first** page, not its span. A 512-token chunk
can cover several pages, so a citation into its tail may point one page early.
```

## Chunking PDFs

```python
from foodscholar import FoodScholar

fs = FoodScholar.from_config("config.yaml")
fs.chunk_documents(
    "data/original/guides",
    out_dir="data/chunks/guides",
    source_type="guide",
    metadata_csv="data/original/guides/guide_metadata.csv",
    excluded_pages="data/pdfs.filtered.txt",
)
fs.ingest("data/chunks/guides")
```

or from the CLI:

```bash
foodscholar chunk-corpus --config config.yaml \
    --pdf-dir data/original/guides \
    --out-dir data/chunks/guides \
    --source-type guide \
    --metadata-csv data/original/guides/guide_metadata.csv \
    --excluded-pages data/pdfs.filtered.txt
```

Needs the `[chunking]` extra (`pip install 'foodscholar[chunking]'`) — Docling
pulls torch and layout models, so it stays optional.

`metadata_csv` is **required for guides**: every PDF must have a row keyed on
its filename stem, and a missing one raises rather than silently producing
metadata-less chunks.

### Excluded pages

Covers, tables of contents and ad pages are dropped via a manifest:

```
filename: ie-key-messages.pdf | removed_pages: [1, 2, 57]
```

```{warning}
A chunk is dropped if **any** of its pages is excluded. A 512-token chunk
spanning a cover page and a content page is discarded whole, content included.
This reproduces the original pipeline; whether to trim instead is an open
question, and changing it produces a different corpus.
```

Textbooks have no manifest in the original pipeline — their excluded pages
were inline Python sets. Pass them explicitly, or export them to a manifest
once:

```python
fs.chunk_documents(
    "data/original/textbooks",
    out_dir="data/chunks/textbooks",
    source_type="textbook",
    excluded_pages={"Human-Nutrition.pdf": {1, 2, 3, 28, 30}},
)
```

## Chunking raw text

Abstracts use the same window over NLTK sentences instead of Docling units —
it is one algorithm, not two:

```python
fs.chunk_texts(
    {"10.1/abc": "Whole-grain intake is associated with ...", ...},
    out_path="data/chunks/abstracts/abstracts.csv",
    source_type="abstract",
    metadata={"10.1/abc": {"title": "...", "year": 2019, "doi": "10.1/abc"}},
)
```

Only texts over `max_tokens` are actually split; shorter ones pass through as
a single chunk.

## How the window works

Four semantics, each of which changes the output:

1. **Grouping is by consecutive units.** Units are grouped by top-level
   heading using `itertools.groupby`, so a window never spans two sections and
   overlap never bleeds across chapters. Non-adjacent runs of the same heading
   form separate groups.
2. **Everything that fits is emitted as one chunk**, then the group ends — no
   redundant tail-only chunk.
3. **`overlap` is a minimum tail, not a target.** The window advances to the
   furthest unit that still leaves at least `overlap` tokens behind it, so the
   realized overlap is >= 64 and frequently more.
4. **A single unit larger than `max_tokens`** is emitted alone, and the window
   advances by one so it always makes progress.

The window itself is pure and dependency-free
(`foodscholar.corpus.window.build_overlapping_chunks`) — it takes a
`count_tokens` callable, so it is testable without loading a tokenizer.

## Chunk ids

```yaml
chunking:
  chunk_id_strategy: uuid4        # or: content_hash
```

- **`uuid4`** (default) reproduces the historical corpus bit-for-bit. Every
  run assigns new ids, so re-chunking orphans downstream provenance.
- **`content_hash`** derives the id from `source_doc_id` + normalized text.
  Re-chunking an unchanged document reproduces its ids, so relations,
  attachments and cards survive. **Recommended for any new corpus.**

## Validating an existing corpus

The library assumes chunks are at most 512 tokens. `fs.ingest` can now check:

```yaml
corpus:
  validation:
    enabled: true
    max_chunk_tokens: 512
    token_estimate: chars     # or `tokenizer` for an exact count
    on_violation: warn        # or `raise`
```

It **warns by default and never raises** — a corpus that trips the check is
still a corpus, and failing ingest over one long chunk is the worse outcome.
Oversized chunks are logged as `corpus.chunk_oversized`, with a summary at the
end of the run.

## Tokenizers

Chunking counts tokens with **`BAAI/bge-large-en-v1.5`** and
`use_fast_tokenizer: false`. That is deliberately *not* the retrieval embedder
(`annotate.embedder`, bge-**base**): the two share a tokenizer so the counts
agree, but the corpus was counted with the large model's slow tokenizer, and
the fast and slow variants can disagree by a token. Keep both pinned.

## See also

- [Corpus input](../concepts/corpus-input.md) — the contract `ingest` expects
- [Building the graph](building-the-graph.md) — what happens after ingest
- [CLI](cli.md) — `chunk-corpus`
