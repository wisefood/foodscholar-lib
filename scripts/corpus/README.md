# Corpus preparation

Tools that run **upstream of the library**, producing the inputs `fs.ingest()`
reads. They are not part of the package.

> **Note:** `kggen/` is no longer tracked in Git — it is a fork of an external
> project whose licence is not reproduced, so it is kept locally rather than
> redistributed. Paths below refer to that local reference snapshot; ask the
> WiseFood team if you need it.

## Superseded by the library

The guides and textbooks chunking notebooks
(`kggen/graph_code/chunking/chunking_pipeline_{guides,textbooks}_overlap.ipynb`)
have been ported to `foodscholar.corpus.chunker`. Use the library instead:

```bash
foodscholar chunk-corpus --config config.yaml \
    --pdf-dir data/original/guides --out-dir data/chunks/guides \
    --source-type guide \
    --metadata-csv data/original/guides/guide_metadata.csv \
    --excluded-pages data/pdfs.filtered.txt
```

The notebooks remain in `kggen/` as provenance — they are what produced the
existing corpus, and the port is validated against them (see the parity check
in the integration brief §6.6). Two behaviours were deliberately preserved
rather than improved, because changing either produces a *different* corpus:

- A chunk is dropped if **any** of its pages is excluded, so a chunk spanning
  a cover page loses its content too.
- `chunk_metadata["page_number"]` is the chunk's **first** page, not its span.

## Still notebook-only

**`chunking_abstracts_split_check.ipynb` §7–11 — semantic clustering.**
`MiniBatchKMeans` over normalized bge-large embeddings, partitioning the
already-chunked abstracts into 20 files. This is *file organization for disk
manageability*, not chunking: it changes which file a chunk lands in, never the
chunk. The splitting half (§1–6) is ported — it is the same sliding window over
NLTK sentences — and is reachable as `fs.chunk_texts()`.

Worth borrowing from it if you touch the caching layer: its embedding cache is
validated on model + row count + dimension + a sha1 over the ordered chunk ids,
and its final block asserts every input row lands in exactly one output file.

**`utils/pdf_page_triage.py` — LLM page triage.** Decides which PDF pages are
non-content and writes the `filename: X.pdf | removed_pages: [...]` manifest
that `chunk-corpus --excluded-pages` consumes. Runs once per document set,
caches decisions to `.triage_cache.json`. It calls Ollama directly; if it is
ever promoted into the library it should route through `LLMClient` like the
relation extractor does.

**`utils/guides_and_guidelines.ipynb` — source acquisition** via the wisefood
client. Stays in `kggen/`.

## Known gap

The textbooks' excluded pages were **inline Python sets** in the notebook, not
a manifest — six sets, hundreds of page numbers. Until they are exported to the
manifest format, textbook chunking is not reproducible outside the notebook.
Export them once and commit the manifest to `data/`.
