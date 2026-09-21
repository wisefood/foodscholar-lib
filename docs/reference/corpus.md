# Corpus

Producing and loading the corpus: the shared sliding window, the PDF and text
chunkers behind `fs.chunk_documents()` / `fs.chunk_texts()`, and the CSV
reader with its ingest-time validation. The task guide is
[Chunking a corpus](../guides/chunking-a-corpus.md).

The window and the PDF producer are a port of the kggen Docling notebooks
(not distributed with FoodScholar) — see the
[Extended KG-Gen](../concepts/extended-kg-gen.md).

## The window

```{automodule} foodscholar.corpus.window
:members:
:member-order: bysource
```

## Chunkers

```{automodule} foodscholar.corpus.chunker
:members:
:member-order: bysource
```

## Loading and validation

```{automodule} foodscholar.corpus.csv_reader
:members:
:member-order: bysource
```

```{automodule} foodscholar.corpus.nel_loader
:members:
:member-order: bysource
```
