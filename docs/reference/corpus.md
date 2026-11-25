# Corpus

Loading the corpus: the CSV reader with its ingest-time validation and the NEL
loader. Producing it is [Chunking](chunking.md); the task guide is
[Chunking a corpus](../guides/chunking-a-corpus.md).

The window and the PDF and text chunkers moved to `foodscholar.chunking` — see the
[chunking reference](chunking.md#legacy-profile). `foodscholar.corpus.window` and
`foodscholar.corpus.chunker` still re-export them.

## Loading and validation

```{eval-rst}
.. automodule:: foodscholar.corpus.csv_reader
   :members:
   :member-order: bysource
```

```{eval-rst}
.. automodule:: foodscholar.corpus.nel_loader
   :members:
   :member-order: bysource
```
