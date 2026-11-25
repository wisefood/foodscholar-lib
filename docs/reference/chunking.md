# Chunking

`foodscholar.chunking` — source documents to cited, navigable chunks, usable
without the rest of the library. The model is on the
[concepts page](../concepts/chunking.md); the task guides are
[Chunking a corpus](../guides/chunking-a-corpus.md) and
[Evaluating a chunker](../guides/evaluating-a-chunker.md).

Everything below is importable from its module, and the main names also from
`foodscholar.chunking` itself (`Chunker`, `ChunkingConfig`, `DocumentIR`,
`SourceRegistry`, …). Importing the package loads nothing else from FoodScholar.

## Chunker

```{eval-rst}
.. automodule:: foodscholar.chunking.chunker
   :members:
   :member-order: bysource
```

## Configuration

```{eval-rst}
.. automodule:: foodscholar.chunking.config
   :members:
   :member-order: bysource
```

## Document IR

```{eval-rst}
.. automodule:: foodscholar.chunking.ir
   :members:
   :member-order: bysource
```

## Assembler

```{eval-rst}
.. automodule:: foodscholar.chunking.assembler
   :members:
   :member-order: bysource
```

## Identity and adjacency

```{eval-rst}
.. automodule:: foodscholar.chunking.identity
   :members:
   :member-order: bysource
```

The chunk stores' `expand()` reads the adjacency keys this module writes:

```{eval-rst}
.. automodule:: foodscholar.storage.adjacency
   :members: expand
```

## Source registry

```{eval-rst}
.. automodule:: foodscholar.chunking.registry
   :members:
   :member-order: bysource
```

## Outputs

```{eval-rst}
.. automodule:: foodscholar.chunking.writers
   :members:
   :member-order: bysource
```

## Heads

```{eval-rst}
.. automodule:: foodscholar.chunking.heads
   :members:
   :member-order: bysource
```

```{eval-rst}
.. automodule:: foodscholar.chunking.heads.record
   :members:
   :member-order: bysource
```

## Model protocols and text utilities

```{eval-rst}
.. automodule:: foodscholar.chunking.protocols
   :members:
   :member-order: bysource
```

```{eval-rst}
.. automodule:: foodscholar.chunking.text
   :members:
   :member-order: bysource
```

## Evaluation

```{eval-rst}
.. automodule:: foodscholar.chunking.evaluation.harness
   :members:
   :member-order: bysource
```

```{eval-rst}
.. automodule:: foodscholar.chunking.evaluation.gold
   :members:
   :member-order: bysource
```

```{eval-rst}
.. automodule:: foodscholar.chunking.evaluation.matching
   :members:
   :member-order: bysource
```

## Legacy profile

The pipeline that produced the stored corpus, unchanged. `foodscholar.corpus.chunker`
and `foodscholar.corpus.window` still re-export it.

```{eval-rst}
.. automodule:: foodscholar.chunking.legacy
   :members:
   :member-order: bysource
```

```{eval-rst}
.. automodule:: foodscholar.chunking.window
   :members:
   :member-order: bysource
```
