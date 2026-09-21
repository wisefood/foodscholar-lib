# Retrieval

Extended KG-Gen hybrid passage retrieval behind `fs.retrieve()`. The concepts,
the cost model and the tuning knobs are on the
[retrieval page](../concepts/retrieval.md); the scoring's provenance is on the
[Extended KG-Gen](../concepts/extended-kg-gen.md).

The library retrieves and returns ranked evidence. Answer synthesis is the
caller's.

## Retriever

```{automodule} foodscholar.retrieval.kggen
:members: KGGenRetriever, RetrievalHit, RetrievalTrace
:member-order: bysource
```

## Configuration

```{autoclass} foodscholar.config.RetrievalConfig
:members:
```
