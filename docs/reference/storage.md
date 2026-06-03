# Storage protocols

Storage is defined by **protocols** — interfaces every backend implements. The `memory`,
`elastic`, and `neo4j` adapters all satisfy these, which is why the entire pipeline runs
unchanged regardless of where data lives. See [](../concepts/architecture.md) for the
two-stores design and [](../getting-started/configuration.md) for selecting backends.

```{automodule} foodscholar.storage.protocols
:members:
:member-order: bysource
```
