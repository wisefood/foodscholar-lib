# Relations (Layer 0)

The Layer 0 pipeline — extract, dedupe, ground, aggregate, persist — behind
`fs.build_relations()`. The `Relation` contract itself is in the
[data model](data-model.md); the concepts are on the
[Layer 0 page](../concepts/layer-0-relations.md).

Most of this package is a port of the **kggen** pipeline
(the WiseFood pipeline), moved onto FoodScholar's
`LLMClient`, linker and stores. Each module's docstring says what came from
kggen and what changed; the summary is on the
[provenance page](../concepts/kggen-provenance.md).

## Orchestration

```{automodule} foodscholar.relations.builder
:members:
:member-order: bysource
```

## Extraction

```{automodule} foodscholar.relations.extract
:members:
:member-order: bysource
```

```{automodule} foodscholar.relations.extracted
:members:
:member-order: bysource
```

## Deduplication

```{automodule} foodscholar.relations.dedupe
:members:
:member-order: bysource
```

## Grounding

```{automodule} foodscholar.relations.ground
:members:
:member-order: bysource
```

## Aggregation and persistence

```{automodule} foodscholar.relations.aggregate
:members:
:member-order: bysource
```

```{automodule} foodscholar.relations.persist
:members:
:member-order: bysource
```
