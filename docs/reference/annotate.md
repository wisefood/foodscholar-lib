# Annotation

The NER backends, the dense linker and the batched runner behind
`fs.annotate()`. Concepts are on the [Annotation page](../concepts/annotation.md).

Two NER backends ship. `gliner2` and its described label set come from the
kggen NER/NEL benchmark (see the provenance page); it is
an option rather than the default, for the reasons on the
[Extended KG-Gen](../concepts/extended-kg-gen.md#what-was-not-adopted-as-a-default).

## NER

```{automodule} foodscholar.annotate.gliner_ner
:members:
:member-order: bysource
```

```{automodule} foodscholar.annotate.gliner2_ner
:members:
:member-order: bysource
```

## Linking

```{automodule} foodscholar.annotate.linker
:members:
:member-order: bysource
```

```{automodule} foodscholar.annotate.nel_index
:members:
:member-order: bysource
```

## Runner

```{automodule} foodscholar.annotate.runner
:members:
:member-order: bysource
```
