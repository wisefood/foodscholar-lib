# Data model

The Pydantic contracts that flow through the pipeline. Every store, layer, and renderer
reads and writes these — they are the stable interface between stages.

## Corpus

A `Chunk` is the atom of the corpus; `Mention` and `EntityLink` are the annotation
output that rides on it. See [Corpus input](../concepts/corpus-input.md) for the file
format these are loaded from, and [Annotation](../concepts/annotation.md) for how the
mentions and links are produced.

```{autopydantic_model} foodscholar.io.chunk.Chunk
```

```{autopydantic_model} foodscholar.io.chunk.Mention
```

```{autopydantic_model} foodscholar.io.chunk.EntityLink
```

`ChunkProvenance` is the typed reader-side view of `Chunk.source_metadata`,
reached as `chunk.provenance`. It is permissive by design: the three corpus
producers emit three different key sets (a textbook chunk carries only `file`,
`heading` and `page_number`), and it normalizes the `DOI`/`doi` spelling split
the corpus contains.

```{autopydantic_model} foodscholar.io.chunk.ChunkProvenance
```

## Relations

A `Relation` is the Layer 0 edge: one record per distinct
`(subject_id, predicate, object_id)`, carrying the surfaces that collapsed
into it and the chunks it came from. Endpoints are ontology ids, or `NIL:`
sentinels paired with the `subject_linked` / `object_linked` flags. See
[Layer 0 — Relations](../concepts/layer-0-relations.md).

```{autopydantic_model} foodscholar.io.relation.Relation
```

```{eval-rst}
.. autofunction:: foodscholar.io.relation.make_relation_id
.. autofunction:: foodscholar.io.relation.nil_id
.. autofunction:: foodscholar.io.relation.is_nil
```

## Graph

The nodes of the knowledge graph. A `Shelf` is a Layer A backbone node, a `Theme` is a
Layer B community, and a `Card` is a Layer C write-up. See the layer concept pages
([A](../concepts/layer-a-backbone.md), [B](../concepts/layer-b-themes.md),
[C](../concepts/layer-c-cards.md)).

```{autopydantic_model} foodscholar.io.graph.Shelf
```

```{autopydantic_model} foodscholar.io.graph.Theme
```

```{autopydantic_model} foodscholar.io.graph.Card
```

## Entities & ontology terms

`Entity` is a deduplicated, corpus-aggregated view of a linked ontology id (produced by
`build_entities`); `OntologyTerm` is a single FoodOn class as loaded from the ontology.

```{autopydantic_model} foodscholar.io.entity.Entity
```

```{autopydantic_model} foodscholar.io.ontology.OntologyTerm
```

## Run metadata

```{autopydantic_model} foodscholar.io.artifacts.ArtifactMeta
```
