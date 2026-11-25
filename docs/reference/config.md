# Configuration models

The full `FoodScholarConfig` schema — every section, field, type, and default. For an
example-first walkthrough and recipes, see the
[Configuration guide](../getting-started/configuration.md); this page is the exhaustive
field reference, generated from the models themselves.

## Top level

```{eval-rst}
.. autopydantic_model:: foodscholar.config.FoodScholarConfig
```

## Loading

```{eval-rst}
.. autofunction:: foodscholar.config.resolve_config
```

```{eval-rst}
.. autofunction:: foodscholar.config.load_config
```

## Storage

```{eval-rst}
.. autopydantic_model:: foodscholar.config.StorageConfig
```

```{eval-rst}
.. autopydantic_model:: foodscholar.config.ChunkStoreConfig
```

```{eval-rst}
.. autopydantic_model:: foodscholar.config.GraphStoreConfig
```

## LLM

```{eval-rst}
.. autopydantic_model:: foodscholar.config.LLMConfig
```

```{eval-rst}
.. autopydantic_model:: foodscholar.config.ProviderConfig
```

## Corpus & ontology

```{eval-rst}
.. autopydantic_model:: foodscholar.config.CorpusConfig
```

```{eval-rst}
.. autopydantic_model:: foodscholar.config.OntologyConfig
```

## Annotation

```{eval-rst}
.. autopydantic_model:: foodscholar.config.AnnotateConfig
```

```{eval-rst}
.. autopydantic_model:: foodscholar.config.GLinerConfig
```

```{eval-rst}
.. autopydantic_model:: foodscholar.config.GLiner2Config
```

```{eval-rst}
.. autopydantic_model:: foodscholar.config.LinkerConfig
```

## Layer A

```{eval-rst}
.. autopydantic_model:: foodscholar.config.LayerAConfig
```

## Layer B

```{eval-rst}
.. autopydantic_model:: foodscholar.config.LayerBConfig
```

```{eval-rst}
.. autopydantic_model:: foodscholar.config.SimilarityConfig
```

```{eval-rst}
.. autopydantic_model:: foodscholar.config.RelatednessConfig
```

```{eval-rst}
.. autopydantic_model:: foodscholar.config.LeidenConfig
```

```{eval-rst}
.. autopydantic_model:: foodscholar.config.MergeConfig
```

```{eval-rst}
.. autopydantic_model:: foodscholar.config.LabelingConfig
```

## Layer C

```{eval-rst}
.. autopydantic_model:: foodscholar.config.LayerCConfig
```

## Chunking

Corpus chunking — see the [chunking guide](../guides/chunking-a-corpus.md). The
`chunking:` section is `foodscholar.chunking.config.ChunkingConfig`, documented
with the chunker so it loads without this module: see the
[chunking reference](chunking.md#configuration). `foodscholar.config.ChunkerConfig`
is the same class under its former name.

```{eval-rst}
.. autopydantic_model:: foodscholar.config.CorpusValidationConfig
```

## Relations (Layer 0)

Typed relations — see [Layer 0](../concepts/layer-0-relations.md).

```{eval-rst}
.. autopydantic_model:: foodscholar.config.RelationsConfig
```

```{eval-rst}
.. autopydantic_model:: foodscholar.config.RelationGroundingConfig
```

```{eval-rst}
.. autopydantic_model:: foodscholar.config.RelationDedupeConfig
```

```{eval-rst}
.. autopydantic_model:: foodscholar.config.RelationStoreConfig
```
