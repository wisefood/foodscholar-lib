# Configuration models

The full `FoodScholarConfig` schema — every section, field, type, and default. For an
example-first walkthrough and recipes, see the
[Configuration guide](../getting-started/configuration.md); this page is the exhaustive
field reference, generated from the models themselves.

## Top level

```{autopydantic_model} foodscholar.config.FoodScholarConfig
```

## Loading

```{autofunction} foodscholar.config.resolve_config
```

```{autofunction} foodscholar.config.load_config
```

## Storage

```{autopydantic_model} foodscholar.config.StorageConfig
```

```{autopydantic_model} foodscholar.config.ChunkStoreConfig
```

```{autopydantic_model} foodscholar.config.GraphStoreConfig
```

## LLM

```{autopydantic_model} foodscholar.config.LLMConfig
```

```{autopydantic_model} foodscholar.config.ProviderConfig
```

## Corpus & ontology

```{autopydantic_model} foodscholar.config.CorpusConfig
```

```{autopydantic_model} foodscholar.config.OntologyConfig
```

## Annotation

```{autopydantic_model} foodscholar.config.AnnotateConfig
```

```{autopydantic_model} foodscholar.config.GLinerConfig
```

```{autopydantic_model} foodscholar.config.LinkerConfig
```

## Layer A

```{autopydantic_model} foodscholar.config.LayerAConfig
```

## Layer B

```{autopydantic_model} foodscholar.config.LayerBConfig
```

```{autopydantic_model} foodscholar.config.SimilarityConfig
```

```{autopydantic_model} foodscholar.config.RelatednessConfig
```

```{autopydantic_model} foodscholar.config.LeidenConfig
```

```{autopydantic_model} foodscholar.config.MergeConfig
```

```{autopydantic_model} foodscholar.config.LabelingConfig
```

## Layer C

```{autopydantic_model} foodscholar.config.LayerCConfig
```
