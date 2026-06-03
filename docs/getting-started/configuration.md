# Configuration

FoodScholar is configured from a single YAML file (or a Python dict / validated
`FoodScholarConfig`). `config.example.yaml` in the repo root is the annotated
reference; this page covers the parts you touch first.

```python
from foodscholar import FoodScholar

fs = FoodScholar.from_config("config.yaml")
```

## Storage backends

Each store independently selects a backend. `memory` needs nothing; `elastic` and
`neo4j` point at running services.

```yaml
storage:
  chunk_store:
    backend: elastic        # or: memory
    url: http://localhost:9200
    index: foodscholar_chunks
  graph_store:
    backend: neo4j          # or: memory
    url: bolt://localhost:7687
    user: neo4j
    password: ${NEO4J_PASSWORD}    # ${VAR} pulls from the environment
```

```{tip}
Use `${VAR}` substitution for anything secret. Values are read from the environment
at load time so credentials never live in the file.
```

## The LLM

The LLM linker tier and Layer C card generation use a provider-agnostic client.
Declare a primary provider and an ordered fallback chain:

```yaml
llm:
  primary:   { provider: groq, model: llama-3.3-70b-versatile }
  fallbacks:
    - { provider: ollama, model: llama3.1 }
  timeout_s: 30
```

Providers: `anthropic`, `openai`, `groq`, `gemini`, `ollama`. The primary is tried
first; fallbacks are tried in order if it errors (timeout, rate limit, auth, outage).
API keys come from the environment (`GROQ_API_KEY`, `ANTHROPIC_API_KEY`, …) — never
the config file. Install providers with the `[llm]` extra.

```{note}
`FoodScholar.in_memory()` and any config with no `llm:` section use a built-in mock
client, so the test suite and quickstart stay deterministic and offline.
```

## Corpus & ontology

```yaml
corpus:
  chunks_path: data/corpus
  annotated_snapshot_path: data/annotated.parquet   # offline in-memory snapshot
ontology:
  foodon_path: data/foodon.owl
  cache_path: data/foodon_cache.parquet             # Parquet cache beside the OWL
  prefix_filter: ["FOODON:"]                         # restrict to a prefix, or null for all
```

The ontology loads lazily on first access to `fs.ontology` and caches to Parquet so
subsequent runs skip the OWL parse.

## Layer-specific knobs

Layer A (`layer_a:`) and Layer B (`layer_b:`) expose their construction parameters —
projection method, prune thresholds, the Pass-1/Pass-2 similarity and relatedness
settings, labeling strategy, and audit gates. These are documented in depth in the
Concepts and Guides sections (coming soon); `config.example.yaml` lists every field
with its default and rationale.
