# Configuration

Everything about a FoodScholar instance — which stores it talks to, which LLM it
calls, how each layer is built — comes from **one validated config object**. This page
explains the config API, walks through each section with examples, and ends with
copy-paste recipes for the common setups.

## The mental model

- **One object, validated up front.** Your YAML (or dict) is parsed into a
  `FoodScholarConfig` Pydantic model. Every field has a type and a default; unknown
  keys are **rejected** (`extra="forbid"`), so a typo like `stroage:` fails loudly at
  load time instead of being silently ignored.
- **Only `corpus` is required.** Every other section has sensible defaults — omit a
  section to take them.
- **Secrets come from the environment**, never the file. Use `${VAR}` anywhere and it's
  substituted from the environment at load time.

## The config API

`FoodScholar.from_config(...)` accepts **three forms** of the same thing:

```python
from foodscholar import FoodScholar
from foodscholar.config import FoodScholarConfig

# 1. A YAML file path
fs = FoodScholar.from_config("config.yaml")

# 2. A plain dict (great for notebooks and tests)
fs = FoodScholar.from_config({
    "corpus": {"chunks_path": "data/corpus"},
    "storage": {"chunk_store": {"backend": "memory"},
                "graph_store": {"backend": "memory"}},
})

# 3. An already-validated config object (build it in code, reuse it)
cfg = FoodScholarConfig.model_validate({"corpus": {"chunks_path": "data/corpus"}})
fs = FoodScholar.from_config(cfg)
```

`${ENV}` substitution runs over **all three** forms (strings, nested dicts, lists), so
in-code configs can carry placeholders exactly like YAML. Under the hood
`resolve_config()` normalizes any of them to a `FoodScholarConfig`; `load_config(path)`
is the YAML-only helper.

```{tip}
**Tune at runtime.** The config object is plain Pydantic and mutable, so you can poke a
single knob between runs without rewriting the file:

    fs.config.layer_b.pass1_mode = "per_shelf"
    fs.build_layer_b(facet="foods")

This is exactly how the notebook sweeps Layer B settings.
```

## Sections at a glance

| Section | Required | Controls |
|---|---|---|
| `corpus` | **yes** | where chunks live; the offline snapshot path |
| `storage` | no | the chunk store (Elasticsearch/memory) and graph store (Neo4j/memory) |
| `llm` | no | provider + fallback chain (omit → built-in mock) |
| `ontology` | no\* | the FoodOn OWL path + cache (\*required for real annotation/Layer A) |
| `annotate` | no | NER (GLiNER) + embedder + the entity linker |
| `layer_a` | no | shelf projection method + prune/aliasing knobs |
| `layer_b` | no | theme discovery passes, merge, labeling |
| `layer_c` | no | card LLM model + grounding/safety |

## `storage` — where data lives

Each store independently selects a backend. `memory` needs nothing; `elastic` and
`neo4j` point at running services.

```yaml
storage:
  chunk_store:
    backend: elastic            # or: memory
    url: http://localhost:9200
    index: foodscholar_chunks
    # auth (optional): basic creds win over api_key; both fall back to env
    username: elastic
    password: ${ELASTIC_PASSWORD}
    bulk_size: 1000             # docs per _bulk request (1000–5000 is the sweet spot)
  graph_store:
    backend: neo4j              # or: memory
    url: bolt://localhost:7687
    user: neo4j
    password: ${NEO4J_PASSWORD}
```

```{note}
Mix freely — e.g. `chunk_store: elastic` with `graph_store: memory`. The all-`memory`
combination is what `FoodScholar.in_memory()` selects and needs zero services.
ES credentials, if unset here, fall back to `ELASTICSEARCH_API_KEY`; Neo4j's password
falls back to `NEO4J_PASSWORD`.
```

## `llm` — the language model

The LLM powers the (optional) linker LLM step and Layer C cards. Declare a primary
provider and an ordered fallback chain; the chain is tried in order, each entry only if
the earlier ones errored (timeout, rate limit, auth, outage).

```yaml
llm:
  primary:   { provider: groq, model: llama-3.3-70b-versatile }
  fallbacks:
    - { provider: ollama, model: llama3.1, host: http://localhost:11434 }
  timeout_s: 30
  max_retries: 2
```

Providers: `anthropic`, `openai`, `groq`, `gemini`, `ollama`. API keys come from the
environment (`GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`),
or you may set `api_key:` inline (the inline value wins). Ollama needs only a running
daemon. Install providers with the `[llm]` extra.

```{important}
**Omit the `llm:` section entirely** (or use `in_memory()`) and FoodScholar uses a
built-in **mock** client that returns a fixed string. That keeps tests and the
quickstart deterministic and offline — but if you forget to configure a real provider,
your Layer C cards and theme labels will be mock text. Wire the provider in at
construction, not after.
```

## `corpus` & `ontology`

```yaml
corpus:
  chunks_path: data/corpus                       # dir of chunk CSVs, or a file
  annotated_snapshot_path: data/annotated.parquet   # offline snapshot (idempotent reruns)
  ignore_source_types: [abstract]                # drop these source types at ingest
ontology:
  foodon_path: data/foodon.owl
  cache_path: data/foodon_cache.parquet          # Parquet cache beside the OWL
  prefix_filter: ["FOODON:"]                      # keep only FOODON ids; null = keep all
```

`prefix_filter` matters: real FoodOn OWL files embed CHEBI/NCBITaxon/BFO terms inline,
and the default keeps the linker from matching food queries against unrelated
ontologies. See [](../concepts/corpus-input.md) for the chunk/NEL input format.

## `annotate` — NER, embeddings, linking

```yaml
annotate:
  ner: gliner                                   # GLiNER-bio NER (the only strategy)
  gliner:
    model_id: urchade/gliner_large_bio-v0.1
    threshold: 0.4
  embedder: BAAI/bge-base-en-v1.5               # 768-d chunk embeddings
  linker:
    nel_backend: hnsw                            # local hnswlib index (or: elastic)
    nel_encoder: biolord                         # sapbert | biolord | minilm | mpnet
    nel_top_k: 1
    nel_min_sim: 0.70                            # reject links below this cosine
```

See [](../concepts/annotation.md) for how these pieces fit together. You can skip live
annotation entirely by supplying pre-computed NEL CSVs at ingest time.

## `layer_a`, `layer_b`, `layer_c` — building the graph

These expose every construction knob. The most useful ones:

```yaml
layer_a:
  projection: backbone        # "backbone" (1a+, default) or "prune" (fallback)
  alias_shelves: true         # LLM display labels for jargon shelves (additive)
layer_b:
  min_chunks_per_shelf: 50    # skip shelves smaller than this
  pass1_mode: per_shelf       # "per_shelf" (default) or "global"
  leiden: { min_community_size: 15 }   # the main Layer B coverage lever
  labeling: { strategy: llm }          # "keyword" (free) or "llm"
layer_c:
  llm_model: claude-sonnet-4-6
  grounding_check: strict     # reject card claims not grounded in cited chunks
  safety_sensitive_facets: [allergies]
```

Each is explained in depth in the Concepts pages —
[Layer A](../concepts/layer-a-backbone.md), [Layer B](../concepts/layer-b-themes.md),
[Layer C](../concepts/layer-c-cards.md) — and the repo's `config.example.yaml` lists
**every** field with its default and rationale.

## Recipes

### Zero-config, in memory

```python
fs = FoodScholar.in_memory()        # all-memory stores + mock embedder + mock LLM
```

### Offline, from a snapshot (no services)

```yaml
corpus:
  chunks_path: data/corpus
  annotated_snapshot_path: data/annotated.parquet
ontology: { foodon_path: data/foodon.owl, cache_path: data/foodon_cache.parquet }
storage:
  chunk_store: { backend: memory }
  graph_store: { backend: memory }
```

### Full local stack (real ES + Neo4j + Groq)

```yaml
corpus: { chunks_path: data/corpus }
ontology: { foodon_path: data/foodon.owl, cache_path: data/foodon_cache.parquet }
llm:
  primary: { provider: groq, model: llama-3.3-70b-versatile }
storage:
  chunk_store: { backend: elastic, url: http://localhost:9200, index: foodscholar_chunks }
  graph_store: { backend: neo4j, url: bolt://localhost:7687, user: neo4j, password: ${NEO4J_PASSWORD} }
```

```{tip}
Because the config is validated, a misnamed key is caught immediately:
`pydantic_core.ValidationError: ... Extra inputs are not permitted`. Treat that as a
helpful spell-checker for your config.
```
