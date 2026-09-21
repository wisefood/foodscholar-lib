# Installation

FoodScholar targets **Python 3.11**. The reference environment is a conda env
named `foodscholar`; the project's test and docs tooling assume it.

```bash
conda create -n foodscholar python=3.11 -y
conda activate foodscholar
pip install -e '.[dev]'
```

```{warning}
Use the `foodscholar` env (Python 3.11) for tests and builds. A `base` env with an
older NumPy on a newer Python can fail to import NumPy (and anything that depends on
it). If you see `Error importing numpy: you should not try to import numpy from its
source directory`, you're almost certainly in the wrong interpreter.
```

## Extras

The core install is light. Heavier capabilities are opt-in via
[extras](https://peps.python.org/pep-0508/#extras):

| Extra | Pulls in | Needed for |
|---|---|---|
| `dev` | pytest, ruff, mypy, … | development & tests |
| `ontology` | pronto | loading FoodOn from OWL |
| `annotate` | torch, transformers, gliner, gliner2, hnswlib, … | live NER + the dense HNSW linker (also needed by `build_relations`) |
| `llm` | anthropic, openai, groq, google-genai, ollama | Layer C cards, Layer 0 extraction, theme labels |
| `relations` | semhash, inflect | Layer 0 dedup (semantic + plural collapsing) |
| `chunking` | docling, docling-core, nltk | `fs.chunk_documents` / `chunk-corpus` — PDFs → corpus CSVs |
| `elastic` | elasticsearch | the Elasticsearch chunk / entity / relation stores |
| `neo4j` | neo4j | the Neo4j graph store |
| `clustering` | leidenalg, python-igraph, scikit-learn, … | Layer B community detection |
| `bertopic` | bertopic | the BERTopic Layer B method |
| `summarization` | sumy, nltk | Layer C extractive stage |
| `viz` | pyvis, graphviz, matplotlib | `fs.viz` renderers |

Combine as needed, e.g. a full local stack:

```bash
pip install -e '.[dev,ontology,annotate,llm,elastic,neo4j,clustering,viz]'
# add relations,chunking for Layer 0 and for producing a corpus from PDFs
```

`chunking` is the heavy one — Docling pulls layout models. Every extra is lazily
imported, so a missing one fails with an `ImportError` naming the extra to install,
never at import time.

```{tip}
Zero extras are required to get started — `FoodScholar.in_memory()` runs entirely
on in-memory stores with a mock embedder and mock LLM. See [](quickstart.md).
```

## Backing services (optional)

The Elasticsearch and Neo4j stores expect local services. A `docker-compose.yaml`
in the repo root brings them up:

```bash
docker compose up -d elasticsearch neo4j
```

API keys for LLM providers come from the **environment**
(`GROQ_API_KEY`, `ANTHROPIC_API_KEY`, …), never from a config file. See
[](configuration.md).
