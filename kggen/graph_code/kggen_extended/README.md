# kggen_extended — passage-aware knowledge-graph generation

An extension of the KGGen library that adds **passage-level provenance**: every triplet
records the passage(s) it was extracted from, and after deduplication a triplet can map
to several passages (when aliases from different passages collapse onto the same
canonical entity/triplet).

```python
from kggen_extended import (
    ExtendedGraph, ExtendedKGGen, DeduplicateMethod,
    export_all, export_graphml, export_triples, export_passages,
    to_nx_extended, to_nx_extended_with_passage_ids,
)
```

## Modules

| Module | Role |
|---|---|
| `models.py` | `ExtendedGraph`: `entities`, `relations`, `triplet_passages`, `entity_clusters`; helpers `_triple_to_key`, `_key_to_triple`, `_loose_normalize`. |
| `kg_gen_extended.py` | `ExtendedKGGen` — the orchestrator: `extract()`, `aggregate()`, `deduplicate()`, `visualize()`. |
| `export.py` | `export_graphml()` (passage nodes + `Source` edges), `export_triples()` (CSV/JSONL), `export_passages()` (CSV/JSON), `to_nx_extended[_with_passage_ids]()`, `export_all()`. |
| `steps/_1_get_entities.py` | Step 1 — entity extraction (`get_entities`, dspy signatures with a LiteLLM fallback). |
| `steps/_2_get_relations.py` | Step 2 — relation extraction (`get_relations`, response parsing, entity filtering). |
| `steps/_3_deduplicate.py` | Step 3 — `DeduplicateMethod` enum and `run_deduplication` dispatch. |
| `utils/chunk_text.py` | Sentence-boundary chunking (NLTK), with resource bootstrap. |
| `utils/deduplicate.py` | Semhash dedup with passage-aware entity/triplet remapping. |
| `utils/llm_deduplicate.py` | `LLMDeduplicate` — KMeans clusters + in-cluster LM deduplication. |
| `utils/visualize_kg.py` | High-fidelity visualisation of the extended graph. |
| `prompts/entities.txt`, `prompts/relations.txt` | Prompt templates loaded by the steps. |

## How it is used

Only from `../build_graph/`:

1. `extract_triplets_from_chunks.py` builds one `ExtendedGraph` per chunk
   (`passage_id` = chunk UUID), aggregates them and deduplicates with
   `DeduplicateMethod.SEMHASH` (threshold 0.95).
2. `aggregate_graphs.py` merges the per-document graphs and exports everything to
   `../data/graph/...`.

```python
from kggen_extended import ExtendedKGGen, DeduplicateMethod

kg = ExtendedKGGen(model="ollama_chat/mistral-small3.2:24b-instruct-2506-q8_0", temperature=0.0)
graph = kg.extract(text="…", passage_ids=["chunk-1"])
combined = kg.aggregate([graph])
deduped = kg.deduplicate(combined, method=DeduplicateMethod.SEMHASH, semhash_similarity_threshold=0.95)
```

## Notes

- **Not installed as a package.** It is imported from `graph_code/` (its parent), which
  the `build_graph/` scripts put on `sys.path`; do the same (or add it to `PYTHONPATH`)
  when importing it from elsewhere.
- Dependencies: `dspy`, `litellm`, `pydantic`, `networkx`, `pandas`, `numpy`,
  `sentence-transformers`, `scikit-learn`, `scipy`, `semhash`, `inflect`, `rank-bm25`,
  `typing-extensions` — all pinned in `../requirements.txt`.
- LLM access is provider-agnostic via LiteLLM (`ollama`, `gpustack`, `openrouter`); the
  model/API base are chosen in the calling script.
