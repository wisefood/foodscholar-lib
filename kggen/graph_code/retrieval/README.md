# test_retrieval

Hybrid passage retrieval (text similarity 0.3 + mean triplet similarity 0.3 +
Personalized PageRank on the unified k-hop subgraph 0.4) extracted from
`retrieval_passages_ppr_subgraph.ipynb`.

| File | Purpose |
|---|---|
| `retrieval_core.py` | All functions: loading, index building (+ disk cache), embedding building (+ disk cache), scoring branches, hybrid retrieval, `Retriever` class. |
| `run_retrieval.py` | Execution: takes a user query and runs retrieval (`run_retrieval(query)` / CLI). |
| `test_retrieval_run.ipynb` | Notebook that runs the query `What is the relationship between salt consumption above 2 g NaCl/day and arterial blood pressure?` (real answer: `an almost linear positive relationship with arterial blood pressure`) over `../data/graph/_aggregated_all/`. |
| Cache | Indexes (`indexes.pkl`) and embeddings (`*.npy`, ~520 MB) are kept in the shared `../data/cache/` directory (`RetrievalConfig.cache_dir`), not inside this folder. Delete it to force a full rebuild. |

## Run from the command line

```bash
python run_retrieval.py "What is the most common mineral in the body?"
```

Useful flags: `--top-k`, `--gpu`, `--device`, `--batch-size`, `--cache-dir`,
`--rebuild-indexes`, `--rebuild-embeddings`, `--save-json out.json`, `--quiet`,
`--metadata-only`, `--metadata-json out.json`, `--metadata-keys file,title,page_number`,
`--no-scores`.

## Use from Python

```python
from retrieval_core import RetrievalConfig, Retriever

retriever = Retriever(RetrievalConfig(gpu_index=1)).warm_up()
result = retriever.retrieve("What is the most common mineral in the body?")
print(result.passage_ids, result.score_breakdown)
```

> **Always pass `config=` to the `run_retrieval_*` helpers.** `RetrievalConfig` defaults to
> `kg_output_chunks/_aggregated_all/...`, which is not where this checkout keeps its data;
> the notebook passes `Path("../data/graph/_aggregated_all/…")` and
> `Path("../data/cache")`. The process-wide retriever is keyed by that config — asking for
> a different one rebuilds it instead of silently reusing the cached instance.

## Retrieving the passage metadata

Metadata of the retrieved passages is available through dedicated accessors that do
not require the passage texts:

```python
# 1. from an existing result
result.get_metadata(keys=("file", "heading", "page_number"), include_scores=True)
result.metadata_json(keys=("file", "page_number"))

# 2. run a query and get only the metadata
retriever.retrieve_metadata("What is the most common mineral in the body?")

# 3. look up metadata for arbitrary passage ids (no encoder / GPU needed)
retriever.get_metadata(["826550f0-d269-472c-b25e-6c7c3600e6e9"])
retriever.get_metadata_index()          # {passage_id: metadata} for all 4619 passages

# 4. from the execution module
from run_retrieval import run_retrieval_metadata
run_retrieval_metadata("What is the most common mineral in the body?", verbose=False)
```

CLI equivalents:

```bash
python run_retrieval.py "What is the most common mineral in the body?" \
    --metadata-only --metadata-keys file,heading,page_number --metadata-json meta.json
```

`--no-scores` drops the score fields from the metadata output. Every entry is a flat
dict: `rank`, `passage_id`, the metadata fields (`file`, `urn`, `pdf_name`, `country`,
`title`, `audience`, `pages`, `heading`, `page_number`) and, by default, the scores.

### Metadata stored on the graph's passage nodes

The aggregated graph stores every passage as a node with three attributes: `type`
(`'passage'`), `text` and `metadata` (a JSON string in the GraphML file, parsed into a
dict by `build_indexes`). These are cached together with the indexes, so they can be
read without touching `passages.json`:

```python
result.get_graph_metadata(keys=("file", "page_number"), include_scores=True)
result.get_graph_metadata(include_graph_attrs=True)   # + graph_type / graph_text
result.graph_metadata_json(keys=("file", "page_number"))
result.passage_graph_attrs                            # raw node attribute dicts
result.passage_graph_attrs[0]["metadata"]             # the node's metadata dict

retriever.retrieve_graph_metadata(query)              # query -> graph metadata only
retriever.get_graph_metadata(passage_ids)             # ids -> graph metadata
retriever.graph_metadata_index()                      # {passage_id: metadata} of all 4619 nodes

from run_retrieval import run_retrieval_graph_metadata
run_retrieval_graph_metadata(query, verbose=False)
```

```bash
python run_retrieval.py "What is the most common mineral in the body?" \
    --metadata-only --metadata-source graph
```

The graph metadata is identical to the `passages.json` metadata for all 4619 passages
(verified), so the two sources can be used interchangeably.

## Caching

* Indexes and embeddings are loaded from the configured cache directory
  (`../data/cache/` for the notebook) when a matching cache exists.
* On the first run they are created (the embeddings take a few minutes on GPU)
  and written there, so later queries / later processes only load them
  (~16 s end-to-end, retrieval itself ~6 s).
* The cache is invalidated automatically when the graph file, the passage count
  or the encoder model changes.
* If the GPU is short on memory, the embedding batch size is halved
  automatically and the encoder falls back to the CPU as a last resort.
