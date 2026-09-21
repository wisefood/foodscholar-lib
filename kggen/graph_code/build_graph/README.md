# build_graph — knowledge-graph construction

Turns the chunk corpora into **passage-aware** knowledge graphs with `kggen_extended`
(every triplet keeps the passage(s) it came from), aggregates them and exports them.

| File | Purpose |
|---|---|
| `extract_triplets_from_chunks.py` | One chunks CSV → one `ExtendedGraph` per chunk → aggregate → dedup → exports. CLI: `--csv`, `--out-dir`, `--model`, `--max-texts`, `--list-models`. |
| `aggregate_graphs.py` | Combine several graph JSONs into one → dedup → exports. CLI: `--graphs a.json b.json …` or `--graphs-dir DIR`, `--out-dir`, `--model`, `--list-models`. |
| `run_all_kggen.sh` | Orchestrator: per-guide extraction → per-textbook extraction → per-corpus aggregation → grand total (`_aggregated_all`). |
| `run_abstracts_clusters_kggen.sh` | Same, but only for `../data/chunks/abstracts/clusters/*.csv`; optional `--aggregate-all`. |

## Run it

```bash
cd /mnt/data/wisefood/foodscholar/graph_code/build_graph

./run_all_kggen.sh --dry-run                  # print the commands without running them
./run_all_kggen.sh                            # default model
./run_all_kggen.sh --model "llama3.1:8b-instruct-fp16" --verbose
./run_all_kggen.sh --textbooks-only --skip-existing --log-dir ./logs

./run_abstracts_clusters_kggen.sh --aggregate-all
```

or a single file at a time:

```bash
python extract_triplets_from_chunks.py --csv ../data/chunks/guides/chunks_guide_ie-key-messages.csv \
    --out-dir ../data/graph/guides/ie-key-messages --model "mistral-small3.2:24b-instruct-2506-q8_0"
python aggregate_graphs.py --graphs-dir ../data/graph/guides --out-dir ../data/graph/_aggregated_all
```

Both runners read `../data/chunks/**` and write to `../data/graph/**`, so they must be
started from `build_graph/`. They invoke Python as
`conda run --no-capture-output -n ${KGEN_CONDA_ENV:-python3.12_venv}`.

Common flags: `--model/-m`, `--dry-run`, `--skip-existing` (skip units whose
`aggregated_graph.json` already exists), `--conda-env`, `--verbose/-v`, `--log-dir DIR`;
`run_all_kggen.sh` adds `--guides-only` / `--textbooks-only`.

## Models

`--list-models` prints the catalogue; the default is
`mistral-small3.2:24b-instruct-2506-q8_0`. Each entry declares its `provider`
(`ollama`, `gpustack`, `openrouter`) and is routed through LiteLLM with the matching
API base/key (`OLLAMA_API_BASE`, `GPUSTACK_API_BASE`, `OPENROUTER_API_BASE`).

## Output per processed unit

```
<out-dir>/
├── chunk_graphs/<chunk_id>.json   # one graph per chunk (also the resume savepoints)
├── aggregated_graph.json          # combined ExtendedGraph (full provenance)
├── aggregated_graph.graphml       # passage nodes + Source edges, for the retrieval step
├── triples.csv / triples.jsonl
├── passages.csv / passages.json
└── metrics.json
```

Re-running an extraction resumes: chunks that already have a savepoint in
`chunk_graphs/` are skipped, and `--skip-existing` skips whole units.

## Note on imports

`kggen_extended` is a **local** package (see `../kggen_extended/`) and is not
pip-installed. Both scripts therefore put `graph_code/` — their parent directory — on
`sys.path` before importing it, so no installation step is needed.
