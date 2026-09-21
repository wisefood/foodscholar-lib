# ner-nel — NER + entity linking experiments

Named-entity recognition and linking (to FoodOn) experiments for the NutriGraphRAG /
Merlin corpora: model comparison, cross-dataset benchmarking, and the corpus-wide
production run.

| File | Purpose |
|---|---|
| `nel_ner_evaluation.ipynb` | NER + NEL evaluation on one passage (Mediterranean diet) with a cross-variant summary. |
| `ner_benchmark_cross_dataset.ipynb` | Cross-dataset NER benchmark (NutriNER) with semantic recall/precision/F1 against GPT-4o-mini. |
| `run_ner_nel_corpus_gliner2_sapbert.py` | Corpus-wide NER + NEL with the selected production defaults. |
| `run_scifoodner_inference.py` | SciFoodNER inference helper, executed as a subprocess by the evaluation notebook. |
| `requirements.txt` | Third-party pins for this folder, matching `python3.12_venv`. |

## `nel_ner_evaluation.ipynb`

Evaluates NER and NEL variants over a single passage and compares them:

- **NER** — GLiNER (`gliner`), GLiNER large v2.1, GLiNER Large Bio v0.1 (with batch/width
  configurations), spaCy, scispaCy, SciFoodNER (via `run_scifoodner_inference.py`),
  followed by a summary table.
- **NEL** — `src.nel` linkers: `LexicalNELLinker` (variant A) and
  `HNSWNELLinker` (variant B) with several encoders — SapBERT, BioLORD, MiniLM, MPNET —
  plus optional cross-encoder reranking.
- **Reranked variants** — FoodSEM-based variants C/D/D+2/E (label-aware ordering, robust
  URI parsing), each with a cross-variant comparison table.

## `ner_benchmark_cross_dataset.ipynb`

Benchmarks NER models across datasets/label sets (`multihop_qa`, `textbook_lyssari`,
`single_abstract`, `mmlu`, `hai_coaching`, `ngqa`): dataset loaders, per-model runner and
checkpointing, SapBERT semantic recall/precision/F1, comparison against GPT-4o-mini
entity extraction over OpenRouter, and heatmap/bar-chart summaries.

## `run_ner_nel_corpus_gliner2_sapbert.py`

The corpus-wide production pipeline (config at the top of the file, no CLI flags):

| Setting | Value |
|---|---|
| NER | GLiNER2 `fastino/gliner2-large-v1`, confidence `0.35`, batch 16 |
| NEL | HNSW + SapBERT, top-1, cosine ≥ `0.70`, no reranker (below the gate → NIL) |
| Index | `<LinearRAG>/ontology/foodon_hnsw_sapbert.bin` + `foodon_metadata.json` |
| Input | all `*.csv` under `CORPUS_DIR` (`chunk_id, chunk_text, type, chunk_metadata`) |
| Output | one `nel_<file>.csv` per input under `OUTPUT_DIR` (`chunk_id, chunk_entities_ner, chunk_uri_nel`) |

Links go through `src.nel.HNSWNELLinker(encoder="sapbert")` so the query encoder matches
the one the HNSW index was built with (SapBERT needs CLS pooling). A fully written output
file is skipped, so re-running resumes where it stopped.

## Requirements & runtime notes

- `requirements.txt` in this folder lists the exact versions installed in
  `python3.12_venv` (`gliner`, `gliner2`, `spacy`, `simpletransformers`, `torch`,
  `transformers`, `sentence-transformers`-based linkers, plotting stack, …).
- The notebooks import the **local** `src.nel` package (`LexicalNELLinker`,
  `HNSWNELLinker`), which is not shipped in this folder — it comes from the wider
  project/module tree, so make sure its root is importable (`sys.path`/`PYTHONPATH`).
- `run_scifoodner_inference.py` is documented to run with a dedicated SciFoodNER
  environment, e.g.
  `/mnt/data/makis/conda_envs/scifoodner/bin/python run_scifoodner_inference.py --input … --output …`.
- NER/NEL runs are GPU-bound; the notebooks select the device themselves and free GPU
  memory between variants.
