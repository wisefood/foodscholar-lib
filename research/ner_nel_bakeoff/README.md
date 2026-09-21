# NER / NEL bake-off

The gate for flipping `annotate.ner` and `annotate.linker.nel_encoder` to the
values the kggen benchmark selected. **Nothing here ships** — it is provenance
and a decision harness, like `research/bakeoff/`.

## Why this exists

The kggen README states GLiNER2 + SapBERT are the selected production
defaults. Reading the saved notebook outputs, the evidence is thinner than
that and the two notebooks disagree:

| notebook | scale | ground truth | winner |
|---|---|---|---|
| `nel_ner_evaluation.ipynb` cell 15 | **one passage**, 39 entities | human-curated | **GLiNER biomed-v0.1 — F1 0.909** (what we ship); GLiNER2 absent from the table |
| `ner_benchmark_cross_dataset.ipynb` cell 33 | 6 datasets | **GPT-4o-mini proxy** | **gliner2_custom — F1 0.729** |

Cross-dataset means, verbatim from cell 33:

| model | Recall | Precision | F1 | entities/Q |
|---|---|---|---|---|
| `gliner2_custom` | 0.709 | **0.610** | **0.729** | 6.06 |
| `gliner_custom_flat` ← *what foodscholar ships* | **0.806** | 0.435 | 0.630 | 8.45 |
| `gliner_custom_nested` | 0.905 | 0.397 | 0.618 | 10.97 |
| `en_core_sci_lg` | 0.764 | 0.319 | 0.526 | 14.68 |
| `scifoodner_cafeteria` | 0.146 | 0.753 | 0.511 | 1.39 |

GLiNER2 wins **against an LLM proxy**, by trading recall for precision:
−12% recall, +40% precision, **~28% fewer mentions per passage**.

That is not free for foodscholar. Layer A shelf support counts,
`min_chunks_per_shelf`, the Layer B relatedness graph (`tau_strict`,
`min_shared_ids`) and `Entity.chunk_count` all key off mention volume.

The SapBERT claim is weaker still. Linking the same ground-truth entities
(notebook cells 25 and 28):

| entity | SapBERT | BioLORD |
|---|---|---|
| `carrots` | `FOODON_03316751` *carrot (quick frozen)* — **wrong specificity** | `FOODON_00001687` *carrot food product* |
| `HbA1c` | NIL — **correct** | `CHEBI_35143` *hemoglobin* — **wrong** |
| `LDL cholesterol` | `CHEBI_39026` @0.799 | `CHEBI_39026` @0.943 |
| `calcium` | `CHEBI_22985` *calcium molecular entity* | `CHEBI_35156` *calcium salt* |

The selection criterion was "highest link rate + URI diversity" — but **link
rate is not accuracy**. `carrots -> carrot (quick frozen)` is a link *and* an
error, and it is exactly the failure `config.LinkBlocklistEntry` already
exists to patch (`fish` -> aquarium fish food).

## What the bake-off must measure

Four cells (2 NER × 2 encoder). Report **all** of the following — the upstream
metrics can improve while the downstream ones regress, and catching that is
the entire point.

**Annotation level**
- mentions per chunk (mean and total), link rate, distinct linked URIs
- a **hand-scored precision sample**: 100 random (surface, URI) pairs per cell

**Downstream — what the original benchmark could not see**
- **Layer A**: shelf count, shelves clearing `min_chunks_per_shelf`, max depth,
  and the diff of the active shelf set against the current build
- **Layer B**: themes per facet, mean relatedness-graph edge count
- **Entities**: total records, and how many carry a `facet_hint` — this should
  *rise* with GLiNER2's richer vocabulary; if it does not, the mapping table in
  `io/chunk.GLINER2_TO_ENTITY_TYPE` is wrong

## Running it

```bash
conda activate foodscholar
python research/ner_nel_bakeoff/run_bakeoff.py --config config.yaml \
    --sample 2000 --out research/ner_nel_bakeoff/results
```

The first `sapbert` run rebuilds the HNSW index from scratch (the cached index
path is derived from the encoder name, so the two coexist on disk). That takes
minutes — it has not hung.

## Deciding

Flip a default only in a follow-up commit citing these numbers. **If a switch
wins upstream and loses downstream, it ships as an option and the default
stands.** That is a result, not a failure.

> **Note:** `kggen/` is no longer tracked in Git — it is a fork of an external
> project whose licence is not reproduced, so it is kept locally rather than
> redistributed. Paths below refer to that local reference snapshot; ask the
> WiseFood team if you need it.

## Provenance

The original notebooks live in `kggen/graph_code/ner-nel/`:
`nel_ner_evaluation.ipynb`, `ner_benchmark_cross_dataset.ipynb`, and
`run_scifoodner_inference.py` (needs its own `scifoodner` conda env).
