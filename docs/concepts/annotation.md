# Annotation — NER & linking

Before any layer is built, each chunk is **annotated**: named-entity recognition finds
food/health mentions, and a tiered linker resolves them to FoodOn IDs. You can supply
this as [pre-computed NEL CSVs](corpus-input.md) or run it live with `fs.annotate()`.

```python
fs.ner.extract("Mediterranean diet rich in olive oil.")   # -> list[Mention]
fs.linker.dry_run("oliv oil")
# EntityLink(ontology_id="FOODON:...", method="lexical_fuzzy", confidence=0.94)
fs.annotate()                                              # full phase over the store
```

## NER strategy

`config.annotate.ner` selects how mentions are found:

- **`keyword`** (default) — a deterministic word-boundary matcher over every FoodOn
  label + synonym. No LLM, no model download, fully offline. The safe default used by
  `in_memory()` and the tests.
- **`agentic`** — an LLM extracts mentions and classifies each as food / nutrient /
  health / dietary_pattern / allergen / population / biomarker / processing. Character
  offsets are recomputed locally, never trusted from the model. Needs the `[llm]`
  extra and a configured provider.

```{note}
There is no bespoke fine-tuned NER model — the project deliberately uses an LLM rather
than a proprietary model for the `agentic` path.
```

## The linker tiers

The linker is a 3-or-4 tier cascade — the **first confident hit wins**:

| Tier | `method` | What it does | Enabled |
|---|---|---|---|
| 1 | `lexical_exact` | exact, case/punctuation-insensitive label or synonym match | always |
| 2 | `lexical_fuzzy` | rapidfuzz `WRatio` over labels + synonyms | always |
| 3 | `dense` | cosine kNN over SapBERT term embeddings | when `linker.dense_model` is set |
| 4 | `llm` | LLM picks from the top-k candidates, or rejects | when `linker.llm_select: true` |

Tiers 1–2 are pure-lexical and need no models. Tier 3 (dense, SapBERT) catches
lexically-distinct synonyms — `ascorbate → vitamin C`, `whole grains → whole grain` —
but **not** opaque abbreviations (`EVOO` and `olive oil` are far apart in SapBERT
space). Tier 4 (LLM) adjudicates the hard residue: abbreviations, and queries lexical
matching can't separate from a food (`iron deficiency` vs. `flat iron steak`). Tier 4
only fires below a confidence threshold.

Each `EntityLink` records its `method` and `confidence`, so it's always auditable which
tier resolved a mention — and downstream stages (Layer A support, Layer B relatedness)
can require a minimum confidence.

```{tip}
The Pydantic defaults keep tiers 3–4 **off**, so `in_memory()` and the test suite stay
deterministic and offline. `config.example.yaml` turns all four on. Install the dense
and LLM tiers with the `[llm]` extra (and a dense model for tier 3).
```

## Quality gate

Entity-linking coverage on a held-out gold set is a **unit test** that fails CI if the
linker regresses below threshold — annotation quality is treated as a contract, not a
best effort.
