# Annotation — NER & linking

Before any layer is built, each chunk is **annotated**: GLiNER finds food/health
mentions, a dense linker resolves each to a FoodOn ID, and an embedder produces the
chunk vector. You can run this live with `fs.annotate()`, or skip it entirely by
supplying [pre-computed NEL CSVs](corpus-input.md) at ingest time.

```{mermaid}
flowchart LR
    Ch[Chunk text] -->|GLiNER NER| Me[Mentions]
    Me -->|encode + HNSW kNN| Li[FoodOn entity links]
    Ch -->|BGE-base| Em[768-d embedding]
```

```python
fs.ner.extract("Mediterranean diet rich in olive oil.")   # -> list[Mention]
fs.linker.dry_run("oliv oil")
# EntityLink(ontology_id="FOODON:...", method="dense", confidence=0.94)
fs.annotate()                                              # full phase over the store
```

## NER — GLiNER

Mentions are found by **GLiNER-bio** (`urchade/gliner_large_bio-v0.1`), a zero-shot
biomedical NER model. It labels spans across the project's entity types (food,
nutrient, health, dietary pattern, allergen, population, biomarker, processing) and is
configured by `config.annotate.gliner`:

```yaml
annotate:
  ner: gliner
  gliner:
    model_id: urchade/gliner_large_bio-v0.1
    threshold: 0.4          # minimum span score to keep
    flat_ner: true
    batch_size: 16
```

```{note}
GLiNER is the only NER strategy in the current pipeline. Earlier `keyword` (deterministic
ontology matching) and `agentic` (LLM-extracted) strategies were removed when the
library standardized on GLiNER + a dense HNSW linker.
```

## Linking — dense nearest neighbour

Each mention is resolved to a FoodOn ID by **dense retrieval**, not lexical matching:

1. Every FoodOn term (label + synonyms) is embedded by the configured biomedical
   encoder and indexed in an **HNSW** graph (built on first use from the loaded
   ontology and cached to disk).
2. A mention's surface form is embedded with the same encoder and matched against the
   index by cosine kNN.
3. The top hit is accepted as an `EntityLink` if its cosine ≥ `nel_min_sim`; otherwise
   the mention is left unlinked.

```yaml
annotate:
  linker:
    nel_backend: hnsw        # local hnswlib index (default) or: elastic (ES dense_vector)
    nel_encoder: biolord     # biolord (default) | sapbert | minilm | mpnet
    nel_top_k: 1
    nel_min_sim: 0.70        # reject links below this cosine
```

The biomedical encoder (BioLORD by default) places synonyms close in vector space, so
`ascorbate → vitamin C` and `whole grains → whole grain` link without any lexical
overlap. Each `EntityLink` records its `method` and `confidence`, so it stays auditable,
and downstream stages (Layer A support, Layer B relatedness) can require a minimum
confidence.

```{tip}
Linking is the expensive part. Supplying NEL CSVs at ingest (`fs.ingest(dir, nel_dir=...)`)
skips GLiNER and the HNSW build entirely — fast, deterministic, and offline. The
`in_memory()` quickstart and the unit tests use that path.
```

## Catching linker drift

Some surface forms are polysemous in a way a generic encoder gets wrong — e.g. "fish"
in food prose means fish-as-food, but the upstream linker can pair it with the FoodOn
class for aquarium feed. A small **link blocklist** (`config.layer_a.link_blocklist`)
filters known `(surface, ontology_id)` drift before Layer A collects support, so those
mislinks never inflate a shelf.

## Quality gate

Entity-linking coverage on a held-out gold set is a **unit test** that fails CI if the
linker regresses below threshold — annotation quality is treated as a contract, not a
best effort.
