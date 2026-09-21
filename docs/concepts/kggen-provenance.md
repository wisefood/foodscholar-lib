# Provenance: the kggen pipeline

Layer 0, the corpus chunker and the GLiNER2 option did not start life in
FoodScholar. They are adapted from **kggen** — a pipeline developed by
colleagues on the WiseFood project for building passage-aware knowledge graphs
over the same nutrition corpus: an extension of KGGen that keeps the passage
each triple came from, a hybrid retriever over that graph, an NER/NEL benchmark,
and the Docling chunking that produced the corpus in the first place.

The original is vendored, unmodified apart from credential removal, at
`kggen/graph_code` in the repository. It is a reference snapshot: nothing
imports from it and it is not on the package path. Its own `README.md` and the
six sub-READMEs describe it on its own terms, and are the right place to start
if you want to see the source rather than the port.

## What FoodScholar took

| in FoodScholar | from kggen | what changed |
|---|---|---|
| The two extraction prompts (`relations/prompts/*.txt`) | `kggen_extended/prompts/` | **Copied verbatim.** They are the benchmarked artifact; treat them as fixtures. |
| Entity and relation extraction (`relations/extract.py`) | `kggen_extended/steps/_1_get_entities.py`, `_2_get_relations.py` | Ported from dspy/LiteLLM onto FoodScholar's `LLMClient`. The enum constraint on relation endpoints — subject and object must come from the step-1 entity list — is kept; it is what keeps triples aligned with entities. |
| Passage provenance (`relations/extracted.py`) | `kggen_extended/models.py` (`ExtendedGraph`) | Slim rewrite, extractor-internal only. The property that matters — a triple keeps every passage it was extracted from, and collapsing triples **union** their passages — is preserved. The upstream validator's stale-key repair (for savepoint files this library does not have) is not. |
| Surface and predicate dedup (`relations/dedupe.py`) | `kggen_extended/utils/deduplicate.py` | Ported: NFKC normalization, per-token singularization, semantic hashing. Made deterministic — the original iterated a `set`, so its canonical pick varied with `PYTHONHASHSEED`. |
| The chunking window (`corpus/window.py`) and PDF producer | `chunking/chunking_pipeline_{guides,textbooks}_overlap.ipynb` | Three notebooks turned out to share one algorithm; it is written once, dependency-free, and the parameters that produced the existing corpus are pinned. The abstracts notebook's sentence splitting is the same window over NLTK sentences. |
| GLiNER2 and its 27 described labels (`annotate/gliner2_ner.py`) | `ner-nel/run_ner_nel_corpus_gliner2_sapbert.py` and the two benchmark notebooks | Label set copied verbatim; shipped as an **option**, not the default — see below. |

**Grounding** (`relations/ground.py`) is the one piece with no kggen
counterpart, and it is the join between the two systems. kggen's entities are
free-text strings; FoodScholar's are ontology ids. Every relation endpoint is
resolved through FoodScholar's own linker, so a kggen-style triple becomes an
edge between the same `Entity` records Layer A projects and Layer B clusters.
Without it there would be two disjoint entity universes in one graph.

## What was not adopted as a default

The kggen README presents GLiNER2 and the SapBERT linking encoder as its
selected production defaults. Read from the notebooks' saved outputs, the
evidence is mixed: on the single-passage human-scored evaluation the model
FoodScholar already ships (GLiNER-bio) scores highest; GLiNER2's win is on a
cross-dataset benchmark scored against a GPT-4o-mini proxy, where it trades
about 12% recall for about 40% precision and yields roughly 28% fewer mentions
per passage. FoodScholar's Layer A support counts and Layer B relatedness graph
key off mention volume, so that trade is not free here. Both ship as
configurable options, and `research/ner_nel_bakeoff/` in the repository holds
the harness — and the published numbers, verbatim — that gates any change to
the defaults.

## Not ported

- **Hybrid retrieval** (`retrieval/retrieval_core.py`: text similarity + mean
  triplet similarity + Personalized PageRank over the entity subgraph).
  Deferred; `RelationStore.for_chunks` exists specifically for its triplet
  branch, and `fs.query()` is where it will land.
- **Abstracts semantic clustering** (`chunking_abstracts_split_check.ipynb`,
  sections 7–11) — partitions the already-chunked corpus into files; disk
  organization rather than chunking.
- **LLM page triage** (`utils/pdf_page_triage.py`) — produces the
  excluded-pages manifest the chunker consumes; stays a corpus-prep script.
- `export.py`, `visualize_kg.py`, `chunk_text.py` — covered by
  `io/graphml.py`, `viz/` and the ≤512-token chunk contract respectively.
- `utils/llm_deduplicate.py` — deferred in favour of reusing
  `layer_a/semantic_consolidation/`.

## Crediting

If you build on Layer 0, the chunker or the GLiNER2 configuration, credit the
kggen pipeline alongside FoodScholar. The vendored tree at `kggen/graph_code`
is the citable form; the port's module docstrings name their source file.
