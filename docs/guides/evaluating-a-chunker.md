# Evaluating a chunker

`foodscholar.chunking.evaluation` scores chunks against **gold statements** —
hand-extracted facts with the page and section they came from. It scores any
pipeline's output the same way: v2, the legacy window, Docling's shipped
chunker, a LangChain splitter or a commercial parser. Nothing in it calls a
model.

## Three stages

A statement is lost at one of three places, and each is scored separately, so
the embedder is never blamed for evidence the extractor dropped:

1. **Extraction coverage** — does the extracted document still contain the
   evidence? Needs the document's IR element texts; skipped otherwise.
2. **Assembly coverage** — is the statement inside **one** chunk (*single*),
   or inside one chunk plus its next neighbour (*bundle*)? Is the gold page
   inside that chunk's page span, and the gold section on its section path?
3. **Retrieval coverage** — does a retriever return a supporting chunk within
   the first *k* results, and within a fixed **retrieved-token budget**? The
   budget stops small chunks and overlap from gaming *k*.

Relevance is decided by evidence support, never by chunk id, so pipelines with
different chunk boundaries are compared on the same truth.

## How support is decided

A passage supports a statement when it contains at least 80 % of the
statement's content words — lower-cased, lightly stemmed, stop words dropped,
counted with multiplicity — **and every number in the statement**. A quantity
is the qualifier a nutrition claim most often loses, so a passage with the
right words and the wrong number does not count.

```{warning}
This is a deterministic **triage** signal, not ground truth. Gold statements
are paraphrased, so it under-counts real support; borderline cases need human
adjudication. An LLM judge may help triage but should never be the sole
validator.
```

## Inputs

**Chunks**, in any of:

| Format | Read by |
|---|---|
| neutral JSONL — one object per chunk; `doc_id` and `text` required; `chunk_id`, `page_start`, `page_end`, `section_path`, `parent_id`, `ordinal`, `tokens` optional | any external pipeline; v2 writes it as `chunks.jsonl` |
| corpus CSV, or a directory of them | the four-column format `fs.ingest` reads |
| parquet snapshot | `write_chunks_parquet` output, e.g. `data/annotated.parquet` |

**Gold**, as JSONL of `GoldStatement` records, or an FSKG pilot extraction
workbook read by `load_fskg_statements` (its `Statements` sheet). With a
[source registry](../concepts/chunking.md#the-source-registry), statements map
to sources through the catalogued statement-id ranges, and chunk files —
including the legacy pipeline's decorated file names — map to the same source
ids.

## Running it

```bash
foodscholar chunk eval \
  --chunks data/annotated.parquet \
  --gold FSKG_Dietary_Guides_Ireland.xlsx \
  --registry sources/datasources/Sources/FSKG_Sources_Extracted.xlsx \
  --label "Stored corpus, Irish guides" \
  --tokenizer hf --retrieval --out report.json
```

Only the documents the gold set covers are scored, so the hygiene numbers
describe them. `--tokenizer hf` counts tokens exactly; the default estimates
from characters. `--retrieval` adds a BM25 run with each statement as its own
query — a **leaky smoke test**, because the query shares the evidence's
wording; a real query set is generated from held-out source evidence and
reviewed. `--out` writes every per-statement outcome for error analysis.

From Python:

```python
from foodscholar.chunking import SourceRegistry
from foodscholar.chunking.evaluation import evaluate, load_chunk_records, load_fskg_statements

reg = SourceRegistry.load("FSKG_Sources_Extracted.xlsx")
gold = load_fskg_statements("FSKG_Dietary_Guides_Ireland.xlsx", reg)
report = evaluate(gold, load_chunk_records("data/chunks/v2/chunks.jsonl", registry=reg))
print(report.to_markdown())
```

## Query sets

`--retrieval` uses each statement as its own query, which leaks wording. A
real query set is a JSONL file, one query per line:

```text
{"query_id": "q1", "text": "Does dietary fibre lower cardiovascular risk?", "evidence": ["AR0012"]}
{"query_id": "q2", "text": "How does salt affect blood pressure in adults?", "evidence_docs": ["10.1146/annurev-nutr-…"]}
```

`evidence` names gold statements: a chunk is relevant when it supports one.
`evidence_docs` names documents: any chunk of one is relevant, which suits
queries written about a whole abstract. Pass the file with `--queries`; the
report then gives recall at *k*, recall within a token budget, and MRR. Write
queries from held-out source evidence, never from a candidate's own chunks,
or content a chunker dropped can never become a question.

Gold catalogued per collection — the journals' statements map to `SRC-JN-…`,
not to a DOI — matches chunks through their `collection_id`.

## Baseline: the stored corpus

The stored corpus, chunked by the legacy window, scored on 15 October 2026
against the 555 Irish gold statements (FSKG pilot workbook, four documents).
One of the four, *Healthy Eating for Older Adults* (`SRC-IE-004`, 124
statements), is not in the stored corpus, so 431 statements are scored.

| Metric | Stored corpus |
|---|---|
| Supported by one chunk | 65.4 % |
| Supported by a chunk + its next neighbour | 69.6 % |
| Gold page inside the chunk's page span | 85.8 % (38.3 % before the gold correction below) |
| Gold section matches the chunk's section | 33.0 % |
| Words found but a number missing | 0.7 % |
| Chunks for these documents | 407 |
| Tokens p50 / p95 / max (exact, bge) | 55 / 232 / 570 |
| Chunks under 120 / over 512 tokens | 80.8 % / 0.2 % |
| BM25 statement-as-query recall@5 / MRR (leaky) | 53.3 % / 51.5 % |

What it says: a third of the statements are not recoverable from any single
chunk, two thirds of the chunks are fragments, and sections match a third of
the time because the legacy window knows only the top-level heading.

```{note}
**The FSAI gold pages are off by one.** For *Healthy Eating, Food Safety and
Food Legislation* (`SRC-IE-001`), 152 of the 178 statements that can be
located in the catalogued PDF carry a `Page (PDF)` one lower than the page they
are actually on; the other Irish documents match exactly. The stored chunks'
pages are correct. Pass `--gold-page-offset SRC-IE-001=1` to correct it; the
report records the correction. Fix it in the workbook when convenient.
```

## Hygiene

Every report also describes the chunks themselves: count, token percentiles,
the share under `min_tokens` and over `max_tokens`, and duplicate texts. A
chunker that wins on coverage by emitting huge or duplicated chunks shows it
here.

## See also

- [Chunking concepts](../concepts/chunking.md)
- [Chunking a corpus](chunking-a-corpus.md)
- [Evaluation reference](../reference/chunking.md#evaluation)
