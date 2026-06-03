# Corpus input format

FoodScholar ingests a **chunked** corpus — the documents are already split into
passage-sized pieces before they reach the library. There are two inputs:

1. **Chunk CSVs** — the text and where it came from (required).
2. **NEL CSVs** — pre-computed named-entity mentions and their ontology links
   (optional; skip them to run NER + linking live instead).

```{mermaid}
flowchart LR
    CSV[chunks_*.csv] -->|iter_chunks| Ck[Chunk objects]
    NEL[nel_*.csv] -->|load_nel_dir| An[Mentions + EntityLinks]
    Ck --> Store[(chunk store)]
    An -.attach by chunk_id.-> Store
```

## Chunk CSVs

One row per chunk. Files are discovered by glob (`*.csv`) when you point at a
directory. **Required columns:**

| Column | Meaning |
|---|---|
| `chunk_id` | unique id for the chunk (stable across re-ingests) |
| `chunk_text` | the passage text |
| `type` | one of `abstract`, `textbook`, `guide` |
| `chunk_metadata` | a Python-literal `dict` string (see below) |

Example:

```text
chunk_id,chunk_text,type,chunk_metadata
ab_001,"Mediterranean diet reduces cardiovascular risk...",abstract,"{'DOI': '10.1/x', 'year': 2019, 'title': '...'}"
tb_014,"Saturated fats raise LDL cholesterol...",textbook,"{'file': 'nutrition_textbook.pdf', 'page': 88}"
```

`chunk_metadata` is parsed with `ast.literal_eval` and preserved verbatim as
`Chunk.source_metadata`. A few core fields are **derived** from it:

- **`source_doc_id`** — for `abstract`, the first of `DOI` / `doi` / `title`; for
  `textbook`/`guide`, the `file` key. Falls back to `chunk_id` if absent.
- **`year`** — parsed from a `year` key when present (int-like).
- **`section_type`** — derived from `type`: `abstract → abstract`,
  `guide → guideline`, `textbook → textbook`.

```{tip}
CSV fields may be large (full abstracts, document-level chunks). The reader raises
the field-size limit to 10 MB. Pass `strict=False` to skip malformed rows instead of
raising.
```

## NEL CSVs (annotations)

If you already have named-entity recognition + linking output, supply it as NEL CSVs
so ingestion skips the live NER/linker entirely (fast, deterministic, no models).
**Required columns:**

| Column | Meaning |
|---|---|
| `chunk_id` | matches a chunk's `chunk_id` |
| `chunk_entities_ner` | `;`-separated surface forms (the mention strings) |
| `chunk_uri_nel` | `;`-separated OBO Foundry URIs, **positionally paired** with the surfaces |

Example:

```text
chunk_id,chunk_entities_ner,chunk_uri_nel
ab_001,olive oil;cardiovascular disease,http://purl.obolibrary.org/obo/FOODON_03309927;http://purl.obolibrary.org/obo/...
```

Each `(surface, uri)` pair becomes a `Mention` plus an `EntityLink` on the matching
chunk. URIs are normalized to compact IDs (`FOODON:03309927`). Empty trailing entries
are tolerated when the two columns differ slightly in length.

## What a chunk becomes

Internally every row normalizes to the `Chunk` contract, which all later layers read:

```python
class Chunk(BaseModel):
    chunk_id: str
    text: str
    source_doc_id: str
    source_type: SourceType        # abstract | textbook | guide
    section_type: SectionType      # abstract | textbook | guideline | ...
    year: int | None
    source_metadata: dict[str, object]   # the parsed chunk_metadata, verbatim

    embedding: list[float] | None        # filled by fs.embed()
    mentions: list[Mention]              # from NEL CSV or live NER
    entity_links: list[EntityLink]       # from NEL CSV or the dense linker
    shelf_ids: list[str]                 # Layer A attachment (denorm)
    theme_ids: list[str]                 # Layer B attachment (denorm)
```

## Three ways to load

```python
# 1. Real stores: ingest chunk CSVs + attach NEL annotations (no models run)
fs.ingest("data/corpus", nel_dir="data/ner")

# 2. Real stores, live annotation: omit nel_dir to run NER + the dense linker
fs.ingest("data/corpus")        # then fs.embed(); fs.annotate() as needed

# 3. Offline: a prebuilt annotated Parquet snapshot (in-memory backend, no ES)
fs.load_chunks("data/annotated.parquet")
```

The **annotated Parquet snapshot** is a frozen, fully-annotated copy of the chunk
store written by `write_chunks_parquet` (the repo's `scripts/make_annotated_parquet.py`
builds one from the NEL CSVs). It lets you run the whole pipeline — Layer A, attach,
Layer B — with zero services, which is exactly how the offline path of
`notebooks/graph_build.ipynb` and the test suite operate.

See [](annotation.md) for what NER and the linker do when you *don't* supply NEL CSVs.
