# Implementation Log — Corpus Ingestion And Chunk Storage

Branch: `corpus-ingestion-storage`

Purpose: make the existing FoodScholar corpus CSVs a first-class input format,
normalize them into the package `Chunk` contract, and prepare storage for later
annotation/linking and Elasticsearch.

## Guardrails

- Respect the current corpus chunking method. The canonical input for this
  milestone is:
  `chunk_id, chunk_text, type, chunk_metadata`.
- Do not re-chunk existing corpus files.
- Preserve source metadata exactly enough for later catalogue alignment:
  abstracts carry title/venue/year/DOI/authors/citation counts; guides and
  textbooks carry file/heading/page/country/year/age group when available.
- Extraction and linking are later pipeline stages. This work must keep
  `mentions`, `entity_links`, `foodon_ids`, and `enrichment_version` ready for
  incremental updates by `chunk_id`.
- Keep Elasticsearch in mind: stable `chunk_id`, explicit core fields, batch
  iteration, and JSON-serializable metadata.

## Planned Steps

1. Add `source_metadata` to `Chunk` and test JSON round-tripping.
2. Add a legacy CSV reader that parses `chunk_metadata` and derives current
   `Chunk` fields.
3. Let corpus loading accept CSV files and directories of CSVs, streaming rows
   where possible.
4. Add batch iteration to the `ChunkStore` protocol and the in-memory store.
5. Add normalized Parquet writing/round-trip support.
6. Add a minimal annotation merge helper keyed by `chunk_id` for future linker
   output and fake Layer A tests.
7. Run focused tests and record verification.

## Next Implementation Slice

1. Promote annotation updates into the `ChunkStore` protocol so future
   Elasticsearch storage can update enrichment fields without forcing callers
   to fetch, copy, and upsert whole chunks.
2. Route `merge_annotations()` through that store method. External extraction
   and linking can keep producing compact records keyed by `chunk_id`.
3. Add a minimal Layer A builder that reads stored `foodon_ids`, propagates
   support to FoodOn ancestors, and upserts supported FoodOn terms as shelves.
4. Wire `FoodScholar.build_layer_a()` to the minimal builder while keeping
   `attach`, Layer B, Layer C, and retrieval deferred.
5. Add focused tests with fake annotations so this can be tested before the
   real annotation/linking stage is integrated.

## Notes

- `/mnt/workspaces/foodscholar/corpus/big` currently has 35 CSVs and roughly
  175k chunks. The largest file is `chunks_abstracts.csv`.
- `chunk_metadata` is a Python literal dict string, not JSON. It parses cleanly
  with `ast.literal_eval` on the sampled corpus.
- For now catalogue alignment is intentionally deferred. Later work can map
  `source_metadata` into richer catalogue citation/provenance fields.

## Implemented In This Pass

- Added `Chunk.source_metadata: dict[str, object]` to preserve the original
  corpus metadata.
- Added `foodscholar.corpus.csv_reader.iter_csv_chunks()` for the current CSV
  format.
- Extended `foodscholar.corpus.loader`:
  - `iter_chunks(path)` streams chunks from CSV, directory, JSONL, or Parquet.
  - `load_chunks(path)` remains the list-returning convenience wrapper.
  - `write_chunks_parquet(chunks, path)` writes normalized chunks.
- Added batch iteration to the chunk-store protocol:
  `ChunkStore.iter_chunks(batch_size=1000)`.
- Implemented `InMemoryChunkStore.iter_chunks()`.
- Added `ChunkAnnotation` and `merge_annotations()` for future external
  extraction/linking output keyed by `chunk_id`.
- Kept memory-backed `FoodScholar.from_config()` offline by default. It now
  uses the mock embedder unless a caller passes an explicit embedder; production
  backends still attempt config-built embedders and fall back loudly if
  unavailable.
- Added interactive and shell smoke checks:
  - `notebooks/corpus_ingestion_storage_smoke.ipynb`
  - `scripts/test_ingestion_storage.sh`
  - `scripts/smoke_corpus_storage.sh`

## Implemented In The Layer A Slice

- Added `ChunkStore.update_annotations(...)` to the storage protocol.
  Annotation/linking outputs can now update `mentions`, `entity_links`,
  `foodon_ids`, and `enrichment_version` by stable `chunk_id`.
- Implemented `InMemoryChunkStore.update_annotations(...)`.
- Routed `merge_annotations()` through the store-level update method instead
  of requiring a fetch/copy/upsert sequence in caller code. This is the shape
  Elasticsearch can later implement as a partial document update.
- Added `foodscholar.layer_a.builder`:
  - `build_shelves(...)` reads stored chunk `foodon_ids`.
  - Support is counted once per chunk per term.
  - Support propagates to known FoodOn ancestors.
  - Terms are filtered by `min_support`, `max_depth`, `facets`, and
    `blacklist_terms`.
  - Supported FoodOn terms become stable shelves with ids like
    `foodon:TEST:0000008`.
- Wired `FoodScholar.build_layer_a()` to build and upsert shelves into the
  graph store, returning `ArtifactMeta`.
- Kept `attach`, Layer B, Layer C, and retrieval deferred.
- Extended shell smoke checks so fake annotations can drive a minimal Layer A
  build before the real extraction/linking stage lands.
- Updated `notebooks/corpus_ingestion_storage_smoke.ipynb` into a linear
  pipeline walkthrough from CSV corpus ingestion to Layer A shelf build, with
  focused shell validation cells at the end.

## Design Details Worth Keeping

- `source_metadata` is written to normalized Parquet as a JSON string. This
  avoids Arrow schema inference failures when the same metadata key has mixed
  types across sources, e.g. `year` as an integer in abstracts and a string in
  guide metadata.
- CSV normalization derives:
  - `source_doc_id`: DOI/title for abstracts, file name for guides/textbooks.
  - `section_type`: abstract -> `abstract`, guide -> `guideline`, textbook ->
    `textbook`.
  - `year`: parsed only when the metadata value is a clean integer.
- Malformed CSV metadata raises in strict mode and is skipped in non-strict
  mode.

## Verification

- Focused tests:
  `PYTHONPATH=src /mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_io_models.py tests/unit/test_corpus_loader.py tests/unit/test_in_memory_stores.py tests/unit/test_annotation_merge.py tests/unit/test_cli.py tests/unit/test_facade.py tests/unit/test_facade_ontology.py`
  -> 51 passed.
- Layer A focused tests:
  `PYTHONPATH=src /mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit/test_in_memory_stores.py tests/unit/test_annotation_merge.py tests/unit/test_layer_a.py tests/unit/test_facade.py tests/unit/test_cli.py`
  -> 37 passed.
- Full unit suite:
  `PYTHONPATH=src /mnt/miniconda3/envs/foodscholar/bin/python -m pytest tests/unit`
  -> 201 passed, 1 skipped.
- Ruff:
  `PYTHONPATH=src /mnt/miniconda3/envs/foodscholar/bin/python -m ruff check ...`
  -> all checks passed on touched files.
- Real corpus smoke:
  streamed `/mnt/workspaces/foodscholar/corpus/big` with `iter_chunks()` ->
  175,513 chunks (`abstract`: 162,169; `textbook`: 12,194; `guide`: 1,150).
- Shell smoke scripts:
  - `scripts/test_ingestion_storage.sh` -> 56 passed.
  - `SAMPLE_SIZE=500 scripts/smoke_corpus_storage.sh` -> streamed the real
    corpus, round-tripped 500 chunks through Parquet, stored them in memory,
    merged a fake annotation, and built 5 mini FoodOn shelves in Layer A.
- Notebook smoke:
  executed all code cells in `notebooks/corpus_ingestion_storage_smoke.ipynb`
  via the conda Python. It passed CSV ingestion, real corpus streaming, Parquet
  round-trip, in-memory batch iteration, fake annotation merge, offline
  annotation against the mini FoodOn fixture, minimal Layer A shelf building,
  and graph handle checks.
- Full `pytest` was also attempted. It originally exposed offline HuggingFace
  model-download failures in memory-backed facade/CLI tests; the facade change
  above fixes the relevant path.

## Current Next Steps

1. Implement the `attach` phase: for every annotated chunk, attach it to the
   matching FoodOn shelves and denormalize `shelf_ids` back onto the chunk.
2. Add the Elasticsearch `ChunkStore` adapter with the same ingestion,
   iteration, search, annotation update, and attachment update contracts.
3. Add a catalogue export/mapping layer that converts normalized chunks plus
   `source_metadata` into WiseFood catalogue fields for citation URLs and
   passage highlighting.
4. Refine Layer A taxonomy rules after real annotations are available:
   single-child-chain collapse, facet routing beyond `foods`, and support
   thresholds per source type.
