# chunking — PDFs/abstracts → chunk corpora

Builds the chunk CSVs that `../build_graph/` turns into knowledge graphs.
Guides and textbooks are chunked with Docling; abstracts are split/curated and then
clustered semantically.

| File | Purpose |
|---|---|
| `chunking_pipeline_guides_overlap.ipynb` | 51 guide PDFs → 512-token chunks (64 overlap) → `../data/chunks/guides/`. |
| `chunking_pipeline_textbooks_overlap.ipynb` | 6 textbook PDFs → same pipeline → `../data/chunks/textbooks/`. |
| `chunking_abstracts_split_check.ipynb` | Abstracts: token/split estimator, splitting in corpus format, then semantic clustering → `../data/chunks/abstracts/clusters/`. |
| `pdfs.filtered.txt` | Manifest of non-content pages (`filename: X.pdf \| removed_pages: [1, 2, …]`) produced by `../utils/pdf_page_triage.py`; the guide pipeline skips those pages. |
| `.cache/` | Local HuggingFace cache (`XDG_CACHE_HOME`, `HF_HOME`, `HF_HUB_CACHE`) used by the notebooks. |

## Guides / textbooks pipeline

1. **Docling conversion** — `do_ocr=False`, `generate_picture_images=False` (images
   ignored), GPU `cuda:1`.
2. **Fine chunking** — `HybridChunker` with the `BAAI/bge-large-en-v1.5` tokenizer,
   `max_tokens=80`, tables serialized as Markdown, images omitted.
3. **Sliding window** — `build_overlapping_chunks()`: groups fine chunks by top-level
   heading (no cross-section overlap) and emits `max_tokens=512` windows with
   `overlap=64` tokens of tail overlap; a heading that fits is emitted as one chunk.
4. **Filtering** — pages listed in `pdfs.filtered.txt` for that PDF are dropped, as are
   chunks that are empty or punctuation-only.
5. **Export** — one CSV per document with `chunk_id`, `chunk_text`, `type`
   (`guide` / `textbook`) and `chunk_metadata` (`file`, `urn`, `country`, `title`,
   `audience`, `pages`, `heading`, `page_number`).

Guide metadata (URN, country, title, audience, page count) comes from
`../data/original/guides/guide_metadata.csv`, matched on the PDF filename.

## Abstracts pipeline

`chunking_abstracts_split_check.ipynb` works on
`../data/original/abstracts/curated_abstracts.csv` and mirrors the same sliding window
over NLTK sentences (no Docling):

1. **§1–3** — tokenizer (`BAAI/bge-large-en-v1.5`), CSV load, `count_tokens`,
   `split_into_sentences`, `estimate_split` (counts only).
2. **§4a** — per abstract: token count, sentence count, `needs_split` (> 512),
   `estimated_chunks`.
3. **§4b** — abstracts > 512 tokens are actually split (512/64) and written in the
   corpus format (`chunk_id, chunk_text, type, chunk_metadata`) to
   `abstract_chunks.csv`; the per-abstract counts go to `abstract_chunk_counts.csv`.
4. **§5–6** — summary statistics and the saved counts.
5. **§7–11** — semantic clustering: normalized bge embeddings (memory-mapped, reusable),
   `MiniBatchKMeans` into `--n-clusters` (default 20) non-overlapping clusters, one CSV
   per cluster, cluster statistics, and a verification pass asserting every input row
   lands in exactly one output file.

## Notes

- Chunking parameters live at the top of each notebook: `OUTPUT_MAX_TOKENS = 512`,
  `OUTPUT_OVERLAP = 64`, `FINE_MAX_TOKENS = 80`.
- Both guide/textbook notebooks must keep their working directory (`chunking/`) because
  inputs/outputs are addressed as `../data/original/...` and `../data/chunks/...`.
- `pdfs.filtered.txt` is produced once with
  `python ../utils/pdf_page_triage.py --manifest ... --out pdfs.filtered.txt`.
