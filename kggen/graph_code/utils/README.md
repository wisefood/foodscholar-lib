# utils — source acquisition and PDF triage helpers

Small tools that prepare the `../data/original/` inputs used by the chunking notebooks.

| File | Purpose |
|---|---|
| `pdf_page_triage.py` | LLM-based triage of PDF pages: decides which pages are non-content (covers, tables of contents, ads, licence pages) and writes a filtered manifest. |
| `guides_and_guidelines.ipynb` | Reads the source catalogue and creates/downloads the guide & guideline PDFs via the wisefood client. |
| `Sources_Catalogue.ods` | Source catalogue (countries, URNs, titles, audience, links). |
| `guides_removed_pages_log.txt` | Log of the pages removed from the guides (one entry per file). |

## `pdf_page_triage.py`

For every page it extracts text (pymupdf4llm by default), asks an Ollama model whether
the page carries content, and emits a manifest in the format the chunking pipelines
consume:

```
filename: <name.pdf> | removed_pages: [1, 2, 3]
```

```bash
# from a manifest (the usual entry point)
python pdf_page_triage.py --manifest ../chunking/pdfs.filtered.txt \
    --pdf-dir ../data/original/guides --out ../chunking/pdfs.filtered.txt \
    --model qwen2.5:14b-instruct --workers 4 --report triage_report.json

# or ad-hoc on PDFs / a directory
python pdf_page_triage.py --pdf ../data/original/guides/ie-key-messages.pdf
python pdf_page_triage.py --pdf-dir ../data/original/guides --out manifest.filtered.txt
```

Useful flags: `--manifest`, `--pdf`, `--pdf-dir`, `--out`, `--report` (JSON with per-page
decisions), `--dump-md DIR` (per-page markdown), `--model` (Ollama tag, default
`qwen2.5:14b-instruct`), `--host`, `--num-ctx`, `--timeout`, `--workers`, `--min-chars`,
`--max-chars`, `--prompt-file`, `--cache` / `--no-cache`, `--zero-based`, `--plain-text`,
`--quiet`. Decisions are cached in `.triage_cache.json` so re-runs are cheap.

## `guides_and_guidelines.ipynb`

1. **Setup** — credentials/username for the wisefood client.
2. **Read the sources catalogue** — parses `Sources_Catalogue.ods`.
3. **Build guide payloads** for the English rows (country ISO mapping, URNs, metadata).
4. **Compare with the catalogue** — queries the existing guides to see which candidates
   are still missing.
5. **Download the source PDFs** into `../data/original/guides/` (keeping the metadata CSV
   in sync; guides whose PDF cannot be fetched are still registered).

## Dependencies

Uses `pymupdf`, `pymupdf4llm`, `pdfplumber` and `requests` (triage) plus `pandas` and the
`wisefood` client (notebook) — all pinned in `../requirements.txt`.
