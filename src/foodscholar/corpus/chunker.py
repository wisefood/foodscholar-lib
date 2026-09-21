"""PDFs and raw text -> the corpus CSVs `fs.ingest()` reads.

Closes the loop: until this existed, foodscholar could not rebuild its own
primary input — every `chunk_id` in Elasticsearch traced back to a notebook in
a separate repository.

The sliding window itself lives in `foodscholar.corpus.window` and is shared by
every producer; this module supplies the producers and the CSV assembly.

.. warning::
   **Re-chunking an ingested corpus is destructive.** With the default
   ``chunk_id_strategy="uuid4"``, re-chunking a document produces entirely new
   ids, orphaning every `Relation.chunk_ids`, `Entity.chunk_ids`, shelf
   attachment, theme membership and `Card.cited_chunk_ids` that referenced the
   old ones. `chunk_documents` therefore refuses to overwrite a non-empty
   output directory unless `force=True`. Set
   ``chunking.chunk_id_strategy="content_hash"`` on new corpora to make
   re-chunking idempotent instead.
"""

from __future__ import annotations

import csv
import hashlib
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foodscholar.corpus.window import (
    FineUnit,
    MergedChunk,
    build_overlapping_chunks,
    has_meaningful_text,
)
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from foodscholar.config import ChunkerConfig
    from foodscholar.io.chunk import SourceType

_log = get_logger("foodscholar.corpus.chunker")

CORPUS_COLUMNS = ["chunk_id", "chunk_text", "type", "chunk_metadata"]

# `filename: X.pdf | removed_pages: [1, 2, 3]`, as written by pdf_page_triage.py
_MANIFEST_RE = re.compile(r"^filename:\s*(.+?)\s*\|\s*removed_pages:\s*\[([^\]]*)\]")


# --------------------------------------------------------------------- ids


def make_chunk_id(strategy: str, *, source_doc_id: str, text: str) -> str:
    """`uuid4` (reproduces the existing corpus) or `content_hash` (idempotent)."""
    if strategy == "content_hash":
        normalized = " ".join(text.split())
        payload = f"{source_doc_id}\x1f{normalized}".encode()
        return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:16]
    return str(uuid.uuid4())


# ------------------------------------------------------------- page exclusion


def load_excluded_pages(manifest_path: str | Path) -> dict[str, set[int]]:
    """Parse a removed-pages manifest into `{filename: {page, ...}}`.

    Note this covers the **guides** only. The textbooks' excluded pages were
    inline Python sets in the notebook; export them to this format to make
    textbook chunking reproducible outside it.
    """
    out: dict[str, set[int]] = {}
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        match = _MANIFEST_RE.match(line.strip())
        if not match:
            continue
        pages = {int(p) for p in match.group(2).split(",") if p.strip()}
        out[match.group(1).strip()] = pages
    return out


def _pages_for(filename: str, excluded: dict[str, set[int]]) -> set[int]:
    """Match on the full filename, then on the stem — as the notebook does."""
    if filename in excluded:
        return excluded[filename]
    stem = Path(filename).stem
    for key, pages in excluded.items():
        if key == stem or Path(key).stem == stem:
            return pages
    return set()


# ------------------------------------------------------------------ producers


def _tokenizer(cfg: ChunkerConfig) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        cfg.tokenizer, use_fast=cfg.use_fast_tokenizer
    )


def token_counter(cfg: ChunkerConfig) -> Callable[[str], int]:
    """Token counter pinned to the CHUNKER's tokenizer.

    `cfg.tokenizer` is bge-**large** while `annotate.embedder` defaults to
    bge-**base**. They share a tokenizer so counts agree today, but the corpus
    was counted with bge-large and this must stay pinned to it.
    """
    tok = _tokenizer(cfg)
    return lambda text: len(tok.tokenize(text))


def fine_units_from_text(text: str, *, group_key: str = "") -> list[FineUnit]:
    """NLTK sentences — the abstracts producer. No Docling required."""
    try:
        from nltk.tokenize import sent_tokenize
    except ImportError as e:
        raise ImportError(
            "the 'nltk' package is required for text chunking. "
            "Install with: pip install 'foodscholar[chunking]'"
        ) from e
    try:
        sentences = sent_tokenize(text)
    except LookupError:  # punkt not downloaded yet
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        sentences = sent_tokenize(text)
    return [FineUnit(text=s.strip(), group_key=group_key) for s in sentences if s.strip()]


def fine_units_from_pdf(pdf_path: str | Path, cfg: ChunkerConfig) -> list[FineUnit]:
    """Docling `HybridChunker` fine units — the guides/textbooks producer.

    Grouped by top-level heading, which is what keeps the window's overlap from
    bleeding across chapters.
    """
    try:
        from docling.chunking import HybridChunker
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorOptions,
            PdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.transforms.chunker.hierarchical_chunker import (
            ChunkingDocSerializer,
            ChunkingSerializerProvider,
        )
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
        )
        from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
    except ImportError as e:
        raise ImportError(
            "the 'docling' packages are required for PDF chunking. "
            "Install with: pip install 'foodscholar[chunking]'"
        ) from e

    class _MDTableNoImageProvider(ChunkingSerializerProvider):  # type: ignore[misc]
        """Tables as Markdown; images ignored entirely."""

        def get_serializer(self, doc: Any) -> Any:
            return ChunkingDocSerializer(
                doc=doc, table_serializer=MarkdownTableSerializer()
            )

    pdf_options = PdfPipelineOptions()
    pdf_options.accelerator_options = AcceleratorOptions(device=cfg.device)
    pdf_options.do_ocr = cfg.do_ocr
    pdf_options.generate_picture_images = cfg.generate_picture_images
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
    )

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=_tokenizer(cfg), max_tokens=cfg.fine_max_tokens
        ),
        serializer_provider=_MDTableNoImageProvider(),
        merge_peers=True,
        repeat_table_header=False,
        always_emit_headings=False,
        omit_header_on_overflow=False,
    )

    doc = converter.convert(source=str(pdf_path)).document
    units: list[FineUnit] = []
    for fine in chunker.chunk(dl_doc=doc):
        headings = getattr(fine.meta, "headings", None) or []
        pages = sorted(
            {
                prov.page_no
                for item in getattr(fine.meta, "doc_items", [])
                for prov in getattr(item, "prov", [])
                if getattr(prov, "page_no", None) is not None
            }
        )
        units.append(
            FineUnit(
                text=fine.text,
                group_key=headings[0] if headings else "",
                page_numbers=tuple(pages),
            )
        )
    return units


# ------------------------------------------------------------------ assembly


def chunks_to_rows(
    merged: list[MergedChunk],
    *,
    source_type: SourceType,
    base_metadata: dict[str, object],
    cfg: ChunkerConfig,
    source_doc_id: str,
    excluded_pages: set[int] | None = None,
) -> list[dict[str, str]]:
    """Filter and render merged chunks as corpus CSV rows.

    A chunk is dropped when it has no meaningful text, or when **any** of its
    pages is excluded. That second rule is the notebooks' behavior and is
    reproduced for parity — note it discards the whole chunk, so a 512-token
    chunk spanning a cover page and a content page loses the content too.
    """
    excluded = excluded_pages or set()
    rows: list[dict[str, str]] = []
    n_dropped_pages = 0
    for chunk in merged:
        if not has_meaningful_text(chunk.text):
            continue
        if chunk.page_numbers and any(p in excluded for p in chunk.page_numbers):
            n_dropped_pages += 1
            continue
        metadata = dict(base_metadata)
        if chunk.group_key:
            metadata["heading"] = chunk.group_key
        if chunk.page_numbers:
            # First page only — what the notebooks write. See the warning on
            # `foodscholar.io.chunk.ChunkProvenance`.
            metadata["page_number"] = chunk.first_page
        rows.append(
            {
                "chunk_id": make_chunk_id(
                    cfg.chunk_id_strategy,
                    source_doc_id=source_doc_id,
                    text=chunk.text,
                ),
                "chunk_text": chunk.text,
                "type": source_type,
                "chunk_metadata": repr(metadata),
            }
        )
    if n_dropped_pages:
        _log.info(
            "chunker.dropped_excluded_pages",
            source_doc_id=source_doc_id,
            n_chunks=n_dropped_pages,
        )
    return rows


def write_corpus_csv(rows: list[dict[str, str]], path: str | Path) -> Path:
    """Write rows in the shape `corpus/csv_reader.py` reads. Atomic."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CORPUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out)
    return out


def load_document_metadata(
    metadata_csv: str | Path, *, key_column: str = "pdf_name"
) -> dict[str, dict[str, object]]:
    """Load per-document metadata keyed on the PDF stem (guides only)."""
    out: dict[str, dict[str, object]] = {}
    with Path(metadata_csv).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row.get(key_column) or "").strip()
            if key:
                out[key] = {k: v for k, v in row.items() if k != key_column}
    return out


class CorpusOverwriteError(RuntimeError):
    """Raised when chunking would orphan an already-ingested corpus."""


def _guard_output_dir(out_dir: Path, cfg: ChunkerConfig, force: bool) -> None:
    """Refuse to silently re-chunk over existing corpus CSVs.

    With `uuid4` ids a re-chunk produces entirely new `chunk_id`s, so every
    relation, entity sample, shelf attachment, theme membership and card
    citation pointing at the old ids is orphaned. That is a rebuild, not an
    update, and it should be a deliberate choice.
    """
    if force or not out_dir.exists():
        return
    existing = sorted(out_dir.glob("*.csv"))
    if not existing:
        return
    if cfg.chunk_id_strategy == "content_hash":
        # Idempotent ids: re-chunking an unchanged document reproduces them, so
        # downstream provenance survives and there is nothing to guard.
        return
    raise CorpusOverwriteError(
        f"{out_dir} already holds {len(existing)} corpus CSV(s) and "
        f"chunk_id_strategy is 'uuid4', so re-chunking would assign new "
        f"chunk_ids and orphan every relation, attachment and card citing the "
        f"old ones. Pass force=True to rebuild from scratch, or set "
        f"chunking.chunk_id_strategy='content_hash' to make re-chunking "
        f"idempotent."
    )


def chunk_documents(
    pdf_dir: str | Path,
    *,
    out_dir: str | Path,
    source_type: SourceType,
    cfg: ChunkerConfig,
    metadata_csv: str | Path | None = None,
    excluded_pages: dict[str, set[int]] | str | Path | None = None,
    force: bool = False,
) -> list[Path]:
    """Chunk every PDF in `pdf_dir` into one corpus CSV per document.

    `excluded_pages` accepts a manifest path (guides) or an explicit
    `{filename: {pages}}` map (textbooks, whose exclusions were inline in the
    notebook). `metadata_csv` is required for guides — the notebook raises when
    a PDF is missing from it, and so do we, with a clearer message.

    Returns the written CSV paths.
    """
    pdf_dir = Path(pdf_dir)
    out_dir = Path(out_dir)
    _guard_output_dir(out_dir, cfg, force)

    if isinstance(excluded_pages, (str, Path)):
        excluded_map = load_excluded_pages(excluded_pages)
    elif excluded_pages is None and cfg.excluded_pages_manifest is not None:
        excluded_map = load_excluded_pages(cfg.excluded_pages_manifest)
    else:
        excluded_map = dict(excluded_pages or {})

    doc_metadata = (
        load_document_metadata(metadata_csv) if metadata_csv is not None else {}
    )

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"no PDFs found in {pdf_dir}")

    count_tokens = token_counter(cfg)
    written: list[Path] = []
    for pdf_path in pdfs:
        stem = pdf_path.stem
        base_metadata: dict[str, object] = {"file": pdf_path.name}
        if doc_metadata:
            meta = doc_metadata.get(stem)
            if meta is None:
                raise KeyError(
                    f"no metadata row for pdf_name={stem!r} in {metadata_csv}. "
                    f"Every guide must be listed there; add it or pass "
                    f"metadata_csv=None for sources without metadata."
                )
            base_metadata.update(meta)
            base_metadata["pdf_name"] = stem

        units = fine_units_from_pdf(pdf_path, cfg)
        merged = build_overlapping_chunks(
            units,
            count_tokens=count_tokens,
            max_tokens=cfg.max_tokens,
            overlap=cfg.overlap,
        )
        rows = chunks_to_rows(
            merged,
            source_type=source_type,
            base_metadata=base_metadata,
            cfg=cfg,
            source_doc_id=pdf_path.name,
            excluded_pages=_pages_for(pdf_path.name, excluded_map),
        )
        out_path = write_corpus_csv(
            rows, out_dir / f"chunks_{source_type}_{stem}.csv"
        )
        written.append(out_path)
        _log.info(
            "chunker.document_done",
            file=pdf_path.name,
            n_fine_units=len(units),
            n_merged=len(merged),
            n_rows=len(rows),
        )

    _log.info(
        "chunker.done", n_documents=len(written), out_dir=str(out_dir),
        chunk_id_strategy=cfg.chunk_id_strategy,
    )
    return written


def chunk_texts(
    texts: dict[str, str],
    *,
    out_path: str | Path,
    source_type: SourceType,
    cfg: ChunkerConfig,
    metadata: dict[str, dict[str, object]] | None = None,
) -> Path:
    """Chunk raw texts (abstracts) into a single corpus CSV.

    Uses the same window as the PDF path over NLTK sentences. Texts shorter
    than `cfg.max_tokens` pass through as one chunk, which is what the
    abstracts notebook does — only the long ones are actually split.
    """
    count_tokens = token_counter(cfg)
    metadata = metadata or {}
    rows: list[dict[str, str]] = []
    for doc_id in sorted(texts):
        text = texts[doc_id]
        if not has_meaningful_text(text):
            continue
        merged = build_overlapping_chunks(
            fine_units_from_text(text),
            count_tokens=count_tokens,
            max_tokens=cfg.max_tokens,
            overlap=cfg.overlap,
        )
        rows.extend(
            chunks_to_rows(
                merged,
                source_type=source_type,
                base_metadata=dict(metadata.get(doc_id, {})),
                cfg=cfg,
                source_doc_id=doc_id,
            )
        )
    path = write_corpus_csv(rows, out_path)
    _log.info("chunker.texts_done", n_texts=len(texts), n_rows=len(rows))
    return path
