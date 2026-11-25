"""Moved to :mod:`foodscholar.chunking.legacy`; this path is kept for compatibility."""

from foodscholar.chunking.legacy import (
    CORPUS_COLUMNS,
    CorpusOverwriteError,
    _guard_output_dir,
    _pages_for,
    _tokenizer,
    chunk_documents,
    chunk_texts,
    chunks_to_rows,
    fine_units_from_pdf,
    fine_units_from_text,
    load_document_metadata,
    load_excluded_pages,
    make_chunk_id,
    token_counter,
    write_corpus_csv,
)

__all__ = [
    "CORPUS_COLUMNS",
    "CorpusOverwriteError",
    "_guard_output_dir",
    "_pages_for",
    "_tokenizer",
    "chunk_documents",
    "chunk_texts",
    "chunks_to_rows",
    "fine_units_from_pdf",
    "fine_units_from_text",
    "load_document_metadata",
    "load_excluded_pages",
    "make_chunk_id",
    "token_counter",
    "write_corpus_csv",
]
