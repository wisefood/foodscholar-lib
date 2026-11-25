"""foodscholar.chunking — source documents to cited, navigable chunks.

Usable on its own: importing this package loads nothing else from FoodScholar
(no stores, no ontology, no LLM clients, no facade), and running it needs no
services. The output feeds ``fs.ingest`` unchanged.

Main entry points:

- :class:`Chunker` — assemble documents into chunks;
- :class:`ChunkingConfig` — the configuration, loadable from a standalone YAML;
- :class:`DocumentIR` — the intermediate representation every head emits;
- :class:`SourceRegistry` — document-level metadata from a catalogue;
- :mod:`foodscholar.chunking.evaluation` — score any chunker against gold statements.
"""

from foodscholar.chunking.assembler import AssembledChunk, Assembler, section_id
from foodscholar.chunking.chunker import Chunker, ChunkingResult, load_previous
from foodscholar.chunking.config import AssemblyConfig, ChunkingConfig, SourceProfile
from foodscholar.chunking.heads import Head, get_head, register_head, registered_genres
from foodscholar.chunking.identity import (
    IdentityReport,
    PriorChunk,
    assign_identity,
    link_adjacency,
)
from foodscholar.chunking.ir import (
    Asset,
    Box,
    Caption,
    Derivation,
    DocumentIR,
    DocumentMeta,
    Figure,
    Footnote,
    Furniture,
    Heading,
    ListItem,
    Paragraph,
    Region,
    Table,
    TableData,
)
from foodscholar.chunking.registry import SourceRecord, SourceRegistry
from foodscholar.chunking.text import HFTokenCounter, RegexSentenceSplitter, WhitespaceTokenCounter
from foodscholar.chunking.writers import (
    CHUNKER_VERSION,
    chunk_metadata,
    to_corpus_rows,
    to_records,
    write_corpus_csv,
    write_jsonl,
)

__all__ = [
    "CHUNKER_VERSION",
    "AssembledChunk",
    "Assembler",
    "AssemblyConfig",
    "Asset",
    "Box",
    "Caption",
    "Chunker",
    "ChunkingConfig",
    "ChunkingResult",
    "Derivation",
    "DocumentIR",
    "DocumentMeta",
    "Figure",
    "Footnote",
    "Furniture",
    "HFTokenCounter",
    "Head",
    "Heading",
    "IdentityReport",
    "ListItem",
    "Paragraph",
    "PriorChunk",
    "RegexSentenceSplitter",
    "Region",
    "SourceProfile",
    "SourceRecord",
    "SourceRegistry",
    "Table",
    "TableData",
    "WhitespaceTokenCounter",
    "assign_identity",
    "chunk_metadata",
    "get_head",
    "link_adjacency",
    "load_previous",
    "register_head",
    "registered_genres",
    "section_id",
    "to_corpus_rows",
    "to_records",
    "write_corpus_csv",
    "write_jsonl",
]
