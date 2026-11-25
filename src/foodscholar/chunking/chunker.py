"""The chunker: IR → assembled chunks with identity, adjacency and outputs.

:class:`Chunker` is the entry point for programmatic use::

    from foodscholar.chunking import Chunker, ChunkingConfig, DocumentIR

    chunker = Chunker(ChunkingConfig.from_yaml("chunking.yaml"))
    result = chunker.assemble(DocumentIR.load("ir/SRC-IE-001.json"))
    result.write("out/")            # corpus CSVs for fs.ingest + chunks.jsonl

It needs no stores, no ontology, no LLM and no services.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from foodscholar.chunking.assembler import AssembledChunk, Assembler
from foodscholar.chunking.config import ChunkingConfig
from foodscholar.chunking.heads import get_head
from foodscholar.chunking.identity import IdentityReport, PriorChunk, assign_identity
from foodscholar.chunking.ir import DocumentIR, DocumentMeta
from foodscholar.chunking.protocols import SentenceSplitter, TokenCounter
from foodscholar.chunking.registry import SourceRegistry
from foodscholar.chunking.writers import to_corpus_rows, to_records, write_corpus_csv, write_jsonl
from foodscholar.logging import get_logger

_log = get_logger("foodscholar.chunking.chunker")


@dataclass
class ChunkingResult:
    chunks: list[AssembledChunk]
    documents: dict[str, DocumentMeta]
    identity: IdentityReport = field(default_factory=IdentityReport)

    def chunks_for(self, source_id: str) -> list[AssembledChunk]:
        return [c for c in self.chunks if c.source_id == source_id]

    def corpus_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for sid, meta in self.documents.items():
            rows.extend(to_corpus_rows(self.chunks_for(sid), meta))
        return rows

    def records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for sid, meta in self.documents.items():
            out.extend(to_records(self.chunks_for(sid), meta))
        return out

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        """Write one corpus CSV per source type, ``chunks.jsonl`` and a run summary."""
        out = Path(out_dir)
        written: dict[str, Path] = {}
        by_type: dict[str, list[dict[str, str]]] = {}
        for sid, meta in self.documents.items():
            by_type.setdefault(meta.source_type, []).extend(to_corpus_rows(self.chunks_for(sid), meta))
        for source_type, rows in sorted(by_type.items()):
            written[f"csv:{source_type}"] = write_corpus_csv(rows, out / f"chunks_{source_type}.csv")
        written["jsonl"] = write_jsonl(self.records(), out / "chunks.jsonl")
        summary = {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "identity": self.identity.summary(),
            "superseding": self.identity.superseding,
            "retired": self.identity.retired,
        }
        path = out / "run.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        written["summary"] = path
        return written


class Chunker:
    """Assembles documents into chunks according to a :class:`ChunkingConfig`."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        *,
        token_counter: TokenCounter | None = None,
        sentence_splitter: SentenceSplitter | None = None,
        registry: SourceRegistry | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self._tokens = token_counter
        self._splitter = sentence_splitter
        self._registry = registry

    @property
    def token_counter(self) -> TokenCounter:
        if self._tokens is None:
            from foodscholar.chunking.text import hf_token_counter

            self._tokens = hf_token_counter(self.config.embedder_tokenizer)
        return self._tokens

    @property
    def registry(self) -> SourceRegistry | None:
        if self._registry is None and self.config.registry is not None:
            self._registry = SourceRegistry.load(self.config.registry)
        return self._registry

    def assembler_for(self, source_type: str) -> Assembler:
        profile = self.config.profile_for(source_type)
        return Assembler(profile.assembly, token_counter=self.token_counter, sentence_splitter=self._splitter)

    def assemble(
        self,
        documents: DocumentIR | Iterable[DocumentIR],
        *,
        previous: Iterable[PriorChunk] | None = None,
    ) -> ChunkingResult:
        """Assemble one or more IRs; `previous` carries ids forward from an earlier run."""
        irs = [documents] if isinstance(documents, DocumentIR) else list(documents)
        chunks: list[AssembledChunk] = []
        metas: dict[str, DocumentMeta] = {}
        for ir in irs:
            if ir.meta.source_id in metas:
                raise ValueError(f"duplicate source_id {ir.meta.source_id!r} in one run")
            metas[ir.meta.source_id] = ir.meta
            produced = self.assembler_for(ir.meta.source_type).assemble(ir)
            chunks.extend(produced)
            _log.info("chunker.document_assembled", source_id=ir.meta.source_id, chunks=len(produced))
        report = assign_identity(chunks, previous)
        return ChunkingResult(chunks=chunks, documents=metas, identity=report)

    def extract(self, path: str | Path, *, source_type: str | None = None) -> list[DocumentIR]:
        """Run the head for a source file's genre. Raises until that head lands."""
        p = Path(path)
        record = self.registry.resolve(p) if self.registry is not None else None
        stype = source_type or (record.source_type if record else None)
        if stype is None:
            raise ValueError(f"cannot tell the source type of {p.name}; pass source_type= or configure a registry")
        head = get_head(self.config.profile_for(stype).genre)
        return head.extract(p, record=record, config=self.config, source_type=stype)

    def chunk_path(self, path: str | Path, *, source_type: str | None = None, previous: Iterable[PriorChunk] | None = None) -> ChunkingResult:
        return self.assemble(self.extract(path, source_type=source_type), previous=previous)


def load_previous(path: str | Path) -> list[PriorChunk]:
    """Prior chunks from an earlier run's ``chunks.jsonl``, for id continuity."""
    out: list[PriorChunk] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if not d.get("chunk_id") or not d.get("chunk_version_id"):
            continue
        out.append(
            PriorChunk(
                chunk_id=d["chunk_id"],
                chunk_version_id=d["chunk_version_id"],
                source_id=d.get("doc_id", ""),
                section_path=tuple(d.get("section_path") or ()),
                text=d.get("text", ""),
            )
        )
    return out
