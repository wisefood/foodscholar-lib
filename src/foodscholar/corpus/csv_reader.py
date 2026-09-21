"""Readers for the current FoodScholar corpus CSV format.

The legacy corpus files use one uniform shape:

    chunk_id, chunk_text, type, chunk_metadata

`chunk_metadata` is a Python literal dict string. We preserve it as
`Chunk.source_metadata` and derive only the core fields needed by FoodScholar.
"""

from __future__ import annotations

import ast
import csv
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foodscholar.io.chunk import Chunk, SectionType, SourceType
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.config import CorpusValidationConfig

_log = get_logger("foodscholar.corpus.csv_reader")

REQUIRED_COLUMNS = {"chunk_id", "chunk_text", "type", "chunk_metadata"}

# Allow up to 10MB per CSV field. Large abstracts and full-document
# chunks routinely exceed the stdlib default and would otherwise raise
# `_csv.Error: field larger than field limit`.
csv.field_size_limit(10 * 1024 * 1024)


# Rough tokens-per-character for the `chars` estimate. English text through a
# BPE/WordPiece tokenizer lands near 4 characters per token; it is an estimate
# by design, to keep `transformers` out of the ingest path.
_CHARS_PER_TOKEN = 4


class ChunkValidator:
    """Counts oversized chunks at ingest.

    The library assumes chunks are <= 512 tokens, but nothing verified it until
    `fs.chunk_documents` existed — that construction happened in a notebook
    outside the library. This makes the assumption observable.

    Warns rather than raises by default: a corpus that trips the check is still
    a corpus, and aborting `fs.ingest` over one long chunk is worse than a
    noisy log.
    """

    def __init__(self, config: CorpusValidationConfig | None) -> None:
        self._config = config
        self._count_tokens = self._build_counter()
        self.n_checked = 0
        self.n_oversized = 0
        self.worst: tuple[str, int] | None = None

    @property
    def enabled(self) -> bool:
        return self._config is not None and self._config.enabled

    def _build_counter(self):
        if self._config is None or self._config.token_estimate == "chars":
            return lambda text: len(text) // _CHARS_PER_TOKEN
        # `tokenizer` mode uses the CHUNKER's tokenizer (bge-large), not the
        # embedder's (bge-base) — the corpus was counted with the former.
        from foodscholar.config import ChunkerConfig

        chunker_defaults = ChunkerConfig()
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            chunker_defaults.tokenizer, use_fast=chunker_defaults.use_fast_tokenizer
        )
        return lambda text: len(tok.tokenize(text))

    def check(self, chunk: Chunk) -> None:
        if not self.enabled or self._config is None:
            return
        self.n_checked += 1
        n_tokens = self._count_tokens(chunk.text)
        if n_tokens <= self._config.max_chunk_tokens:
            return
        self.n_oversized += 1
        if self.worst is None or n_tokens > self.worst[1]:
            self.worst = (chunk.chunk_id, n_tokens)
        if self._config.on_violation == "raise":
            raise ValueError(
                f"chunk {chunk.chunk_id} is ~{n_tokens} tokens, over the "
                f"{self._config.max_chunk_tokens} limit "
                f"(corpus.validation.on_violation='raise')"
            )
        _log.warning(
            "corpus.chunk_oversized",
            chunk_id=chunk.chunk_id,
            source_doc_id=chunk.source_doc_id,
            estimated_tokens=n_tokens,
            limit=self._config.max_chunk_tokens,
        )

    def summary(self) -> dict[str, object]:
        return {
            "n_checked": self.n_checked,
            "n_oversized": self.n_oversized,
            "worst_chunk_id": self.worst[0] if self.worst else None,
            "worst_tokens": self.worst[1] if self.worst else None,
        }

    def log_summary(self) -> None:
        if self.enabled and self.n_checked:
            _log.info("corpus.validation.summary", **self.summary())


def iter_csv_chunks(
    path: str | Path,
    *,
    strict: bool = True,
    validator: ChunkValidator | None = None,
) -> Iterator[Chunk]:
    """Yield normalized `Chunk` objects from one legacy corpus CSV file.

    `validator`, when supplied, counts oversized chunks as they stream past —
    see `ChunkValidator`. Pass the same instance across files to get a
    corpus-wide count.
    """
    p = Path(path)
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"{p} is missing required columns: {sorted(missing)}")

        for row in reader:
            try:
                chunk = _row_to_chunk(row)
            except Exception:
                if strict:
                    raise
                continue
            if validator is not None:
                validator.check(chunk)
            yield chunk


def _row_to_chunk(row: dict[str, str | None]) -> Chunk:
    chunk_id = _required(row, "chunk_id")
    text = _required(row, "chunk_text")
    source_type = _source_type(_required(row, "type"))
    metadata = _parse_metadata(row.get("chunk_metadata") or "")
    section_type = _section_type(source_type)

    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=_source_doc_id(source_type, metadata, chunk_id),
        source_type=source_type,
        section_type=section_type,
        year=_parse_year(metadata.get("year")),
        source_metadata=metadata,
    )


def _required(row: dict[str, str | None], key: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"missing required value: {key}")
    return value


def _parse_metadata(raw: str) -> dict[str, object]:
    if not raw.strip():
        return {}
    value = ast.literal_eval(raw)
    if not isinstance(value, dict):
        raise ValueError("chunk_metadata must parse to a dict")
    return {str(k): _jsonable(v) for k, v in value.items()}


def _jsonable(value: Any) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _source_type(raw: str) -> SourceType:
    normalized = raw.strip().lower()
    if normalized not in {"abstract", "textbook", "guide"}:
        raise ValueError(f"unsupported chunk type: {raw!r}")
    return normalized  # type: ignore[return-value]


def _section_type(source_type: SourceType) -> SectionType:
    if source_type == "abstract":
        return "abstract"
    if source_type == "guide":
        return "guideline"
    return "textbook"


def _source_doc_id(
    source_type: SourceType, metadata: dict[str, object], chunk_id: str
) -> str:
    if source_type == "abstract":
        for key in ("DOI", "doi", "title"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return chunk_id

    value = metadata.get("file")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return chunk_id


def _parse_year(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None
