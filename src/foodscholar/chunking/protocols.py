"""The model backends the chunker depends on, as protocols.

Every model the chunker can call sits behind one of these, with a stub that
needs no model, so the unit suite runs on a box with no GPU, no Docling and no
network. Real backends are adapters loaded lazily and only when configured;
none of them is imported by :mod:`foodscholar.chunking` itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from foodscholar.chunking.ir import DocumentIR, Region


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens the way the embedder will.

    `count` excludes special tokens; `special_tokens` is what the tokenizer
    adds per sequence (2 for BERT-style ``[CLS] … [SEP]``). The assembler adds
    them back when it checks the hard budget.
    """

    @property
    def special_tokens(self) -> int: ...

    def count(self, text: str) -> int: ...


@runtime_checkable
class SentenceSplitter(Protocol):
    def split(self, text: str, language: str | None = None) -> list[str]: ...


@runtime_checkable
class Extractor(Protocol):
    """Source file → IR. Docling, PaddleOCR-VL and end-to-end page readers are
    adapters behind this; the head chooses one by configuration."""

    name: str

    def extract(self, path: Path, **context: Any) -> DocumentIR: ...


@runtime_checkable
class Renderer(Protocol):
    """Rasterises a page region (vector drawings included) to image bytes."""

    def render(self, path: Path, region: Region, *, dpi: int = 150) -> bytes: ...


@runtime_checkable
class VisionModel(Protocol):
    """Image + prompt → a structured record (JSON-compatible dict)."""

    name: str

    def describe(self, image: bytes, prompt: str, *, schema: dict[str, Any] | None = None) -> dict[str, Any]: ...


@runtime_checkable
class Translator(Protocol):
    name: str

    def translate(self, texts: list[str], *, source: str, target: str = "en") -> list[str]: ...


@runtime_checkable
class LanguageIdentifier(Protocol):
    def identify(self, text: str) -> tuple[str | None, float]:
        """Return ``(iso_code, confidence)``; ``(None, 0.0)`` for unknown or mixed."""
        ...
