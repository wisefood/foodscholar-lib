"""Configuration for :mod:`foodscholar.chunking`, loadable on its own.

`ChunkingConfig` is the ``chunking:`` section of a full FoodScholar config, and
it also loads from a standalone YAML file with :meth:`ChunkingConfig.from_yaml`.
A standalone run never needs the ``corpus``, ``storage``, ``ontology`` or
``llm`` sections — that is what lets the chunker run with none of the rest of
the library installed or configured.

The model carries two generations of settings:

- the **legacy** fields (``max_tokens`` … ``excluded_pages_manifest``) drive the
  ``legacy`` profile, which reproduces the historical corpus bit-for-bit. They
  are pinned, not tuned: changing one makes new chunks incomparable with the
  stored ones;
- the **v2** fields (``profile`` … ``embedder_tokenizer``) drive the
  structure-first assembler. Per-source behaviour lives in ``sources``, keyed
  by source type, so a new domain is a new mapping rather than new code.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Genre = Literal["record", "long_document", "leaflet"]
"""Document genres the heads are organised by — not domains.

- ``record``: a short structured record (a journal abstract, an earnings release);
- ``long_document``: a long structured PDF with an outline (a textbook, an annual report);
- ``leaflet``: a layout-heavy leaflet, poster or infographic (a dietary guide, an investor one-pager).
"""

_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


class AssemblyConfig(BaseModel):
    """How the v2 assembler packs elements into chunks, per source type."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = 512
    """Hard ceiling on the *embedded* text — context header + content + the
    tokenizer's special tokens — counted with `ChunkingConfig.embedder_tokenizer`.
    The embedder silently truncates past it; in the stored corpus 20 % of the
    abstract chunks already exceed 512 and lose their tail."""

    target_tokens: int = 400
    """Soft packing target for content tokens. A chunk closes when adding the
    next unit would pass it, unless the chunk is still under `min_tokens`."""

    min_tokens: int = 120
    """Chunks under this are merged with a neighbour in the same section, or
    with an adjacent small chunk sharing a parent section
    (`merge_small_sections`). Fragments retrieve badly: two thirds of the
    stored guide chunks are under 120 tokens."""

    overlap_sentences: int = 0
    """Sentences copied from the previous chunk of the same section. **Zero by
    default**: neighbours are reached through adjacency links and the chunk
    store's ``expand()`` instead, which costs no duplicated evidence. Kept as
    an ablation; it never crosses a section boundary."""

    overlap_max_tokens: int = 64
    """Cap on the tokens `overlap_sentences` may copy."""

    split_mode: Literal["paragraph"] = "paragraph"
    """Where an over-budget section is cut: at paragraph boundaries, then
    sentences. The only mode implemented; embedding-breakpoint splitting is not."""

    context_header: bool = True
    """Prefix the embedded text with a deterministic one-line header — document
    short name, then the section path. The header is stored separately; the
    chunk text stays the faithful passage."""

    merge_small_sections: bool = True
    """Merge adjacent chunks that are both under `min_tokens` and share a
    parent section, labelling the result with the common section path."""

    @model_validator(mode="after")
    def _check_budget(self) -> AssemblyConfig:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 < self.target_tokens <= self.max_tokens:
            raise ValueError("target_tokens must be in (0, max_tokens]")
        if not 0 <= self.min_tokens <= self.target_tokens:
            raise ValueError("min_tokens must be in [0, target_tokens]")
        if self.overlap_sentences < 0 or not 0 <= self.overlap_max_tokens < self.max_tokens:
            raise ValueError("overlap must be non-negative and below max_tokens")
        return self


class RecordHeadOptions(BaseModel):
    """How the record head reads a file of records (JSON list or JSONL).

    The field names default to the journal-abstract files; another domain's
    records (an earnings release, a product sheet) map their own keys here.
    """

    model_config = ConfigDict(extra="forbid")

    text_field: str = "abstract"
    title_field: str = "title"
    id_field: str = "id"
    doi_field: str = "doi"
    keywords_field: str = "keywords"
    licence_field: str = "license"
    language_field: str | None = None

    header: Literal["title", "title_keywords"] = "title"
    """What the context header carries. ``title`` is the default; adding the
    keywords helps lexical matching but adds broad terms and displaces content
    tokens, so it is an ablation. To embed the abstract alone, set the
    profile's ``assembly.context_header: false``."""

    structured_sections: bool = True
    """Split labelled abstracts (*Background … Methods … Results …*) into one
    paragraph per section, so an abstract that overflows the budget is cut at
    section boundaries and each part records its section type."""


class SourceProfile(BaseModel):
    """What a source type is (its genre) and how it is assembled."""

    model_config = ConfigDict(extra="forbid")

    genre: Genre
    assembly: AssemblyConfig = Field(default_factory=AssemblyConfig)
    record: RecordHeadOptions = Field(default_factory=RecordHeadOptions)
    """Options for the ``record`` genre's head; ignored by the other genres."""


def nutrition_sources() -> dict[str, SourceProfile]:
    """The nutrition domain's source types mapped onto genres.

    An abstract packs up to the hard budget (``target_tokens = max_tokens``),
    so it stays one chunk whenever it fits and is only cut when it must be.
    """
    return {
        "abstract": SourceProfile(genre="record", assembly=AssemblyConfig(target_tokens=512)),
        "textbook": SourceProfile(genre="long_document"),
        "guide": SourceProfile(genre="leaflet"),
    }


class ChunkingConfig(BaseModel):
    """Corpus chunking — source documents to chunks, assets and provenance.

    The legacy fields keep the values that produced the existing corpus. The v2
    fields are opt-in through ``profile: v2``.
    """

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------ legacy (v1)

    max_tokens: int = 512
    """Legacy window size. The v2 budget is `AssemblyConfig.max_tokens`."""
    overlap: int = 64
    """Legacy minimum tail, not a target — the window advances as far as it can
    while still leaving this many tokens behind. Realized overlap is >= this."""
    fine_max_tokens: int = 80
    """Docling `HybridChunker` budget for the fine units the legacy window merges."""

    tokenizer: str = "BAAI/bge-large-en-v1.5"
    """Legacy token counter. NOT the same model as `annotate.embedder`
    (bge-*base*). They share a vocabulary so counts agree, but the corpus was
    counted with bge-large and this must stay pinned to it."""
    use_fast_tokenizer: bool = False
    """Load-bearing for the legacy profile: the corpus was counted with the SLOW
    tokenizer, and the fast and slow bge tokenizers can disagree by a token."""

    device: str = "cpu"
    """Docling accelerator device (``cpu``, ``cuda``, ``cuda:1``)."""
    do_ocr: bool = False
    generate_picture_images: bool = False

    chunk_id_strategy: Literal["uuid4", "content_hash"] = "uuid4"
    """Legacy ids. `uuid4` reproduces the existing corpus bit-for-bit;
    `content_hash` makes re-chunking an unchanged document idempotent. The v2
    profile ignores this and always uses the versioned identity scheme in
    :mod:`foodscholar.chunking.identity`."""

    excluded_pages_manifest: Path | None = None
    """Legacy ``filename: X.pdf | removed_pages: [1, 2]`` manifest, as produced by
    ``pdf_page_triage.py``. The v2 profile excludes *elements*, not pages."""

    # --------------------------------------------------------------------- v2

    profile: Literal["legacy", "v2"] = "legacy"
    """Which pipeline turns *source files* into chunks. `legacy` is the Docling
    window that produced the stored corpus (`fs.chunk_documents`, the
    `chunk-corpus` command); `v2` is the head + structure-first assembler path.
    The default stays `legacy` until the v2 heads cover every source type.
    `Chunker.assemble()` always runs v2 — it takes an IR, which only v2 has."""

    domain: str = "nutrition"
    """Names the built-in `sources` mapping. Only `nutrition` ships today."""

    sources: dict[str, SourceProfile] = Field(default_factory=nutrition_sources)
    """Source type → genre + assembly settings. Override one entry to tune a
    source type; add an entry to support a new one."""

    registry: Path | None = None
    """Source catalogue (the FSKG workbook, or a CSV/JSON with the same core
    columns). Supplies document-level metadata: title, licence, language,
    content page range, printed-page offset, edition pairs."""

    work_dir: Path | None = None
    """Where cached extractions, assets and reports are written."""

    embedder_tokenizer: str = "BAAI/bge-base-en-v1.5"
    """Tokenizer the v2 budget is counted with. Must match `annotate.embedder`,
    because the budget exists to stop the embedder truncating."""

    def profile_for(self, source_type: str) -> SourceProfile:
        """The profile for `source_type`, or a clear error naming the known ones."""
        try:
            return self.sources[source_type]
        except KeyError:
            known = ", ".join(sorted(self.sources)) or "(none)"
            raise KeyError(
                f"no chunking profile for source type {source_type!r}; "
                f"known: {known}. Add it under chunking.sources."
            ) from None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ChunkingConfig:
        """Validate a mapping, after `${ENV}` substitution.

        Accepts the chunking section itself or a whole FoodScholar config with
        a ``chunking:`` key — in which case every other section is ignored.
        """
        if not isinstance(data, dict):
            raise TypeError(f"expected a mapping, got {type(data).__name__}")
        section = data.get("chunking", data) if "chunking" in data else data
        if section is None:
            section = {}
        return cls.model_validate(substitute_env(section))

    @classmethod
    def from_yaml(cls, path: str | Path) -> ChunkingConfig:
        """Load from a standalone YAML file (or a full FoodScholar config)."""
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"chunking config at {path} must be a YAML mapping")
        return cls.from_mapping(raw)


def substitute_env(value: Any) -> Any:
    """Replace ``${VAR}`` with the environment value; unknown variables stay literal.

    Mirrors the main config loader so a chunking section behaves the same
    whether it is loaded alone or as part of a FoodScholar config.
    """
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute_env(v) for v in value]
    return value
