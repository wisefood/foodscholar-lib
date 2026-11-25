"""The document intermediate representation (IR).

Every head turns a source file into a :class:`DocumentIR`: document metadata
plus an ordered list of typed elements in reading order. The assembler, the
identity scheme and the evaluation harness read only this model, never an
extractor's native output — so swapping Docling for another layout engine in a
head changes nothing downstream, and a bake-off is cheap.

Pages are a unit of **provenance**, not of content. An element that continues
across a page break (a paragraph, a list, a table) is one element carrying one
:class:`Region` per page it touches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CoordinateSpace = Literal["pdf_top_left", "pdf_bottom_left", "pixel_top_left"]
"""Coordinate conventions differ across extractors (Docling defaults to a
bottom-left origin; PyMuPDF and most viewers use top-left), so every bbox
states its space. Units are PDF points unless the space says pixels."""

FigureKind = Literal["photo", "diagram", "chart", "table_image", "infographic", "logo", "other"]
"""A domain-free figure taxonomy. The imagery path dispatches on it: a
``chart`` goes to chart-to-data, a ``table_image`` to table transcription,
a ``logo`` is dropped."""

BoxKind = Literal["tip", "key_takeaway", "learning_objective", "definition", "example", "note", "other"]
FurnitureKind = Literal["page_header", "page_footer", "page_number", "watermark", "other"]


class Region(BaseModel):
    """Where an element sits on one physical page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page: int
    """1-based physical PDF page — the stable locator. Printed page labels are
    display metadata on :class:`DocumentMeta`."""
    bbox: tuple[float, float, float, float] | None = None
    """``(x0, y0, x1, y1)`` in `coordinate_space`."""
    coordinate_space: CoordinateSpace = "pdf_top_left"
    page_size: tuple[float, float] | None = None
    """``(width, height)`` of the page, so a viewer can rescale the box."""


class _ElementBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    """Unique within this IR snapshot. Stable only per `extraction_run_id`."""
    text: str = ""
    regions: list[Region] = Field(default_factory=list)
    language: str | None = None
    """ISO 639-1 code when known (``en``, ``hu``, ``sl``)."""
    confidence: float | None = None
    source_ref: str | None = None
    """The extractor's own id or label, for debugging only."""

    @property
    def pages(self) -> tuple[int, ...]:
        return tuple(sorted({r.page for r in self.regions}))


class Heading(_ElementBase):
    kind: Literal["heading"] = "heading"
    level: int = 1
    """1 is the top of the outline. Levels need not be contiguous."""


class Paragraph(_ElementBase):
    kind: Literal["paragraph"] = "paragraph"
    section_label: str | None = None
    """For records: the labelled section the paragraph belongs to, verbatim
    (``"Background"``, ``"Purpose of review"``)."""
    section_type: str | None = None
    """The label normalised by the head (``introduction``, ``methods``,
    ``results``, ``conclusion``, ``discussion``, ``other``)."""


class ListItem(_ElementBase):
    kind: Literal["list_item"] = "list_item"
    depth: int = 0
    ordered: bool = False
    marker: str | None = None
    """The bullet or number as printed (``"3."``, ``"•"``)."""


class TableData(BaseModel):
    """Table content. Units, header rows and footnotes are first-class because
    financial and nutrition tables put units in headers, conditions in
    footnotes and negatives in parentheses — retrofitting them later would
    change every stored table."""

    model_config = ConfigDict(extra="forbid")

    cells: list[list[str]] = Field(default_factory=list)
    header_rows: int = 0
    units: list[str | None] | None = None
    """One per column when known."""
    footnotes: list[str] = Field(default_factory=list)

    @property
    def header(self) -> list[list[str]]:
        return self.cells[: self.header_rows]

    @property
    def body(self) -> list[list[str]]:
        return self.cells[self.header_rows :]

    def to_markdown(self, rows: list[list[str]] | None = None) -> str:
        """Render as a Markdown table, repeating the header over `rows` if given."""
        header = self.header or ([self.cells[0]] if self.cells else [])
        body = rows if rows is not None else (self.body if self.header_rows else self.cells[1:])
        if not header:
            return ""
        width = max(len(r) for r in [*header, *body])

        def line(row: list[str]) -> str:
            padded = [*(c.replace("|", "\\|").replace("\n", " ") for c in row), *[""] * (width - len(row))]
            return "| " + " | ".join(padded) + " |"

        out = [line(header[0]), "| " + " | ".join(["---"] * width) + " |"]
        out += [line(r) for r in header[1:]]
        out += [line(r) for r in body]
        out += [f"[{n}] {f}" if not f.startswith(("*", "†", "[")) else f for n, f in enumerate(self.footnotes, 1)]
        return "\n".join(out)


class Table(_ElementBase):
    kind: Literal["table"] = "table"
    table: TableData = Field(default_factory=TableData)
    caption: str | None = None
    asset_id: str | None = None
    """The rendered crop, when the imagery path made one."""


class Figure(_ElementBase):
    kind: Literal["figure"] = "figure"
    figure_kind: FigureKind = "other"
    asset_id: str | None = None
    caption: str | None = None
    description: str | None = None
    """Generated by a vision model; versioned through the asset's derivation."""
    alt: str | None = None


class Caption(_ElementBase):
    kind: Literal["caption"] = "caption"
    target_id: str | None = None
    """The figure or table this caption belongs to, once paired."""
    label: str | None = None
    """The normalised reference label (``"Figure 3.10"``) used to resolve
    in-text references."""


class Box(_ElementBase):
    kind: Literal["box"] = "box"
    box_kind: BoxKind = "other"
    title: str | None = None


class Footnote(_ElementBase):
    kind: Literal["footnote"] = "footnote"
    marker: str | None = None


class Furniture(_ElementBase):
    """Running headers, footers, page numbers. Kept for audit, never chunked."""

    kind: Literal["furniture"] = "furniture"
    furniture_kind: FurnitureKind = "other"


Element = Annotated[
    Heading | Paragraph | ListItem | Table | Figure | Caption | Box | Footnote | Furniture,
    Field(discriminator="kind"),
]


class Derivation(BaseModel):
    """How a generated text came to be — translation, description, context."""

    model_config = ConfigDict(extra="forbid")

    model: str
    prompt_version: str | None = None
    inputs: list[str] = Field(default_factory=list)
    review_state: Literal["unreviewed", "accepted", "rejected"] = "unreviewed"


class Asset(BaseModel):
    """A rendered figure, table crop or page image, stored next to the chunks."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    source_id: str
    kind: Literal["figure", "table", "page"] = "figure"
    figure_kind: FigureKind = "other"
    regions: list[Region] = Field(default_factory=list)
    path: str | None = None
    """Relative to the run's assets directory."""
    original_digest: str | None = None
    caption: str | None = None
    description: str | None = None
    transcribed_text: str | None = None
    derivation: Derivation | None = None
    licence: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)


class DocumentMeta(BaseModel):
    """Document-level facts, mostly from the source registry."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    """Persistent logical identity of the document (``SRC-IE-001``, a DOI)."""
    source_type: str
    """The domain's source type (``guide``, ``textbook``, ``abstract``)."""
    genre: str
    document_version_id: str = ""
    """Digest of the exact source bytes; changes when the file changes."""
    extraction_run_id: str = ""
    """Extractor, models, configuration and schema versions that produced this IR."""
    title: str | None = None
    short_name: str | None = None
    file: str | None = None
    language: str | None = None
    licence: str | None = None
    page_count: int | None = None
    page_labels: dict[int, str] = Field(default_factory=dict)
    """Physical page → printed label, when the PDF or the registry provides one."""
    edition_pair_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    """Domain metadata the core does not interpret (country, population, DOI, …)."""


IR_SCHEMA_VERSION = "1"


class DocumentIR(BaseModel):
    """One document: metadata plus elements in reading order."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = IR_SCHEMA_VERSION
    meta: DocumentMeta
    elements: list[Element] = Field(default_factory=list)
    assets: list[Asset] = Field(default_factory=list)

    def element(self, element_id: str) -> Any:
        for el in self.elements:
            if el.id == element_id:
                return el
        raise KeyError(element_id)

    def body_text(self) -> str:
        """All non-furniture text in reading order — what extraction retained."""
        parts: list[str] = []
        for el in self.elements:
            if el.kind == "furniture":
                continue
            if el.kind == "table":
                parts.append(el.table.to_markdown() or el.text)
            elif el.kind == "figure":
                parts.extend(p for p in (el.caption, el.description, el.text) if p)
            else:
                parts.append(el.text)
        return "\n".join(p for p in parts if p)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(self.model_dump_json(indent=None), encoding="utf-8")
        tmp.replace(out)
        return out

    @classmethod
    def load(cls, path: str | Path) -> DocumentIR:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("schema_version")
        if version != IR_SCHEMA_VERSION:
            raise ValueError(f"{path}: IR schema {version!r}, expected {IR_SCHEMA_VERSION!r}")
        return cls.model_validate(data)
