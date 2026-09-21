from datetime import UTC, datetime
from functools import cached_property
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ChunkId = str
SectionType = Literal[
    "abstract",
    "results",
    "discussion",
    "methods",
    "introduction",
    "conclusion",
    "guideline",
    "textbook",
    "other",
]
SourceType = Literal["abstract", "textbook", "guide"]

# Class of an extracted mention. The vocabulary is the UNION of the two NER
# backends the library ships:
#
#   - GLiNER-bio (`annotate/gliner_ner.py`) emits the first block verbatim, so
#     that bridge stays a no-op string copy.
#   - GLiNER2 (`annotate/gliner2_ner.py`) emits a richer 27-label set; the
#     labels that differ only in spelling or case are folded onto the existing
#     members by `GLINER2_TO_ENTITY_TYPE` below, and the genuinely new ones are
#     added to the second block.
#
# `other` is the safe default for NER impls that don't classify. Adding a
# member here is additive and safe; *renaming* one is not — stored chunks carry
# these strings.
EntityType = Literal[
    # ── GLiNER-bio vocabulary (original; do not rename) ──
    "food",
    "nutrient",
    "micronutrient",
    "macronutrient",
    "food component",
    "dietary supplement",
    "dietary pattern",
    "medical condition",
    "biomarker",
    "Country",
    "Measurement",
    "Population",
    "Time expression",
    # ── GLiNER2 additions ──
    "food additive",
    "vitamin",
    "mineral",
    "amino acid",
    "lipid",
    "chemical",
    "drug",
    "enzyme",
    "hormone",
    "gene",
    "genotype",
    "microbe",
    "symptom",
    "organ or tissue",
    "physiological process",
    "life stage",
    "exercise",
    # ── fallback ──
    "other",
]

# GLiNER2 labels whose spelling/case differs from the established member.
# Labels absent from this map are used verbatim (they are members above).
# Keeping the old capitalized forms avoids rewriting every stored chunk.
GLINER2_TO_ENTITY_TYPE: dict[str, str] = {
    "disease": "medical condition",
    "population": "Population",
    "measurement": "Measurement",
    "time expression": "Time expression",
    "country": "Country",
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


# Keys that appear under more than one spelling across the corpus producers.
# `csv_reader._source_doc_id` already probes ("DOI", "doi", "title") in order,
# which is the evidence that both casings are live in stored chunks.
_PROVENANCE_ALIASES: dict[str, str] = {
    "DOI": "doi",
    "Doi": "doi",
    "Year": "year",
    "Title": "title",
    "paperId": "paper_id",
    "pdfName": "pdf_name",
}


class ChunkProvenance(BaseModel):
    """Reader-side view of `Chunk.source_metadata`.

    The corpus has **three** producers emitting **three different key sets**:

    ===========  =========================================================
    source_type  keys emitted
    ===========  =========================================================
    guide        file, urn, pdf_name, country, title, audience, pages,
                 heading, page_number
    textbook     file, heading, page_number  *(only these three)*
    abstract     title, year, paper identifiers (doi / paperId)
    ===========  =========================================================

    So every field is optional and unknown keys are preserved (``extra="allow"``):
    this documents what the library *reads*, it does not constrain what a
    producer may write. Consumers that render citations must degrade per
    ``source_type`` rather than assume the guide shape — a textbook chunk has
    no title and no country.

    .. warning::
       ``page_number`` is the chunk's **first** page, not its span. The
       chunkers write ``page_numbers[0]`` while a 512-token chunk may cover
       several pages, so a citation into a chunk's tail can point one page
       early. See the integration brief §6.2.
    """

    model_config = ConfigDict(extra="allow")

    # guide + textbook
    file: str | None = None
    page_number: int | None = None
    heading: str | None = None
    # guide only
    pdf_name: str | None = None
    urn: str | None = None
    country: str | None = None
    audience: str | None = None
    pages: int | None = None
    # guide + abstract
    title: str | None = None
    # abstract only
    doi: str | None = None
    paper_id: str | None = None
    year: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_keys(cls, data: Any) -> Any:
        """Fold alias spellings onto the canonical key without losing values."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for alias, canonical in _PROVENANCE_ALIASES.items():
            if alias not in out:
                continue
            value = out.pop(alias)
            if out.get(canonical) in (None, ""):
                out[canonical] = value
        return out

    @model_validator(mode="before")
    @classmethod
    def _coerce_ints(cls, data: Any) -> Any:
        """Producers write page/year as str or float depending on the CSV round-trip."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for key in ("page_number", "pages", "year"):
            value = out.get(key)
            if value is None or isinstance(value, int):
                continue
            if isinstance(value, float) and value.is_integer():
                out[key] = int(value)
            elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
                out[key] = int(value.strip())
            else:
                out[key] = None
        return out

    def citation_fields(self) -> dict[str, object]:
        """The non-empty subset a citation renderer can actually use."""
        base = {
            "file": self.file,
            "title": self.title,
            "country": self.country,
            "heading": self.heading,
            "page_number": self.page_number,
            "doi": self.doi,
            "year": self.year,
        }
        return {k: v for k, v in base.items() if v not in (None, "")}


class Mention(BaseModel):
    text: str
    start: int
    end: int
    score: float
    ner_model_version: str
    entity_type: EntityType = "other"


class EntityLink(BaseModel):
    mention: Mention
    ontology_id: str
    confidence: float
    method: Literal["lexical_exact", "lexical_fuzzy", "dense", "llm"]
    linker_version: str


class Chunk(BaseModel):
    chunk_id: ChunkId
    text: str
    source_doc_id: str
    source_type: SourceType
    section_type: SectionType
    year: int | None = None
    source_metadata: dict[str, object] = Field(default_factory=dict)

    embedding: list[float] | None = None
    embedding_model: str | None = None

    mentions: list[Mention] = Field(default_factory=list)
    entity_links: list[EntityLink] = Field(default_factory=list)
    foodon_ids: list[str] = Field(default_factory=list)

    shelf_ids: list[str] = Field(default_factory=list)
    theme_ids: list[str] = Field(default_factory=list)

    enrichment_version: str = "v0"
    created_at: datetime = Field(default_factory=_utcnow)

    @cached_property
    def provenance(self) -> ChunkProvenance:
        """Typed view over `source_metadata`. See `ChunkProvenance`.

        Prefer this to `source_metadata.get(...)`: it normalizes the alias
        spellings (`DOI`/`doi`) and states which fields each source type
        actually carries.
        """
        return ChunkProvenance.model_validate(self.source_metadata)
