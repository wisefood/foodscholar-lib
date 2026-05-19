from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

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

# Coarse class of an extracted mention. `other` is the safe default for NER
# implementations that don't classify (e.g. KeywordNER). The agentic NER does
# classify. The type drives two things downstream: Layer A facet routing, and
# linker gating — only food-like types are resolved against FoodOn (a food
# ontology), so e.g. a `population` mention is kept but never linked.
EntityType = Literal[
    "food",
    "nutrient",
    "health",
    "dietary_pattern",
    "allergen",
    "population",   # demographic / life-stage group: children, adults, pregnant women, elderly
    "biomarker",    # measurable outcome or marker: glycemic control, inflammation markers, LDL cholesterol
    "processing",   # preparation / processing method or qualifier: fermentation, roasting, extra virgin
    "other",
]


def _utcnow() -> datetime:
    return datetime.now(UTC)


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

    embedding: list[float] | None = None
    embedding_model: str | None = None

    mentions: list[Mention] = Field(default_factory=list)
    entity_links: list[EntityLink] = Field(default_factory=list)
    foodon_ids: list[str] = Field(default_factory=list)

    shelf_ids: list[str] = Field(default_factory=list)
    theme_ids: list[str] = Field(default_factory=list)

    enrichment_version: str = "v0"
    created_at: datetime = Field(default_factory=_utcnow)
