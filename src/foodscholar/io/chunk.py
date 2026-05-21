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

# Class of an extracted mention. The set is the literal label vocabulary
# emitted by GLiNER-bio so the NER bridge is a no-op string copy. `other` is
# kept as the safe default for NER impls that don't classify.
EntityType = Literal[
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
