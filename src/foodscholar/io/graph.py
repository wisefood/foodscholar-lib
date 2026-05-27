from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

ShelfId = str
ThemeId = str
CardId = str

Facet = Literal[
    "foods",
    "health",
    "sustainability",
    "dietary_patterns",
    "allergies",
    "nutrients",
]
EvidenceQuality = Literal["high", "medium", "low", "debated", "unclear"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Shelf(BaseModel):
    shelf_id: ShelfId
    label: str
    facet: Facet
    depth: int
    foodon_id: str | None = None
    parent_shelf_id: ShelfId | None = None
    chunk_count: int = 0
    support_direct: int = 0
    support_lifted: int = 0
    see_also: list[str] = Field(default_factory=list)


class Theme(BaseModel):
    theme_id: ThemeId
    label: str
    parent_theme_id: ThemeId | None = None
    shelf_ids: list[ShelfId]
    chunk_count: int = 0
    discovered_by: Literal["leiden", "hdbscan", "bertopic"]
    discovery_version: str
    # Layer B extensions (per layer_b_construction_brief.md §3)
    facet: Facet
    discovery_pass: Literal["relatedness", "merged", "global_similarity"]
    keyword_terms: list[str] = Field(default_factory=list)
    foodon_id_signature: list[str] = Field(default_factory=list)
    config_hash: str = ""
    version: str = ""


class Card(BaseModel):
    card_id: CardId
    target_id: str
    target_type: Literal["shelf", "theme"]
    title: str
    summary: str
    tip: str | None = None
    evidence_quality: EvidenceQuality
    controversy_note: str | None = None
    confidence_note: str | None = None
    cited_chunk_ids: list[str]
    llm_model: str
    prompt_version: str
    safety_flagged: bool = False
    generated_at: datetime = Field(default_factory=_utcnow)
