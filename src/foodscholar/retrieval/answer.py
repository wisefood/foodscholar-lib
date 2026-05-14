"""Answer type — output of the retrieval pipeline (Section 14 of BRIEF)."""

from __future__ import annotations

from pydantic import BaseModel

from foodscholar.io.chunk import ChunkId
from foodscholar.io.graph import CardId, ShelfId, ThemeId


class Answer(BaseModel):
    text: str
    tips: list[str]
    cited_chunks: list[ChunkId]
    cited_cards: list[CardId]
    activated_shelves: list[ShelfId]
    activated_themes: list[ThemeId]
    grounding_passed: bool
    llm_model: str
    prompt_version: str
