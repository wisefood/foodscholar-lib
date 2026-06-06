"""Internal Layer C models: Stage-1 provenance, benchmark records, run report."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Stage1Output(BaseModel):
    """The extractive summary plus the provenance the builder records."""

    model_config = ConfigDict(extra="forbid")
    text: str
    n_input_chunks: int
    n_input_chars: int
    strategy: Literal["single", "mapreduce"]
    n_groups: int = 1


class MethodResult(BaseModel):
    """One benchmark record — matches the spec's evaluation JSON exactly."""

    model_config = ConfigDict(extra="forbid")
    method: str
    summary: str
    input_chunks: int
    input_chars: int
    execution_time_ms: int
    summary_length_chars: int


class LayerCReport(BaseModel):
    """Summary of a `build_layer_c` run."""

    model_config = ConfigDict(extra="forbid")
    n_themes: int
    n_cards: int
    n_skipped: int
    n_failed: int
    strategy_counts: dict[str, int] = Field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Layer C: {self.n_cards}/{self.n_themes} cards "
            f"(skipped {self.n_skipped}, failed {self.n_failed}); "
            f"strategies={self.strategy_counts}"
        )
