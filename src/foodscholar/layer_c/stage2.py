"""Stage 2 — refine the Stage-1 extract into a Card via the LLM.

The LLM sees ONLY the compact extract (plus the theme label/keywords for
context), never the raw chunks — that is the cost win. Output is mapped onto
the existing `Card` model; `cited_chunk_ids` carry theme-level provenance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from foodscholar.io.graph import Card

if TYPE_CHECKING:
    from foodscholar.config import LayerCConfig
    from foodscholar.layer_c.models import Stage1Output

_CARD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "tip": {"type": ["string", "null"]},
        "evidence_quality": {
            "type": "string",
            "enum": ["high", "medium", "low", "debated", "unclear"],
        },
        "controversy_note": {"type": ["string", "null"]},
        "confidence_note": {"type": ["string", "null"]},
    },
    "required": ["title", "summary", "evidence_quality"],
}

_PROMPT = """You are writing a concise knowledge card about the food topic "{label}".
Related keywords: {keywords}.

Below is an extractive summary distilled from {n_chunks} source passages. Use ONLY
this material — do not invent facts.

--- EXTRACT ---
{extract}
--- END EXTRACT ---

Produce a JSON object with:
- "title": a short topic title (<= 8 words)
- "summary": a clear narrative that organizes the key messages, main claims, and
  insights from the extract into flowing prose. Remove redundancy; reorganize freely.
- "tip": one practical takeaway, or null
- "evidence_quality": one of high|medium|low|debated|unclear
- "controversy_note": note any conflicting claims, or null
- "confidence_note": caveats about coverage, or null
"""


class _ThemeLike(Protocol):
    theme_id: str
    label: str
    facet: str
    keyword_terms: list[str]


def run_stage2(
    llm: Any,
    stage1: Stage1Output,
    theme: _ThemeLike,
    cited_chunk_ids: list[str],
    cfg: LayerCConfig,
) -> Card:
    """Refine `stage1.text` into a Card for `theme`. Raises ValueError if the
    strict grounding guard fails."""
    prompt = _PROMPT.format(
        label=theme.label,
        keywords=", ".join(theme.keyword_terms),
        n_chunks=stage1.n_input_chunks,
        extract=stage1.text,
    )
    data = llm.generate_json(prompt, _CARD_SCHEMA, max_tokens=1024)

    summary = str(data.get("summary", "")).strip()
    if cfg.grounding_check == "strict" and (
        not summary or len(summary) > cfg.max_summary_chars
    ):
        raise ValueError(
            f"grounding(strict): summary length {len(summary)} "
            f"outside (0, {cfg.max_summary_chars}]"
        )

    safety = theme.facet in cfg.safety_sensitive_facets

    return Card(
        card_id=f"card:theme:{theme.theme_id}",
        target_id=theme.theme_id,
        target_type="theme",
        title=str(data.get("title", theme.label)).strip(),
        summary=summary,
        tip=(data.get("tip") or None),
        evidence_quality=data.get("evidence_quality", "unclear"),
        controversy_note=(data.get("controversy_note") or None),
        confidence_note=(data.get("confidence_note") or None),
        cited_chunk_ids=list(cited_chunk_ids),
        llm_model=cfg.llm_model,
        prompt_version=cfg.prompt_version,
        safety_flagged=safety,
    )
