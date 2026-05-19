"""Agentic NER — an LLM extracts food/health entity mentions from a chunk.

`AgenticNER` implements the `NER` protocol, so it is a drop-in alternative to
`KeywordNER`. One `generate_json` call per chunk asks the model for every
relevant mention plus its entity type; the model returns mention *strings*
(not offsets — LLMs cannot count characters reliably), and this module locates
each string in the source text itself to produce correct `Mention` spans.

This is the first piece of the agentic annotation redesign
(see docs/DESIGN_agentic_annotate.md). It is deliberately a standalone NER
stage: it produces `Mention`s for the existing linker to resolve. A later
iteration may fuse extraction and linking into one tool-using agent.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, get_args

from foodscholar.io.chunk import EntityType, Mention
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.storage.protocols import LLMClient

_log = get_logger("foodscholar.annotate.agent_ner")

PROMPT_VERSION = "agent-ner-v1"

# Valid entity_type values, derived from the io contract so the two never drift.
_ENTITY_TYPES: tuple[str, ...] = get_args(EntityType)

# JSON schema handed to generate_json — constrains the output shape.
_NER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "entity_type": {"type": "string", "enum": list(_ENTITY_TYPES)},
                },
                "required": ["text", "entity_type"],
            },
        }
    },
    "required": ["mentions"],
}

_PROMPT = """\
You are a nutrition-domain entity recognizer. Extract every mention of a \
food, nutrient, health concept, dietary pattern, or allergen from the text \
below.

For each mention return:
  - "text": the mention exactly as it appears in the source text, verbatim \
(same casing, same words — it must be a literal substring of the text).
  - "entity_type": one of {entity_types}.

Rules:
  - Extract the mention span as written; do not normalize, pluralize, or \
expand it.
  - A clinical condition or disease (e.g. "iron deficiency", "coeliac \
disease") is "health", not "food".
  - If the same surface form appears multiple times, list it multiple times.
  - If nothing relevant appears, return an empty list.

Text:
\"\"\"
{text}
\"\"\"
"""


def _reconcile_spans(text: str, raw_mentions: list[dict]) -> list[Mention]:
    """Turn (text, entity_type) dicts into Mention objects with correct spans.

    The model returns mention strings; we find each in the source text. A
    cursor advances so repeated mentions map to successive occurrences rather
    than all collapsing onto the first. A returned string that is not a
    verbatim substring is dropped with a warning — that catches the model
    paraphrasing instead of quoting.
    """
    mentions: list[Mention] = []
    # Per distinct surface form, remember where the last occurrence ended so
    # the next identical mention is found *after* it.
    search_from: dict[str, int] = {}

    for item in raw_mentions:
        if not isinstance(item, dict):
            continue
        m_text = item.get("text")
        if not isinstance(m_text, str) or not m_text.strip():
            continue
        e_type = item.get("entity_type", "other")
        if e_type not in _ENTITY_TYPES:
            e_type = "other"

        start_at = search_from.get(m_text, 0)
        idx = text.find(m_text, start_at)
        if idx == -1:
            # Not found from the cursor — try from the very start in case the
            # model reordered mentions; only then give up.
            idx = text.find(m_text)
        if idx == -1:
            _log.warning("agent_ner.mention_not_in_text", mention=m_text)
            continue

        end = idx + len(m_text)
        search_from[m_text] = end
        mentions.append(
            Mention(
                text=m_text,
                start=idx,
                end=end,
                score=1.0,
                ner_model_version=f"{PROMPT_VERSION}",
                entity_type=e_type,  # type: ignore[arg-type]
            )
        )
    return mentions


class AgenticNER:
    """LLM-driven NER. One `generate_json` call per chunk; spans reconciled
    locally so `Mention.start`/`end` are always correct.
    """

    def __init__(self, llm: LLMClient, *, max_tokens: int = 2048) -> None:
        self._llm = llm
        self._max_tokens = max_tokens
        self.model_id = f"agentic-ner({llm.model_id};{PROMPT_VERSION})"

    def extract(self, text: str) -> list[Mention]:
        if not text or not text.strip():
            return []
        prompt = _PROMPT.format(
            entity_types=", ".join(_ENTITY_TYPES), text=text
        )
        try:
            result = self._llm.generate_json(
                prompt, _NER_SCHEMA, max_tokens=self._max_tokens
            )
        except Exception as e:
            # An LLM failure must degrade NER to "no mentions", never crash
            # the annotate phase over a corpus.
            _log.warning("agent_ner.llm_failed", error=str(e))
            return []

        raw = result.get("mentions", [])
        if not isinstance(raw, list):
            _log.warning("agent_ner.bad_shape", got=type(raw).__name__)
            return []
        mentions = _reconcile_spans(text, raw)
        _log.info(
            "agent_ner.extracted",
            n_returned=len(raw),
            n_kept=len(mentions),
            model=self._llm.model_id,
        )
        return mentions


def schema_json() -> str:
    """The NER output schema as a JSON string — handy for prompt docs/tests."""
    return json.dumps(_NER_SCHEMA, indent=2)
