"""Step 1+2 — entity and relation extraction, on foodscholar's `LLMClient`.

Ported from kggen's `steps/_1_get_entities.py` and `steps/_2_get_relations.py`,
which were structured-output calls wearing a dspy costume: both build a JSON
schema, send system+user messages, and validate the response. That is exactly
`LLMClient.generate_json(prompt, schema)`.

Going through the library's client rather than dspy/litellm means dspy and
litellm stay out of the dependency set, and provider fallback, `${ENV}` config
and the Groq reasoning-model guardrail all apply here for free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from foodscholar.logging import get_logger
from foodscholar.relations.extracted import ExtractedGraph, Triple
from foodscholar.relations.prompts import (
    PROMPT_VERSION,
    entities_prompt,
    relations_prompt,
)

if TYPE_CHECKING:
    from foodscholar.io.chunk import ChunkId
    from foodscholar.storage.protocols import LLMClient

_log = get_logger("foodscholar.relations.extract")

# Providers vary in how large an enum they will accept in a structured-output
# schema. Past this many entities we send the unconstrained schema and rely on
# the post-filter instead.
MAX_ENUM_ENTITIES = 120

ENTITIES_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
    "required": ["entities"],
    "additionalProperties": False,
}


def relations_schema(entities: list[str]) -> dict[str, object]:
    """Relation schema with subject/object constrained to the step-1 entities.

    **The enum is load-bearing — do not simplify it to a plain string.**
    kggen's `_create_relations_model` builds `Literal[tuple(entities)]` for
    both endpoints, so the provider's structured-output mode forces the model
    to pick endpoints from the entity list rather than inventing them. That is
    why their triples align with their entities. A plain string schema pushes
    the mismatch downstream into grounding, where an invented endpoint becomes
    a NIL node and quietly pollutes the graph.
    """
    endpoint: dict[str, object] = {"type": "string"}
    if entities and len(entities) <= MAX_ENUM_ENTITIES:
        endpoint = {"type": "string", "enum": list(entities)}
    return {
        "type": "object",
        "properties": {
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": endpoint,
                        "predicate": {"type": "string"},
                        "object": endpoint,
                    },
                    "required": ["subject", "predicate", "object"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["relations"],
        "additionalProperties": False,
    }


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def parse_entities(payload: dict[str, Any]) -> list[str]:
    """Deduplicate case-insensitively, first occurrence wins, order preserved."""
    raw = payload.get("entities") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        surface = _clean(item)
        key = surface.lower()
        if surface and len(surface) >= 2 and key not in seen:
            seen.add(key)
            out.append(surface)
    return out


def parse_relations(payload: dict[str, Any], entities: list[str]) -> list[Triple]:
    """Second tier of kggen's two-tier recovery.

    Even with the enum, a local model behind an OpenAI-compatible endpoint may
    ignore it, and `ExtendedGraph`'s own validator exists because "some
    relation extractors emit subject/object strings that were not returned by
    the entity extractor". So always filter endpoints back to the entity set.
    """
    items = payload.get("relations") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        # Tolerate a bare list, which some providers return.
        items = payload if isinstance(payload, list) else []

    allowed = {e.lower(): e for e in entities}
    out: list[Triple] = []
    dropped = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        subject = _clean(item.get("subject"))
        predicate = _clean(item.get("predicate"))
        obj = _clean(item.get("object"))
        if not (subject and predicate and obj):
            continue
        canon_s = allowed.get(subject.lower())
        canon_o = allowed.get(obj.lower())
        if canon_s is None or canon_o is None:
            dropped += 1
            continue
        if canon_s == canon_o:
            continue  # self-loops carry no information
        out.append((canon_s, predicate, canon_o))
    if dropped:
        _log.debug("relations.endpoints_filtered", dropped=dropped)
    return out


def extractor_version(llm: LLMClient) -> str:
    return f"{PROMPT_VERSION}({getattr(llm, 'model_id', 'unknown')})"


def extract_chunk(
    text: str,
    chunk_id: ChunkId,
    *,
    llm: LLMClient,
    max_tokens: int = 4096,
) -> ExtractedGraph:
    """Two LLM calls -> one `ExtractedGraph` carrying `chunk_id` as provenance.

    Returns an empty graph rather than raising when the model produces nothing
    usable: one bad chunk must not abort a corpus-wide run.
    """
    graph = ExtractedGraph()
    if not text or not text.strip():
        return graph

    try:
        entity_payload = llm.generate_json(
            entities_prompt(text), ENTITIES_SCHEMA, max_tokens=max_tokens
        )
    except Exception as e:
        _log.warning("relations.entities_failed", chunk_id=chunk_id, error=str(e))
        return graph

    entities = parse_entities(entity_payload)
    if not entities:
        return graph
    for surface in entities:
        graph.add_entity(surface, chunk_id)

    try:
        relation_payload = llm.generate_json(
            relations_prompt(text, entities),
            relations_schema(entities),
            max_tokens=max_tokens,
        )
    except Exception as e:
        message = str(e).lower()
        if "context length" in message or "too long" in message:
            # Chunks are <= 512 tokens by construction; if this fires the
            # corpus has a problem worth seeing rather than papering over.
            _log.warning("relations.chunk_too_long", chunk_id=chunk_id, error=str(e))
            return graph
        _log.warning("relations.enum_schema_failed", chunk_id=chunk_id, error=str(e))
        try:
            relation_payload = llm.generate_json(
                relations_prompt(text, entities),
                relations_schema([]),  # unconstrained fallback + post-filter
                max_tokens=max_tokens,
            )
        except Exception as e2:
            _log.warning("relations.relations_failed", chunk_id=chunk_id, error=str(e2))
            return graph

    for triple in parse_relations(relation_payload, entities):
        graph.add_triple(triple, chunk_id)
    return graph
