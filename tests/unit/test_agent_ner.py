"""Tests for AgenticNER — LLM-driven entity recognition.

A scripted mock LLM stands in for a real provider: it returns a fixed
`generate_json` payload so the tests are deterministic and offline. The focus
is the parts that carry logic — offset reconciliation, entity-type handling,
and graceful degradation — not the model itself.
"""

from __future__ import annotations

from foodscholar.annotate.agent_ner import AgenticNER
from foodscholar.storage.protocols import NER


class _ScriptedLLM:
    """Mock LLM whose generate_json returns a preset object."""

    model_id = "scripted-llm"

    def __init__(self, payload: dict | None = None, *, raise_exc: bool = False) -> None:
        self._payload = payload if payload is not None else {"mentions": []}
        self._raise = raise_exc
        self.json_calls = 0

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return ""

    def generate_json(self, prompt, schema, max_tokens=1024):  # type: ignore[no-untyped-def]
        self.json_calls += 1
        if self._raise:
            raise RuntimeError("LLM backend down")
        return self._payload


def _ner(mentions: list[dict]) -> AgenticNER:
    return AgenticNER(_ScriptedLLM({"mentions": mentions}))


def test_agentic_ner_implements_protocol() -> None:
    assert isinstance(AgenticNER(_ScriptedLLM()), NER)


def test_extract_locates_mention_offsets() -> None:
    text = "The Mediterranean diet is rich in olive oil."
    ner = _ner(
        [
            {"text": "Mediterranean diet", "entity_type": "dietary_pattern"},
            {"text": "olive oil", "entity_type": "food"},
        ]
    )
    out = ner.extract(text)
    assert len(out) == 2
    # offsets are computed locally — verify they actually index the substring
    for m in out:
        assert text[m.start : m.end] == m.text


def test_extract_preserves_entity_type() -> None:
    ner = _ner([{"text": "olive oil", "entity_type": "food"}])
    [m] = ner.extract("I use olive oil daily.")
    assert m.entity_type == "food"


def test_extract_unknown_entity_type_falls_back_to_other() -> None:
    ner = _ner([{"text": "olive oil", "entity_type": "condiment"}])  # not a valid type
    [m] = ner.extract("olive oil is great")
    assert m.entity_type == "other"


def test_extract_drops_mention_not_in_text() -> None:
    # The model paraphrased ("olives") instead of quoting the source ("olive").
    ner = _ner(
        [
            {"text": "olive", "entity_type": "food"},
            {"text": "quinoa pasta", "entity_type": "food"},  # not in text
        ]
    )
    out = ner.extract("An olive branch.")
    assert [m.text for m in out] == ["olive"]


def test_extract_repeated_mention_maps_to_successive_occurrences() -> None:
    text = "olive oil and more olive oil"
    ner = _ner(
        [
            {"text": "olive oil", "entity_type": "food"},
            {"text": "olive oil", "entity_type": "food"},
        ]
    )
    out = ner.extract(text)
    assert len(out) == 2
    assert out[0].start == 0
    assert out[1].start == text.index("olive oil", 1)
    assert out[0].start != out[1].start


def test_extract_empty_text_returns_empty() -> None:
    assert _ner([{"text": "olive oil", "entity_type": "food"}]).extract("") == []
    assert _ner([]).extract("   ") == []


def test_extract_llm_failure_degrades_to_empty() -> None:
    ner = AgenticNER(_ScriptedLLM(raise_exc=True))
    # An LLM exception must NOT propagate — NER returns [] and the phase continues.
    assert ner.extract("Mediterranean diet and olive oil.") == []


def test_extract_bad_shape_returns_empty() -> None:
    # generate_json returned the wrong shape (mentions not a list).
    ner = AgenticNER(_ScriptedLLM({"mentions": "not-a-list"}))
    assert ner.extract("olive oil") == []


def test_extract_missing_mentions_key_returns_empty() -> None:
    ner = AgenticNER(_ScriptedLLM({"unexpected": []}))
    assert ner.extract("olive oil") == []


def test_extract_skips_malformed_items() -> None:
    ner = _ner(
        [
            {"text": "olive oil", "entity_type": "food"},
            {"entity_type": "food"},          # missing text
            {"text": "", "entity_type": "food"},  # empty text
            "not-a-dict",                      # wrong type entirely
        ]
    )
    out = ner.extract("olive oil please")
    assert [m.text for m in out] == ["olive oil"]


def test_model_id_records_provider_and_prompt_version() -> None:
    ner = AgenticNER(_ScriptedLLM())
    assert "scripted-llm" in ner.model_id
    assert "agent-ner-v1" in ner.model_id


def test_ner_version_stamped_on_mentions() -> None:
    ner = _ner([{"text": "olive oil", "entity_type": "food"}])
    [m] = ner.extract("olive oil")
    assert m.ner_model_version == "agent-ner-v1"
