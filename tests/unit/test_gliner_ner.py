"""Unit tests for GLinerNER — no real model load.

The real model is downloaded behind `@pytest.mark.slow` (see
tests/integration/test_real_models.py). These tests inject a fake gliner
model into the NER's model slot to exercise the wrapper logic deterministically.
"""

from __future__ import annotations

import pytest

from foodscholar.annotate.gliner_ner import GLinerNER


class _FakeGLiNER:
    """Stand-in for the real `GLiNER` model. Records the calls it receives."""

    def __init__(self, by_text: dict[str, list[dict]]) -> None:
        self._by_text = by_text
        self.inference_calls: list[tuple[list[str], list[str], int]] = []

    def inference(
        self,
        texts: list[str],
        labels: list[str],
        *,
        batch_size: int,
        threshold: float,
        max_length: int,
        flat_ner: bool,
    ) -> list[list[dict]]:
        self.inference_calls.append((list(texts), list(labels), batch_size))
        return [list(self._by_text.get(t, [])) for t in texts]

    def predict_entities(
        self, text: str, labels: list[str], *, threshold: float, max_length: int, flat_ner: bool
    ) -> list[dict]:
        return list(self._by_text.get(text, []))


def _ner(by_text: dict[str, list[dict]]) -> GLinerNER:
    ner = GLinerNER(labels=["food", "dietary pattern"], threshold=0.4, batch_size=8)
    ner._model = _FakeGLiNER(by_text)  # bypass real model load
    return ner


def test_constructor_requires_at_least_one_label() -> None:
    with pytest.raises(ValueError, match="at least one label"):
        GLinerNER(labels=[])


def test_extract_empty_text_returns_no_mentions() -> None:
    ner = _ner({})
    assert ner.extract("") == []
    assert ner.extract("   ") == []


def test_extract_returns_mention_with_correct_offsets() -> None:
    text = "Mediterranean diet rich in olive oil."
    ner = _ner(
        {
            text: [
                {"text": "Mediterranean diet", "start": 0, "end": 18, "label": "dietary pattern", "score": 0.9},
                {"text": "olive oil", "start": 27, "end": 36, "label": "food", "score": 0.88},
            ]
        }
    )
    out = ner.extract(text)
    assert {(m.text, m.start, m.end, m.entity_type) for m in out} == {
        ("Mediterranean diet", 0, 18, "dietary pattern"),
        ("olive oil", 27, 36, "food"),
    }
    # All offsets must locate the surface verbatim.
    for m in out:
        assert text[m.start : m.end] == m.text


def test_unknown_label_falls_back_to_other() -> None:
    text = "Quinoa is rich."
    ner = _ner({text: [{"text": "Quinoa", "start": 0, "end": 6, "label": "grain", "score": 0.9}]})
    [m] = ner.extract(text)
    assert m.entity_type == "other"


def test_extract_batch_preserves_alignment_with_empties() -> None:
    """Empty inputs in the batch must not shift other rows' results."""
    ner = _ner(
        {
            "Apples are red.": [{"text": "Apples", "start": 0, "end": 6, "label": "food", "score": 0.9}],
            "Olive oil is healthy.": [{"text": "Olive oil", "start": 0, "end": 9, "label": "food", "score": 0.9}],
        }
    )
    out = ner.extract_batch(["Apples are red.", "", "Olive oil is healthy.", "   "])
    assert len(out) == 4
    assert {m.text for m in out[0]} == {"Apples"}
    assert out[1] == []
    assert {m.text for m in out[2]} == {"Olive oil"}
    assert out[3] == []


def test_extract_batch_dedupes_repeated_spans_at_same_offset() -> None:
    text = "olive oil and olive oil"
    ner = _ner(
        {
            text: [
                {"text": "olive oil", "start": 0, "end": 9, "label": "food", "score": 0.9},
                {"text": "olive oil", "start": 0, "end": 9, "label": "food", "score": 0.9},  # dup
                {"text": "olive oil", "start": 14, "end": 23, "label": "food", "score": 0.9},
            ]
        }
    )
    [mentions] = ner.extract_batch([text])
    starts = sorted(m.start for m in mentions)
    assert starts == [0, 14]


def test_extract_recovers_when_offsets_are_off() -> None:
    """If GLiNER returns invalid offsets, we locate the surface ourselves."""
    text = "An apple a day."
    ner = _ner(
        {
            text: [
                # offsets pointing past the end of the string — should be recovered
                {"text": "apple", "start": 999, "end": 1004, "label": "food", "score": 0.9}
            ]
        }
    )
    [m] = ner.extract(text)
    assert (m.start, m.end) == (3, 8)
    assert text[m.start : m.end] == "apple"


def test_model_id_carries_threshold_and_label_count() -> None:
    ner = GLinerNER(labels=["food", "nutrient"], threshold=0.4, batch_size=2)
    assert "gliner(" in ner.model_id
    assert "labels=2" in ner.model_id
    assert "t=0.4" in ner.model_id


def test_batch_call_logged_with_configured_size() -> None:
    """The runner relies on the batch call; assert it actually goes through inference()."""
    ner = _ner(
        {
            "A.": [],
            "B.": [],
        }
    )
    ner.extract_batch(["A.", "B."])
    fake = ner._model
    assert isinstance(fake, _FakeGLiNER)
    assert len(fake.inference_calls) == 1
    texts, labels, batch_size = fake.inference_calls[0]
    assert texts == ["A.", "B."]
    assert batch_size == 2
    assert labels == ["food", "dietary pattern"]
