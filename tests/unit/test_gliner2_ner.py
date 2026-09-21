"""GLiNER2 adapter against a fake model — no weights, no GPU."""

from __future__ import annotations

import pytest

from foodscholar.annotate.gliner2_ner import GLiner2NER, map_label
from foodscholar.config import GLiner2Config
from foodscholar.storage.protocols import NER


class FakeModel:
    def __init__(self, payload, *, fail_batch: bool = False) -> None:
        self.payload = payload
        self.fail_batch = fail_batch
        self.per_text_calls = 0

    def batch_extract_entities(self, texts, labels, **kwargs):
        if self.fail_batch:
            raise RuntimeError("CUDA OOM")
        return [self.payload for _ in texts]

    def extract_entities(self, text, labels, **kwargs):
        self.per_text_calls += 1
        return self.payload


def ner(payload, **kwargs) -> GLiner2NER:
    model = FakeModel(payload, **kwargs)
    instance = GLiner2NER(labels=GLiner2Config().labels)
    instance._model = model
    instance.fake = model  # type: ignore[attr-defined]
    return instance


PAYLOAD = {
    "entities": {
        "food": [
            {"text": "olive oil", "start": 0, "end": 9, "confidence": 0.9},
            {"text": "Olive Oil", "start": 0, "end": 9, "confidence": 0.8},
        ],
        "disease": [{"text": "diabetes", "start": 21, "end": 29, "confidence": 0.7}],
    }
}
TEXT = "olive oil helps with diabetes"


def test_satisfies_the_ner_protocol():
    assert isinstance(ner(PAYLOAD), NER)


def test_requires_labels():
    with pytest.raises(ValueError):
        GLiner2NER(labels={})


def test_extracts_with_offsets_and_mapped_types():
    mentions = ner(PAYLOAD).extract_batch([TEXT])[0]
    assert [(m.text, m.start, m.end) for m in mentions] == [
        ("olive oil", 0, 9),
        ("diabetes", 21, 29),
    ]
    assert mentions[1].entity_type == "medical condition"


def test_dedups_case_insensitively_first_occurrence_wins():
    mentions = ner(PAYLOAD).extract_batch([TEXT])[0]
    assert [m.text for m in mentions].count("olive oil") == 1


def test_recovers_missing_offsets_by_locating_the_surface():
    payload = {"entities": {"vitamin": [{"text": "vitamin D", "start": -1, "end": -1}]}}
    mentions = ner(payload).extract_batch(["take vitamin D daily"])[0]
    assert (mentions[0].start, mentions[0].end) == (5, 14)


def test_drops_surfaces_absent_from_the_text():
    payload = {"entities": {"food": [{"text": "kiwi", "start": -1, "end": -1}]}}
    assert ner(payload).extract_batch(["no fruit here"])[0] == []


def test_drops_too_short_surfaces():
    payload = {"entities": {"food": [{"text": "a", "start": 0, "end": 1}]}}
    assert ner(payload).extract_batch(["a banana"])[0] == []


def test_falls_back_to_per_text_when_the_batch_call_fails():
    instance = ner(PAYLOAD, fail_batch=True)
    mentions = instance.extract_batch([TEXT, TEXT])
    assert instance.fake.per_text_calls == 2
    assert len(mentions) == 2 and mentions[0]


def test_preserves_positional_alignment_around_empty_texts():
    results = ner(PAYLOAD).extract_batch(["", TEXT, "   "])
    assert results[0] == [] and results[2] == []
    assert results[1]


def test_empty_batch():
    assert ner(PAYLOAD).extract_batch([]) == []


def test_unknown_label_becomes_other():
    assert map_label("not a real label") == "other"


def test_model_id_records_the_settings():
    instance = GLiner2NER(labels={"food": "d"}, threshold=0.35)
    assert "gliner2(" in instance.model_id and "t=0.35" in instance.model_id
