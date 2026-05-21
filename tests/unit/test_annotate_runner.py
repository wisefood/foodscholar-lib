"""Unit tests for the annotate runner.

The runner is pure orchestration over NER/Linker/Embedder protocols, so we
inject scripted fakes here instead of loading the real GLiNER + HNSW models.
The real-models integration test (`tests/integration/test_real_models.py`)
exercises the GLiNER+HNSW pipeline end-to-end behind `@pytest.mark.slow`.
"""

from __future__ import annotations

import pytest

from foodscholar import FoodScholar
from foodscholar.annotate.runner import dry_run, run
from foodscholar.io.chunk import Chunk, EntityLink, Mention


class _FakeNER:
    model_id = "fake-ner-v0"

    def __init__(self, by_text: dict[str, list[Mention]]) -> None:
        self._by_text = by_text
        self.batch_calls: list[list[str]] = []

    def extract_batch(self, texts: list[str]) -> list[list[Mention]]:
        self.batch_calls.append(list(texts))
        return [list(self._by_text.get(t, [])) for t in texts]

    def extract(self, text: str) -> list[Mention]:
        return list(self._by_text.get(text, []))


class _FakeLinker:
    linker_id = "fake-linker-v0"

    def __init__(self, surface_to_id: dict[str, str]) -> None:
        self._table = surface_to_id
        self.batch_calls: list[int] = []

    def link_many(self, mentions: list[Mention]) -> list[EntityLink | None]:
        self.batch_calls.append(len(mentions))
        return [self._one(m) for m in mentions]

    def link(self, mention: Mention) -> EntityLink | None:
        return self._one(mention)

    def _one(self, m: Mention) -> EntityLink | None:
        term_id = self._table.get(m.text.lower())
        if term_id is None:
            return None
        return EntityLink(
            mention=m,
            ontology_id=term_id,
            confidence=0.9,
            method="dense",
            linker_version=self.linker_id,
        )


def _mention(text: str, start: int) -> Mention:
    return Mention(
        text=text,
        start=start,
        end=start + len(text),
        score=1.0,
        ner_model_version="fake",
        entity_type="food",
    )


@pytest.fixture
def fs() -> FoodScholar:
    fs = FoodScholar.in_memory()
    fs.upsert_chunks(
        [
            Chunk(
                chunk_id="c1",
                text="Mediterranean diet rich in olive oil reduces cardiovascular risk.",
                source_doc_id="d1",
                source_type="abstract",
                section_type="abstract",
            ),
            Chunk(
                chunk_id="c2",
                text="An apple a day keeps the doctor away.",
                source_doc_id="d2",
                source_type="abstract",
                section_type="results",
            ),
        ]
    )
    fs.attach_ner(
        _FakeNER(
            {
                "Mediterranean diet rich in olive oil reduces cardiovascular risk.": [
                    _mention("olive oil", 27),
                ],
                "An apple a day keeps the doctor away.": [
                    _mention("apple", 3),
                ],
            }
        )
    )
    fs.attach_linker(_FakeLinker({"olive oil": "FOODON:03301710", "apple": "FOODON:00001141"}))
    return fs


def test_annotate_writes_mentions(fs: FoodScholar) -> None:
    fs.annotate()
    c1 = fs.graph.chunk("c1")
    assert any(m.text == "olive oil" for m in c1.mentions)


def test_annotate_writes_links(fs: FoodScholar) -> None:
    fs.annotate()
    c1 = fs.graph.chunk("c1")
    assert any(link.ontology_id == "FOODON:03301710" for link in c1.entity_links)


def test_annotate_denormalizes_foodon_ids(fs: FoodScholar) -> None:
    fs.annotate()
    assert "FOODON:03301710" in fs.graph.chunk("c1").foodon_ids
    assert "FOODON:00001141" in fs.graph.chunk("c2").foodon_ids


def test_annotate_writes_embedding(fs: FoodScholar) -> None:
    fs.annotate()
    c1 = fs.graph.chunk("c1")
    assert c1.embedding is not None
    assert len(c1.embedding) > 0
    assert c1.embedding_model is not None


def test_annotate_stamps_enrichment_version(fs: FoodScholar) -> None:
    fs.annotate()
    c1 = fs.graph.chunk("c1")
    assert c1.enrichment_version == "annotate-v2"


def test_annotate_returns_artifact_meta(fs: FoodScholar) -> None:
    meta = fs.annotate()
    assert meta.phase == "annotate"
    assert meta.record_count == 2
    assert meta.artifact_id.startswith("annotate-")


def test_annotate_is_idempotent(fs: FoodScholar) -> None:
    fs.annotate()
    first = [link.ontology_id for link in fs.graph.chunk("c1").entity_links]
    fs.annotate()
    second = [link.ontology_id for link in fs.graph.chunk("c1").entity_links]
    assert first == second


def test_annotate_batches_ner_calls(fs: FoodScholar) -> None:
    """All chunks land in a single ner.extract_batch() with the configured size."""
    fs.config.annotate.batch_size = 16
    fs.annotate()
    fake_ner = fs.ner  # the _FakeNER attached in the fixture
    assert isinstance(fake_ner, _FakeNER)
    assert len(fake_ner.batch_calls) == 1
    assert len(fake_ner.batch_calls[0]) == 2


def test_annotate_respects_small_batch(fs: FoodScholar) -> None:
    """With batch_size=1 the runner makes one NER call per chunk."""
    fs.config.annotate.batch_size = 1
    fs.annotate()
    fake_ner = fs.ner
    assert isinstance(fake_ner, _FakeNER)
    assert len(fake_ner.batch_calls) == 2


def test_dry_run_returns_mentions_and_links(fs: FoodScholar) -> None:
    mentions, links = dry_run(
        "Mediterranean diet rich in olive oil reduces cardiovascular risk.",
        ner=fs.ner,
        linker=fs.linker,
    )
    assert any(m.text == "olive oil" for m in mentions)
    assert any(link.ontology_id == "FOODON:03301710" for link in links)


def test_run_function_directly(fs: FoodScholar) -> None:
    meta = run(
        fs.chunk_store,
        ner=fs.ner,
        linker=fs.linker,
        embedder=fs.embedder,
        config=fs.config,
    )
    assert meta.record_count == 2


def test_attach_linker_overrides_default(fs: FoodScholar) -> None:
    class NullLinker:
        linker_id = "null"

        def link(self, mention):  # type: ignore[no-untyped-def]
            return None

    fs.attach_linker(NullLinker())
    fs.annotate()
    c1 = fs.graph.chunk("c1")
    assert c1.entity_links == []
    assert c1.foodon_ids == []
