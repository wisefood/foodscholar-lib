"""Tests for the annotate phase runner + facade integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from foodscholar import FoodOnAPI, FoodScholar
from foodscholar.annotate import dry_run, run
from foodscholar.io.chunk import Chunk
from foodscholar.ontology import load_ontology

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def fs() -> FoodScholar:
    fs = FoodScholar.in_memory()
    fs.attach_ontology(FoodOnAPI(load_ontology(FIXTURES / "mini_foodon.obo")))
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
    return fs


def test_annotate_writes_mentions(fs: FoodScholar) -> None:
    fs.annotate()
    c1 = fs.graph.chunk("c1")
    assert any(m.text.lower() == "olive oil" for m in c1.mentions)


def test_annotate_writes_links(fs: FoodScholar) -> None:
    fs.annotate()
    c1 = fs.graph.chunk("c1")
    assert any(link.ontology_id == "TEST:0000008" for link in c1.entity_links)


def test_annotate_denormalizes_foodon_ids(fs: FoodScholar) -> None:
    fs.annotate()
    c1 = fs.graph.chunk("c1")
    c2 = fs.graph.chunk("c2")
    assert "TEST:0000008" in c1.foodon_ids
    assert "TEST:0000006" in c2.foodon_ids


def test_annotate_writes_embedding(fs: FoodScholar) -> None:
    fs.annotate()
    c1 = fs.graph.chunk("c1")
    assert c1.embedding is not None
    assert len(c1.embedding) > 0
    assert c1.embedding_model is not None


def test_annotate_stamps_enrichment_version(fs: FoodScholar) -> None:
    fs.annotate()
    c1 = fs.graph.chunk("c1")
    assert c1.enrichment_version == "annotate-v1"


def test_annotate_returns_artifact_meta(fs: FoodScholar) -> None:
    meta = fs.annotate()
    assert meta.phase == "annotate"
    assert meta.record_count == 2
    assert meta.artifact_id.startswith("annotate-")


def test_annotate_is_idempotent(fs: FoodScholar) -> None:
    fs.annotate()
    first_links = [link.ontology_id for link in fs.graph.chunk("c1").entity_links]
    fs.annotate()
    second_links = [link.ontology_id for link in fs.graph.chunk("c1").entity_links]
    assert first_links == second_links


def test_dry_run_returns_mentions_and_links(fs: FoodScholar) -> None:
    mentions, links = dry_run(
        "Mediterranean diet rich in olive oil.",
        ner=fs.ner,
        linker=fs.linker,
    )
    assert any(m.text.lower() == "olive oil" for m in mentions)
    assert any(link.ontology_id == "TEST:0000008" for link in links)


def test_run_function_directly(fs: FoodScholar) -> None:
    # Bypass the facade method to confirm runner works with raw protocols.
    meta = run(
        fs.chunk_store,
        ner=fs.ner,
        linker=fs.linker,
        embedder=fs.embedder,
        config=fs.config,
    )
    assert meta.record_count == 2


def test_facade_linker_dry_run(fs: FoodScholar) -> None:
    link = fs.linker.dry_run("evo")
    # Fuzzy should resolve "evo" to olive oil via the EVOO synonym
    assert link is not None
    assert link.ontology_id == "TEST:0000008"
    assert link.method == "lexical_fuzzy"


def test_attach_ner_overrides_default(fs: FoodScholar) -> None:
    from foodscholar.annotate.ner import KeywordNER

    custom = KeywordNER(["olive oil"])  # only matches one term
    fs.attach_ner(custom)
    fs.annotate()
    c2 = fs.graph.chunk("c2")
    # apple is no longer in the NER vocabulary
    assert "TEST:0000006" not in c2.foodon_ids


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
