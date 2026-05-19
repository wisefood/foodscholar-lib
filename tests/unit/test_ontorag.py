"""Tests for the OntoRAG tri-hybrid retriever.

Indexes are built over the mini FoodOn fixture with `HashEmbedder` standing in
for MiniLM/SapBERT — deterministic, fast, no model download. The focus is the
retrieval + RRF-merge logic; real-model behavior is a slow integration test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foodscholar.annotate.embedder import HashEmbedder
from foodscholar.annotate.ontorag import OntoRagRetriever, RetrievedCandidate, build_index
from foodscholar.ontology import FoodOnAPI, load_ontology

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def api() -> FoodOnAPI:
    return FoodOnAPI(load_ontology(FIXTURES / "mini_foodon.obo"), prefix_filter=None)


@pytest.fixture
def retriever(api: FoodOnAPI, tmp_path: Path) -> OntoRagRetriever:
    minilm = HashEmbedder(dim=16)
    sapbert = HashEmbedder(dim=24)
    index = build_index(api, minilm=minilm, sapbert=sapbert, index_dir=tmp_path / "idx")
    return OntoRagRetriever(index, api, minilm=minilm, sapbert=sapbert)


# ---------------------------------------------------------------- index build


def test_build_index_excludes_obsolete(api: FoodOnAPI, tmp_path: Path) -> None:
    index = build_index(
        api, minilm=HashEmbedder(dim=8), sapbert=HashEmbedder(dim=8),
        index_dir=tmp_path / "idx",
    )
    # mini fixture: 11 terms, 1 obsolete
    assert index.size == 10


def test_build_index_persists_and_reloads(api: FoodOnAPI, tmp_path: Path) -> None:
    d = tmp_path / "idx"
    minilm, sapbert = HashEmbedder(dim=8), HashEmbedder(dim=12)
    first = build_index(api, minilm=minilm, sapbert=sapbert, index_dir=d)
    assert (d / "meta.json").exists()
    assert (d / "minilm.faiss").exists()
    assert (d / "sapbert.faiss").exists()
    # Second build hits the cache (same fingerprint) — same term set.
    second = build_index(api, minilm=minilm, sapbert=sapbert, index_dir=d)
    assert second.term_ids == first.term_ids


def test_build_index_rebuilds_on_embedder_change(api: FoodOnAPI, tmp_path: Path) -> None:
    d = tmp_path / "idx"
    build_index(api, minilm=HashEmbedder(dim=8), sapbert=HashEmbedder(dim=8), index_dir=d)

    class _OtherEmbedder(HashEmbedder):
        model_id = "other-embedder"

    # Different embedder id → fingerprint mismatch → rebuild, still usable.
    rebuilt = build_index(
        api, minilm=_OtherEmbedder(dim=8), sapbert=HashEmbedder(dim=8), index_dir=d
    )
    assert rebuilt.size == 10


# ---------------------------------------------------------------- retrieval


def test_retrieve_returns_candidates(retriever: OntoRagRetriever) -> None:
    cands = retriever.retrieve("olive oil", k=5)
    assert cands
    assert all(isinstance(c, RetrievedCandidate) for c in cands)


def test_retrieve_respects_k(retriever: OntoRagRetriever) -> None:
    assert len(retriever.retrieve("olive", k=2)) <= 2


def test_retrieve_lexical_arm_finds_exact_label(retriever: OntoRagRetriever) -> None:
    # "olive oil" is a verbatim label — the lexical (Whoosh) arm must surface it.
    cands = retriever.retrieve("olive oil", k=5)
    assert any(c.label == "olive oil" for c in cands)


def test_retrieve_candidates_ranked_by_fusion_score(retriever: OntoRagRetriever) -> None:
    cands = retriever.retrieve("olive", k=5)
    scores = [c.fusion_score for c in cands]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_records_contributing_sources(retriever: OntoRagRetriever) -> None:
    cands = retriever.retrieve("olive oil", k=5)
    for c in cands:
        assert c.sources, "every candidate must record >=1 contributing arm"
        assert c.source in {"lexical", "minilm", "sapbert"}
        assert c.source in c.sources


def test_retrieve_empty_query_returns_empty(retriever: OntoRagRetriever) -> None:
    assert retriever.retrieve("", k=5) == []
    assert retriever.retrieve("   ", k=5) == []


def test_retrieve_label_populated(retriever: OntoRagRetriever) -> None:
    # Every candidate carries a human-readable label, not just an id.
    for c in retriever.retrieve("apple", k=3):
        assert c.label and not c.label.startswith("TEST:")
