"""Layer C builder: themes -> cards, skip/fail accounting, dry_run."""

from __future__ import annotations

from foodscholar.config import FoodScholarConfig, LayerCConfig
from foodscholar.io.chunk import Chunk
from foodscholar.io.graph import Shelf, Theme
from foodscholar.layer_c.builder import build_layer_c
from foodscholar.storage.memory import (
    InMemoryCardStore,
    InMemoryChunkStore,
    InMemoryGraphStore,
)


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, text=text, source_doc_id="d",
                 source_type="abstract", section_type="abstract")


def _theme(tid: str) -> Theme:
    return Theme(theme_id=tid, label="Oats", shelf_ids=["s1"], chunk_count=2,
                 discovered_by="leiden", discovery_version="v", facet="foods",
                 discovery_pass="merged", keyword_terms=["oat", "fiber"])


class _OKJsonLLM:
    model_id = "stub"

    def generate(self, prompt, max_tokens=1024):  # pragma: no cover
        return ""

    def generate_json(self, prompt, schema, max_tokens=1024):
        return {"title": "Oats", "summary": "Oats have beta glucan.",
                "tip": None, "evidence_quality": "high",
                "controversy_note": None, "confidence_note": None}


class _FailLLM(_OKJsonLLM):
    def generate_json(self, prompt, schema, max_tokens=1024):
        raise RuntimeError("llm down")


def _fs(llm):
    """Minimal stand-in for the FoodScholar facade the builder needs."""
    cs = InMemoryChunkStore()
    gs = InMemoryGraphStore()
    cs.upsert([_chunk("c1", "Oats are a whole grain. They have fiber."),
               _chunk("c2", "Beta glucan lowers cholesterol.")])
    gs.upsert_shelves([Shelf(shelf_id="s1", label="cereal", facet="foods", depth=1)])
    gs.upsert_themes([_theme("t1")])
    # signature is (chunk_id, theme_id, primary, weight)
    gs.attach_chunks_to_themes_bulk([("c1", "t1", True, 1.0), ("c2", "t1", False, 1.0)])

    class _Embedder:
        model_id = "test-embedder"

        @property
        def dim(self) -> int:
            return 3

        def embed(self, texts: list[str]) -> list[list[float]]:
            # deterministic 3-d vector per text (length-based, non-zero)
            return [[float(len(t)), 1.0, 0.5] for t in texts]

    class _FS:
        pass

    fs = _FS()
    fs.chunk_store = cs
    fs.graph_store = gs
    fs.card_store = InMemoryCardStore()
    fs.embedder = _Embedder()
    from foodscholar.graph_view import GraphView
    fs.graph = GraphView(cs, gs)
    fs.llm = llm
    fs.config = FoodScholarConfig(corpus={"chunks_path": "x"})
    fs.config.layer_c = LayerCConfig()
    return fs


def test_build_creates_card_per_theme() -> None:
    fs = _fs(_OKJsonLLM())
    rep = build_layer_c(fs)
    assert rep.n_themes == 1
    assert rep.n_cards == 1
    assert rep.n_failed == 0
    assert fs.graph_store.get_card("t1", "theme") is not None


def test_build_embeds_and_persists_to_card_store() -> None:
    fs = _fs(_OKJsonLLM())
    build_layer_c(fs)
    cards = fs.card_store.get_many(["card:theme:t1"])
    assert len(cards) == 1
    assert cards[0].embedding is not None
    assert cards[0].embedding_model == "test-embedder"
    # the card is vector-searchable
    hits = fs.card_store.knn_search_cards(cards[0].embedding, k=1)
    assert hits and hits[0][0] == "card:theme:t1"


def test_dry_run_persists_nothing() -> None:
    fs = _fs(_OKJsonLLM())
    rep = build_layer_c(fs, dry_run=True)
    assert rep.n_cards == 1
    assert fs.graph_store.get_card("t1", "theme") is None
    assert fs.card_store.get_many(["card:theme:t1"]) == []


def test_llm_failure_counted_not_raised() -> None:
    fs = _fs(_FailLLM())
    rep = build_layer_c(fs)
    assert rep.n_failed == 1
    assert rep.n_cards == 0
