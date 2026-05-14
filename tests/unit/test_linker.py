"""Tests for ThreeTierLinker (BRIEF §2 linker)."""

from __future__ import annotations

from pathlib import Path

import pytest

from foodscholar.annotate.linker import ThreeTierLinker
from foodscholar.io.chunk import Mention
from foodscholar.ontology import FoodOnAPI, load_ontology
from foodscholar.storage.protocols import Linker

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def api() -> FoodOnAPI:
    return FoodOnAPI(load_ontology(FIXTURES / "mini_foodon.obo"))


def _mention(text: str) -> Mention:
    return Mention(
        text=text, start=0, end=len(text), score=1.0, ner_model_version="test"
    )


def test_linker_implements_protocol(api: FoodOnAPI) -> None:
    assert isinstance(ThreeTierLinker(api), Linker)


def test_exact_label_match_returns_full_confidence(api: FoodOnAPI) -> None:
    link = ThreeTierLinker(api).link(_mention("olive oil"))
    assert link is not None
    assert link.ontology_id == "TEST:0000008"
    assert link.method == "lexical_exact"
    assert link.confidence == 1.0


def test_exact_synonym_match(api: FoodOnAPI) -> None:
    link = ThreeTierLinker(api).link(_mention("EVOO"))
    assert link is not None
    assert link.ontology_id == "TEST:0000008"
    assert link.method == "lexical_exact"


def test_exact_match_is_case_insensitive(api: FoodOnAPI) -> None:
    assert ThreeTierLinker(api).link(_mention("APPLE")).ontology_id == "TEST:0000006"


def test_obsolete_term_does_not_resolve(api: FoodOnAPI) -> None:
    assert ThreeTierLinker(api).link(_mention("legacy term")) is None


def test_fuzzy_plural(api: FoodOnAPI) -> None:
    link = ThreeTierLinker(api).link(_mention("olives"))
    assert link is not None
    assert link.ontology_id == "TEST:0000007"
    assert link.method == "lexical_fuzzy"
    assert 0.85 <= link.confidence < 1.0


def test_fuzzy_typo(api: FoodOnAPI) -> None:
    link = ThreeTierLinker(api).link(_mention("oliv oil"))
    assert link is not None
    assert link.ontology_id == "TEST:0000008"
    assert link.method == "lexical_fuzzy"


def test_fuzzy_threshold_blocks_low_quality(api: FoodOnAPI) -> None:
    # Crank threshold; "evo" should no longer make it.
    linker = ThreeTierLinker(api, fuzzy_threshold=0.95)
    assert linker.link(_mention("evo")) is None


def test_unrecognized_returns_none(api: FoodOnAPI) -> None:
    assert ThreeTierLinker(api).link(_mention("quinoa")) is None


def test_empty_text_returns_none(api: FoodOnAPI) -> None:
    assert ThreeTierLinker(api).link(_mention("   ")) is None


def test_dry_run_builds_mention(api: FoodOnAPI) -> None:
    link = ThreeTierLinker(api).dry_run("olive oil")
    assert link is not None
    assert link.method == "lexical_exact"


# ------------------------------------------------------------ dense tier


class _OilEmbedder:
    """Toy embedder where any string containing 'oil' shares a vector."""

    model_id = "toy-oil"

    @property
    def dim(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0, 1.0 if "oil" in t.lower() else 0.0] for t in texts
        ]


def test_dense_tier_falls_through_when_lexical_misses(api: FoodOnAPI) -> None:
    # Crank fuzzy threshold so it doesn't fire; let dense pick up.
    linker = ThreeTierLinker(
        api,
        fuzzy_threshold=0.99,
        dense_embedder=_OilEmbedder(),
        dense_threshold=0.5,
    )
    link = linker.link(_mention("oil-based culinary fat"))
    assert link is not None
    assert link.method == "dense"


def test_dense_threshold_blocks_low_similarity(api: FoodOnAPI) -> None:
    class _OrthoEmbedder:
        """Maps each text to a one-hot vector by hash, so unrelated text -> low cosine."""

        model_id = "ortho"

        @property
        def dim(self) -> int:
            return 64

        def embed(self, texts: list[str]) -> list[list[float]]:
            import hashlib

            out = []
            for t in texts:
                v = [0.0] * 64
                v[hash(t) % 64] = 1.0
                # Add a noise component so multi-word terms aren't all orthogonal.
                v[int(hashlib.md5(t.encode()).hexdigest(), 16) % 64] = 0.5
                out.append(v)
            return out

    linker = ThreeTierLinker(
        api,
        fuzzy_threshold=0.99,
        dense_embedder=_OrthoEmbedder(),
        dense_threshold=0.99,
    )
    assert linker.link(_mention("totally unrelated phrase that wont match")) is None


def test_no_dense_embedder_means_no_dense_tier(api: FoodOnAPI) -> None:
    linker = ThreeTierLinker(api, fuzzy_threshold=0.99)
    # With dense disabled and fuzzy too strict, "oliv oil" should miss
    assert linker.link(_mention("oliv oil")) is None
