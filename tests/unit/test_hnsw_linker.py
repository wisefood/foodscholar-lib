"""Unit tests for HNSWLinker — uses a fake NELIndex (no real hnswlib/encoder)."""

from __future__ import annotations

from foodscholar.annotate.linker import HNSWLinker
from foodscholar.io.chunk import Mention


class _FakeNELIndex:
    backend_id = "fake-nel-v0"

    def __init__(self, surface_to_hit: dict[str, tuple[str, float] | None]) -> None:
        self._table = surface_to_hit
        self.batch_calls: list[int] = []

    def link(self, surface: str) -> tuple[str, float] | None:
        return self._table.get(surface.strip(), None)

    def link_batch(self, surfaces: list[str]) -> list[tuple[str, float] | None]:
        self.batch_calls.append(len(surfaces))
        return [self._table.get(s.strip(), None) for s in surfaces]


def _mention(text: str) -> Mention:
    return Mention(
        text=text, start=0, end=len(text), score=1.0, ner_model_version="test"
    )


def test_link_returns_entity_link_above_threshold() -> None:
    linker = HNSWLinker(_FakeNELIndex({"olive oil": ("FOODON:O1", 0.85)}))
    link = linker.link(_mention("olive oil"))
    assert link is not None
    assert link.ontology_id == "FOODON:O1"
    assert link.method == "dense"
    assert link.confidence == 0.85


def test_link_returns_none_below_threshold() -> None:
    linker = HNSWLinker(
        _FakeNELIndex({"olive oil": ("FOODON:O1", 0.50)}), min_sim=0.70
    )
    assert linker.link(_mention("olive oil")) is None


def test_link_returns_none_for_empty_mention() -> None:
    linker = HNSWLinker(_FakeNELIndex({"": ("FOODON:O1", 0.99)}))
    m = Mention(text="   ", start=0, end=3, score=1.0, ner_model_version="test")
    assert linker.link(m) is None


def test_link_many_uses_batch_path() -> None:
    nel = _FakeNELIndex(
        {
            "olive oil": ("FOODON:O1", 0.85),
            "apple": ("FOODON:A1", 0.82),
            "ufo": None,
        }
    )
    linker = HNSWLinker(nel)
    out = linker.link_many([_mention("olive oil"), _mention("apple"), _mention("ufo")])
    assert nel.batch_calls == [3]
    assert out[0] is not None and out[0].ontology_id == "FOODON:O1"
    assert out[1] is not None and out[1].ontology_id == "FOODON:A1"
    assert out[2] is None


def test_dry_run_constructs_mention_from_text() -> None:
    linker = HNSWLinker(_FakeNELIndex({"olive oil": ("FOODON:O1", 0.88)}))
    link = linker.dry_run("olive oil")
    assert link is not None
    assert link.ontology_id == "FOODON:O1"
    assert link.mention.text == "olive oil"


def test_linker_id_includes_backend() -> None:
    linker = HNSWLinker(_FakeNELIndex({}))
    assert "fake-nel-v0" in linker.linker_id
