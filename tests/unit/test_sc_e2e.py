"""End-to-end consolidate(): cluster judging, N-way merge, block-list."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from foodscholar.config import SemanticConsolidationConfig
from foodscholar.io.graph import Shelf
from foodscholar.layer_a.attach import ShelfIndex
from foodscholar.layer_a.semantic_consolidation import consolidate
from foodscholar.ontology import FoodOnAPI, load_ontology
from foodscholar.storage.memory import InMemoryChunkStore


def _mini_foodon() -> FoodOnAPI:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"
    return FoodOnAPI(load_ontology(path), prefix_filter=None)


class LabelEmbedder:
    """Deterministic embedder: olive-oil variants cluster; apple is distinct."""

    model_id = "label-embedder-v0"
    dim = 3

    _VECTORS: ClassVar[dict[str, list[float]]] = {
        "olive oil": [1.0, 0.0, 0.0],
        "pressed olive fruit fat": [0.99, 0.02, 0.0],
        "liquid olive lipid": [0.98, 0.03, 0.0],
        "apple": [0.0, 1.0, 0.0],
    }

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._VECTORS.get(t.split(" | ")[0], [0.0, 0.0, 1.0]) for t in texts]


class ClusterJudge:
    """Merges the whole olive-oil cluster; keeps apple alone."""

    model_id = "cluster-judge-v0"

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:  # pragma: no cover
        return "{}"

    def generate_json(self, prompt, schema, max_tokens=1024):
        # All shelves in the cluster prompt are olive-oil variants → merge all.
        # The prompt numbers them Shelf 1..N; merge every index.
        n = prompt.count("Shelf ")
        # subtract the few-shot mentions of "Shelf" — none in our examples block
        members = list(range(1, n + 1))
        return {"merge_groups": [{"members": members,
                                  "canonical_name": "olive oil",
                                  "confidence": 0.92, "rationale": "same oil"}],
                "keep_alone": []}


def _shelves() -> list[Shelf]:
    return [
        Shelf(shelf_id="facet:foods", label="Foods", facet="foods", depth=0,
              foodon_id=None),
        Shelf(shelf_id="foodon:8", label="olive oil", facet="foods", depth=2,
              foodon_id="TEST:0000008", support_direct=10, support_lifted=12),
        Shelf(shelf_id="foodon:7", label="pressed olive fruit fat", facet="foods",
              depth=2, foodon_id="TEST:0000007", support_direct=3, support_lifted=3),
        Shelf(shelf_id="foodon:11", label="liquid olive lipid", facet="foods",
              depth=2, foodon_id="TEST:0000011", support_direct=2, support_lifted=2),
        Shelf(shelf_id="foodon:6", label="apple", facet="foods", depth=2,
              foodon_id="TEST:0000006"),
    ]


def _cfg(**kw):
    base = dict(enabled=True, cosine_threshold=0.9, auto_merge_confidence=0.80,
                subtype_patterns=[], max_synonyms=0)
    base.update(kw)
    return SemanticConsolidationConfig(**base)


def test_e2e_merges_whole_cluster_nway() -> None:
    shelves = _shelves()
    new_shelves, art = consolidate(
        shelves, InMemoryChunkStore(), _mini_foodon(), LabelEmbedder(),
        ClusterJudge(), _cfg(), "deadbeef", facet="foods",
    )
    ids = {s.shelf_id for s in new_shelves}
    # Three olive-oil variants collapse to one; apple + root remain.
    assert ids == {"facet:foods", "foodon:8", "foodon:6"}
    assert art.cluster_count == 1            # one connected component
    assert len(art.applied_groups) == 1
    assert art.shelves_removed == 2          # 3 members -> 1 canonical

    canonical = next(s for s in new_shelves if s.shelf_id == "foodon:8")
    assert "TEST:0000007" in canonical.see_also
    assert "TEST:0000011" in canonical.see_also
    idx = ShelfIndex.from_shelves(new_shelves)
    assert idx.per_facet["foods"].by_seealso["TEST:0000007"] is canonical
    assert idx.per_facet["foods"].by_seealso["TEST:0000011"] is canonical


def test_e2e_blocklist_vetoes_merge() -> None:
    # Block the olive oil + pressed-olive-fat pair → whole cluster blocked.
    cfg = _cfg(permanent_block_list=[("TEST:0000008", "TEST:0000007")])
    shelves = _shelves()
    new_shelves, art = consolidate(
        shelves, InMemoryChunkStore(), _mini_foodon(), LabelEmbedder(),
        ClusterJudge(), cfg, "deadbeef", facet="foods",
    )
    # Nothing merged — every original shelf survives.
    assert len(new_shelves) == len(shelves)
    assert art.applied_groups == []
    assert len(art.blocked_groups) == 1


def test_dry_run_changes_nothing() -> None:
    shelves = _shelves()
    new_shelves, art = consolidate(
        shelves, InMemoryChunkStore(), _mini_foodon(), LabelEmbedder(),
        ClusterJudge(), _cfg(), "deadbeef", facet="foods", dry_run=True,
    )
    assert new_shelves is shelves
    assert len(art.applied_groups) == 1  # decision still recorded


def test_judge_disabled_skips_llm() -> None:
    class Boom:
        model_id = "boom"

        def generate(self, *a, **k):  # pragma: no cover
            raise AssertionError("LLM must not be called")

        def generate_json(self, *a, **k):
            raise AssertionError("LLM must not be called")

    new_shelves, art = consolidate(
        _shelves(), InMemoryChunkStore(), _mini_foodon(), LabelEmbedder(),
        Boom(), _cfg(judge_enabled=False), "deadbeef", facet="foods",
    )
    assert art.cluster_count == 1       # cluster found
    assert art.applied_groups == []     # but never judged
    assert len(new_shelves) == len(_shelves())
