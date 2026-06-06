"""Layer B tuning sweep + scoring."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("igraph")
pytest.importorskip("leidenalg")

from foodscholar import FoodScholar  # noqa: E402
from foodscholar.config import FoodScholarConfig, LayerBConfig  # noqa: E402
from foodscholar.io.chunk import Chunk  # noqa: E402
from foodscholar.io.graph import Shelf  # noqa: E402
from foodscholar.layer_b.models import LayerBQualityReport  # noqa: E402
from foodscholar.layer_b.sweep import score_report, sweep_layer_b  # noqa: E402
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore  # noqa: E402

# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------


def _report(**kw) -> LayerBQualityReport:
    base = {"facet": "foods", "n_shelves": 2, "n_themes": 6}
    base.update(kw)
    return LayerBQualityReport(**base)


def test_score_rewards_coverage_and_merges() -> None:
    cfg = LayerBConfig()
    lo = _report(theme_coverage=0.3, n_merged=1)
    hi = _report(theme_coverage=0.9, n_merged=4)
    assert score_report(hi, cfg) > score_report(lo, cfg)


def test_score_penalizes_noise() -> None:
    cfg = LayerBConfig()
    clean = _report(theme_coverage=0.8, n_merged=3)
    noisy = _report(
        theme_coverage=0.8,
        n_merged=3,
        n_duplicate_label_themes=4,
        n_tiny_themes=4,
        n_cross_shelf_leakage=3,
    )
    assert score_report(clean, cfg) > score_report(noisy, cfg)


def test_score_monotonic_strictly_better() -> None:
    cfg = LayerBConfig()
    worse = _report(theme_coverage=0.5, n_merged=2, n_tiny_themes=2, n_cross_shelf_leakage=1)
    better = _report(theme_coverage=0.7, n_merged=3, n_tiny_themes=0, n_cross_shelf_leakage=0)
    assert score_report(better, cfg) > score_report(worse, cfg)


# ----------------------------------------------------------------------------
# sweep (tiny grid, real dry-run builds)
# ----------------------------------------------------------------------------


@pytest.fixture
def sweepable_fs():
    """Two shelves, each with a tight 4-chunk embedding cluster, so per-shelf
    Pass 1 produces themes — enough for the sweep to score."""
    chunk_store = InMemoryChunkStore()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_shelves([
        Shelf(shelf_id="shelf:a", label="a", facet="foods", depth=1, chunk_count=4),
        Shelf(shelf_id="shelf:b", label="b", facet="foods", depth=1, chunk_count=4),
    ])

    def vec(base, i):
        v = np.array(base, dtype=float) + np.array([0.01 * i, 0.0, 0.0])
        return (v / np.linalg.norm(v)).tolist()

    chunks = []
    for i in range(4):
        chunks.append(Chunk(
            chunk_id=f"a{i}", text=f"calcium dairy {i}", source_doc_id="d",
            source_type="textbook", section_type="other",
            embedding=vec([1, 0, 0], i), embedding_model="m",
        ))
    for i in range(4):
        chunks.append(Chunk(
            chunk_id=f"b{i}", text=f"omega fish oil {i}", source_doc_id="d",
            source_type="textbook", section_type="other",
            embedding=vec([0, 1, 0], i), embedding_model="m",
        ))
    chunk_store.upsert(chunks)
    graph_store.attach_chunks_to_shelf("shelf:a", [(f"a{i}", []) for i in range(4)])
    graph_store.attach_chunks_to_shelf("shelf:b", [(f"b{i}", []) for i in range(4)])

    cfg = FoodScholarConfig.model_validate({
        "corpus": {"chunks_path": "data/chunks.parquet"},
        "storage": {
            "chunk_store": {"backend": "memory"},
            "graph_store": {"backend": "memory"},
        },
        "layer_b": {
            "min_chunks_per_shelf": 2,
            "min_embedded_fraction": 0.0,
            "similarity": {"knn_k": 3},
        },
    })
    return FoodScholar(cfg, chunk_store=chunk_store, graph_store=graph_store)


def test_sweep_returns_ranked_rows_and_does_not_persist(sweepable_fs) -> None:
    fs = sweepable_fs
    grid = {
        "leiden.min_community_size": [2, 3],
        "similarity.edge_threshold": [0.40, 0.55],
    }
    result = sweep_layer_b(fs, facet="foods", grid=grid)

    assert len(result.rows) == 4  # 2 x 2
    # Ranked best-first.
    scores = [r.score for r in result.rows]
    assert scores == sorted(scores, reverse=True)
    assert result.best is not None
    assert set(result.best.keys()) == {"leiden.min_community_size", "similarity.edge_threshold"}

    # Non-mutating: store untouched, original config restored.
    assert fs.graph_store.list_themes() == []
    assert fs.config.layer_b.labeling.strategy == "llm"  # default, not overwritten
    assert fs.config.layer_b.leiden.min_community_size == 15  # original default

    # Markdown renders.
    assert "sweep" in str(result).lower()


def test_sweep_restores_config_even_if_build_raises(sweepable_fs, monkeypatch) -> None:
    fs = sweepable_fs
    original = fs.config.layer_b

    import foodscholar.layer_b.sweep as sweep_mod

    def boom(*a, **k):
        raise RuntimeError("build blew up")

    monkeypatch.setattr(sweep_mod, "_build", boom, raising=False)
    # _build is imported inside the function, so patch the builder module too.
    import foodscholar.layer_b.builder as builder_mod
    monkeypatch.setattr(builder_mod, "build_layer_b", boom)

    with pytest.raises(RuntimeError):
        sweep_layer_b(fs, facet="foods", grid={"leiden.resolution": [1.0]})

    assert fs.config.layer_b is original
