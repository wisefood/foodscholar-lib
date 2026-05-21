"""Tests for the domain-expert quality report.

Quality isn't pass/fail — these tests pin the shape of the report and the
behavior of each section's heuristics, not "the graph is good."
"""

from __future__ import annotations

from pathlib import Path

from foodscholar import FoodScholar
from foodscholar.config import FoodScholarConfig, LayerAConfig
from foodscholar.evaluation.quality import (
    QualityReport,
    quality_report,
)
from foodscholar.io.chunk import Chunk, EntityLink, Mention
from foodscholar.io.graph import Shelf
from foodscholar.layer_a import attach, build_layer_a
from foodscholar.ontology import FoodOnAPI, load_ontology
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore

# ---------------------------------------------------------------- fixtures


def _mini_foodon() -> FoodOnAPI:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"
    return FoodOnAPI(load_ontology(path), prefix_filter=None)


def _full_config(layer_a: LayerAConfig | None = None) -> FoodScholarConfig:
    data: dict = {
        "corpus": {"chunks_path": "data/chunks.parquet"},
        "storage": {
            "chunk_store": {"backend": "memory"},
            "graph_store": {"backend": "memory"},
        },
    }
    cfg = FoodScholarConfig.model_validate(data)
    if layer_a is not None:
        cfg = cfg.model_copy(update={"layer_a": layer_a})
    return cfg


def _chunk(chunk_id: str, foodon_ids: list[str], *, text: str = "lorem ipsum") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id="d",
        source_type="abstract",
        section_type="abstract",
        foodon_ids=foodon_ids,
        entity_links=[
            EntityLink(
                mention=Mention(
                    text=fid, start=0, end=1, score=1.0,
                    ner_model_version="t", entity_type="food",
                ),
                ontology_id=fid,
                confidence=1.0,
                method="dense",
                linker_version="t",
            )
            for fid in foodon_ids
        ],
    )


def _build() -> tuple[InMemoryChunkStore, InMemoryGraphStore, FoodOnAPI]:
    chunk_store = InMemoryChunkStore()
    chunk_store.upsert([
        _chunk("c1", ["TEST:0000008"], text="olive oil is rich in antioxidants"),
        _chunk("c2", ["TEST:0000006"], text="apples contain quercetin"),
        _chunk("c3", ["TEST:0000008", "TEST:0000004"], text="EVOO and fruit"),
    ])
    ontology = _mini_foodon()
    graph_store = InMemoryGraphStore()
    cfg = LayerAConfig(
        min_support=1,
        max_depth=5,
        collapse_single_child_chains=False,
        facets=["foods"],
        umbrella_direct_share_max=0.0,
    )
    full_cfg = _full_config(cfg)
    build_layer_a(chunk_store, graph_store, ontology, config=cfg, full_config=full_cfg)
    attach(chunk_store, graph_store, ontology, full_config=full_cfg)
    return chunk_store, graph_store, ontology


# ---------------------------------------------------------------- report shape


def test_quality_report_has_all_sections() -> None:
    chunk_store, graph_store, ontology = _build()
    report = quality_report(
        chunk_store, graph_store, ontology, config_hash="t", facet="foods"
    )
    assert isinstance(report, QualityReport)
    assert report.facet == "foods"
    assert report.n_shelves > 0
    # All five sections populated (or, when empty, defaulted to []).
    assert isinstance(report.top_shelves, list)
    assert isinstance(report.hierarchy_walkthrough, list)
    assert isinstance(report.suspicious_shelves, list)
    assert isinstance(report.canonical_vocab_check, list)
    assert isinstance(report.chunk_sample, list)


def test_quality_report_is_json_roundtrippable() -> None:
    chunk_store, graph_store, ontology = _build()
    report = quality_report(chunk_store, graph_store, ontology, config_hash="t")
    j = report.model_dump_json()
    restored = QualityReport.model_validate_json(j)
    assert restored.config_hash == "t"
    assert restored.n_shelves == report.n_shelves


def test_quality_report_pretty_prints_markdown_sections() -> None:
    chunk_store, graph_store, ontology = _build()
    s = str(quality_report(chunk_store, graph_store, ontology, config_hash="t"))
    assert "# Layer A quality report" in s
    assert "## 1. Top shelves" in s
    assert "## 2. Hierarchy walkthrough" in s
    assert "## 3. Suspicious shelves" in s
    assert "## 4. Canonical vocabulary check" in s
    assert "## 5. Random chunk sample" in s


# ---------------------------------------------------------------- 1. top shelves


def test_top_shelves_sorted_by_chunk_count_with_snippets() -> None:
    chunk_store, graph_store, ontology = _build()
    report = quality_report(
        chunk_store, graph_store, ontology, config_hash="t", top_n=10
    )
    counts = [s.chunk_count for s in report.top_shelves]
    assert counts == sorted(counts, reverse=True), "top shelves must be sorted desc"
    # At least one shelf has a sample snippet.
    assert any(s.sample_chunks for s in report.top_shelves)


# ---------------------------------------------------------------- 2. hierarchy


def test_hierarchy_walkthrough_includes_ancestors() -> None:
    chunk_store, graph_store, ontology = _build()
    report = quality_report(chunk_store, graph_store, ontology, config_hash="t")
    # At least one entry has a non-empty ancestor chain (the deepest shelf).
    assert any(h.ancestor_chain for h in report.hierarchy_walkthrough)


# ---------------------------------------------------------------- 3. suspicious


def test_flags_efsa_code_label() -> None:
    chunk_store = InMemoryChunkStore()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_shelves([
        Shelf(
            shelf_id="foodon:FOODON:1", label="10210 - legumes (efsa foodex2)",
            facet="foods", depth=1, foodon_id="FOODON:1",
            parent_shelf_id=None, chunk_count=10,
        ),
    ])
    ontology = _mini_foodon()
    report = quality_report(chunk_store, graph_store, ontology, config_hash="t")
    flagged_labels = [s.label for s in report.suspicious_shelves]
    assert "10210 - legumes (efsa foodex2)" in flagged_labels


def test_flags_datum_label() -> None:
    chunk_store = InMemoryChunkStore()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_shelves([
        Shelf(
            shelf_id="foodon:FOODON:2", label="food calorie datum",
            facet="foods", depth=1, foodon_id="FOODON:2",
            parent_shelf_id=None, chunk_count=10,
        ),
    ])
    ontology = _mini_foodon()
    report = quality_report(chunk_store, graph_store, ontology, config_hash="t")
    flagged_reasons = [s.reason for s in report.suspicious_shelves]
    assert any("datum" in r for r in flagged_reasons)


def test_flags_zero_chunk_shelf() -> None:
    chunk_store = InMemoryChunkStore()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_shelves([
        Shelf(
            shelf_id="foodon:FOODON:3", label="kale",
            facet="foods", depth=1, foodon_id="FOODON:3",
            parent_shelf_id=None, chunk_count=0,
        ),
    ])
    ontology = _mini_foodon()
    report = quality_report(chunk_store, graph_store, ontology, config_hash="t")
    flagged_labels = [s.label for s in report.suspicious_shelves]
    assert "kale" in flagged_labels


def test_clean_shelf_is_not_flagged() -> None:
    chunk_store, graph_store, ontology = _build()
    report = quality_report(chunk_store, graph_store, ontology, config_hash="t")
    # Plain mini-fixture shelves (e.g. "olive oil", "apple") should NOT trigger
    # the EFSA / datum / raw heuristics. zero-chunk could still fire for some
    # — that's expected. Just check that obvious-good labels aren't flagged for
    # the wrong reason.
    olive_flag = next(
        (s for s in report.suspicious_shelves if s.label == "olive oil"), None
    )
    if olive_flag is not None:
        # Only acceptable flag: zero-chunk. Anything else is a false positive.
        assert "zero" in olive_flag.reason, (
            f"olive oil flagged for wrong reason: {olive_flag.reason}"
        )


# ---------------------------------------------------------------- 4. vocab


def test_canonical_vocab_records_found_in_ontology_only() -> None:
    """A term in FoodOn but not on any shelf is correctly classified."""
    chunk_store, graph_store, ontology = _build()
    # Override the canonical list to one term we know exists in the fixture.
    report = quality_report(
        chunk_store, graph_store, ontology,
        config_hash="t",
        canonical_terms=("olive oil", "completely-unknown-term-xyz"),
    )
    by_term = {v.term: v for v in report.canonical_vocab_check}
    assert by_term["olive oil"].status in (
        "found_as_shelf", "found_in_ontology_only"
    )
    assert by_term["completely-unknown-term-xyz"].status == "not_found"


def test_canonical_vocab_found_as_shelf_includes_chunk_count() -> None:
    chunk_store, graph_store, ontology = _build()
    report = quality_report(
        chunk_store, graph_store, ontology,
        config_hash="t",
        canonical_terms=("olive oil",),
    )
    v = report.canonical_vocab_check[0]
    if v.status == "found_as_shelf":
        assert v.shelf_label is not None
        assert v.shelf_id is not None
        assert v.chunk_count >= 0


# ---------------------------------------------------------------- 5. sample


def test_chunk_sample_uses_seed_for_reproducibility() -> None:
    chunk_store, graph_store, ontology = _build()
    r1 = quality_report(
        chunk_store, graph_store, ontology, config_hash="t", sample_size=2, seed=42
    )
    r2 = quality_report(
        chunk_store, graph_store, ontology, config_hash="t", sample_size=2, seed=42
    )
    ids_1 = [c.chunk_id for c in r1.chunk_sample]
    ids_2 = [c.chunk_id for c in r2.chunk_sample]
    assert ids_1 == ids_2, "same seed must produce identical samples"


def test_chunk_sample_only_includes_attached_chunks() -> None:
    chunk_store = InMemoryChunkStore()
    # One attached, one not.
    chunk_store.upsert([
        _chunk("c1", ["TEST:0000008"]),
        _chunk("c-orphan", []),  # no foodon_ids -> won't attach
    ])
    ontology = _mini_foodon()
    graph_store = InMemoryGraphStore()
    cfg = LayerAConfig(
        min_support=1, max_depth=5, collapse_single_child_chains=False,
        facets=["foods"], umbrella_direct_share_max=0.0,
    )
    full_cfg = _full_config(cfg)
    build_layer_a(chunk_store, graph_store, ontology, config=cfg, full_config=full_cfg)
    attach(chunk_store, graph_store, ontology, full_config=full_cfg)

    report = quality_report(chunk_store, graph_store, ontology,
                            config_hash="t", sample_size=10)
    sample_ids = {c.chunk_id for c in report.chunk_sample}
    assert "c-orphan" not in sample_ids, "unattached chunks must be excluded"


def test_chunk_sample_text_is_capped() -> None:
    long_text = "x" * 500
    chunk_store = InMemoryChunkStore()
    chunk_store.upsert([_chunk("c1", ["TEST:0000008"], text=long_text)])
    ontology = _mini_foodon()
    graph_store = InMemoryGraphStore()
    cfg = LayerAConfig(
        min_support=1, max_depth=5, collapse_single_child_chains=False,
        facets=["foods"], umbrella_direct_share_max=0.0,
    )
    full_cfg = _full_config(cfg)
    build_layer_a(chunk_store, graph_store, ontology, config=cfg, full_config=full_cfg)
    attach(chunk_store, graph_store, ontology, full_config=full_cfg)

    report = quality_report(chunk_store, graph_store, ontology, config_hash="t")
    assert all(len(c.text_snippet) <= 281 for c in report.chunk_sample), (
        "snippets must be capped to ~280 chars"
    )


# ---------------------------------------------------------------- facade


def test_facade_quality_report_returns_report() -> None:
    fs = FoodScholar.in_memory()
    fs.attach_ontology(_mini_foodon())
    # No data — should still produce a report (empty-ish but valid).
    report = fs.quality_report(facet="foods")
    assert isinstance(report, QualityReport)
    assert report.config_hash == fs.config_hash
    assert report.facet == "foods"


def test_facade_quality_report_accepts_custom_canonical_terms() -> None:
    fs = FoodScholar.in_memory()
    fs.attach_ontology(_mini_foodon())
    report = fs.quality_report(canonical_terms=("custom-term-1", "custom-term-2"))
    terms = [v.term for v in report.canonical_vocab_check]
    assert terms == ["custom-term-1", "custom-term-2"]
