"""Layer B WARN-level quality report + warnings.

Covers `build_quality_report` / `compute_quality_metrics` and each warning kind,
built against the in-memory stores so the metrics are exercised end to end.
"""

from __future__ import annotations

from foodscholar import FoodScholar
from foodscholar.config import LayerBConfig
from foodscholar.io.chunk import Chunk
from foodscholar.io.graph import Shelf, Theme
from foodscholar.layer_b.quality import build_quality_report


def _chunk(cid: str, *, shelf_ids=None, theme_ids=None) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=cid,
        source_doc_id="d",
        source_type="abstract",
        section_type="abstract",
        shelf_ids=shelf_ids or [],
        theme_ids=theme_ids or [],
    )


def _shelf(sid, *, label="lbl", depth=1, parent=None, direct=5, lifted=5, chunks=10, display=None) -> Shelf:
    return Shelf(
        shelf_id=sid,
        label=label,
        display_label=display,
        facet="foods",
        depth=depth,
        parent_shelf_id=parent,
        chunk_count=chunks,
        support_direct=direct,
        support_lifted=lifted,
    )


def _theme(tid, shelf_ids, *, label="theme", chunks=20, pass_kind="merged", signature=None) -> Theme:
    return Theme(
        theme_id=tid,
        label=label,
        shelf_ids=shelf_ids if isinstance(shelf_ids, list) else [shelf_ids],
        chunk_count=chunks,
        discovered_by="leiden",
        discovery_version="v0.2",
        facet="foods",
        discovery_pass=pass_kind,  # type: ignore[arg-type]
        foodon_id_signature=signature or [],
    )


def _fs_with(shelves, themes, chunks, attachments):
    """attachments: list[(chunk_id, shelf_id)]; themes attach by their chunk
    membership via bulk_set_theme_ids on the chunks already carrying theme_ids."""
    fs = FoodScholar.in_memory()
    fs.graph_store.upsert_shelves(shelves)
    fs.graph_store.upsert_themes(themes)
    fs.upsert_chunks(chunks)
    by_shelf: dict[str, list[tuple[str, list[str]]]] = {}
    for cid, sid in attachments:
        by_shelf.setdefault(sid, []).append((cid, []))
    for sid, items in by_shelf.items():
        fs.graph_store.attach_chunks_to_shelf(sid, items)
    return fs


def test_basic_structure_metrics() -> None:
    shelves = [
        _shelf("s1", depth=1, chunks=10),
        _shelf("s2", depth=3, parent="s1", chunks=30),
        _shelf("s3", depth=2, parent="s1", chunks=20),
    ]
    fs = _fs_with(shelves, [], [_chunk("c1", theme_ids=[])], [("c1", "s1")])
    rep = build_quality_report(fs.chunk_store, fs.graph_store, fs.config.layer_b)
    assert rep.n_shelves == 3
    assert rep.max_depth == 3
    assert rep.max_fanout == 2  # s1 has two children
    assert rep.chunks_per_shelf_max == 30
    assert rep.direct_to_lifted_ratio == 1.0  # equal direct/lifted across shelves


def test_coverage_and_sources() -> None:
    shelves = [_shelf("s1")]
    themes = [
        _theme("t-merged", "s1", pass_kind="merged"),
        _theme("t-sim", "s1", pass_kind="global_similarity"),
        _theme("t-rel", "s1", pass_kind="relatedness"),
    ]
    chunks = [
        _chunk("c1", theme_ids=["t-merged"]),
        _chunk("c2", theme_ids=[]),  # orphan
    ]
    fs = _fs_with(shelves, themes, chunks, [("c1", "s1"), ("c2", "s1")])
    rep = build_quality_report(fs.chunk_store, fs.graph_store, fs.config.layer_b)
    assert rep.n_merged == 1
    assert rep.n_similarity_only == 1
    assert rep.n_relatedness_only == 1
    assert rep.theme_coverage == 0.5
    assert rep.n_orphan_chunks == 1


def test_tiny_and_leakage() -> None:
    cfg = LayerBConfig()
    cfg.leiden.min_community_size = 15
    shelves = [_shelf("s1"), _shelf("s2")]
    themes = [
        _theme("t-tiny", "s1", chunks=3),  # below min_community_size
        _theme("t-leak", ["s1", "s2"], chunks=20),  # cross-shelf
    ]
    fs = _fs_with(shelves, themes, [_chunk("c1", theme_ids=["t-leak"])], [("c1", "s1")])
    fs.config.layer_b = cfg
    rep = build_quality_report(fs.chunk_store, fs.graph_store, cfg)
    assert rep.n_tiny_themes == 1
    assert rep.n_cross_shelf_leakage == 1


def test_warn_high_lifted_low_direct() -> None:
    shelves = [_shelf("s1", direct=1, lifted=50)]
    fs = _fs_with(shelves, [], [_chunk("c1")], [("c1", "s1")])
    rep = build_quality_report(fs.chunk_store, fs.graph_store, fs.config.layer_b)
    kinds = {w.kind for w in rep.warnings}
    assert "high_lifted_low_direct" in kinds


def test_warn_shelf_no_themes() -> None:
    cfg = LayerBConfig()
    cfg.min_chunks_per_shelf = 5
    shelves = [_shelf("s1", chunks=50)]
    fs = _fs_with(shelves, [], [_chunk("c1")], [("c1", "s1")])
    fs.config.layer_b = cfg
    rep = build_quality_report(fs.chunk_store, fs.graph_store, cfg)
    assert any(w.kind == "shelf_no_themes" and w.shelf_id == "s1" for w in rep.warnings)


def test_warn_mostly_single_pass() -> None:
    shelves = [_shelf("s1")]
    themes = [
        _theme(f"t{i}", "s1", label=f"t{i}", pass_kind="global_similarity")
        for i in range(5)
    ]
    fs = _fs_with(shelves, themes, [_chunk("c1")], [("c1", "s1")])
    rep = build_quality_report(fs.chunk_store, fs.graph_store, fs.config.layer_b)
    assert any(w.kind == "mostly_single_pass" for w in rep.warnings)


def test_warn_near_duplicate_labels() -> None:
    shelves = [_shelf("s1")]
    themes = [
        _theme("t1", "s1", label="green leafy vegetables"),
        _theme("t2", "s1", label="leafy green vegetables"),  # same token set
    ]
    fs = _fs_with(shelves, themes, [_chunk("c1")], [("c1", "s1")])
    rep = build_quality_report(fs.chunk_store, fs.graph_store, fs.config.layer_b)
    assert any(w.kind == "near_duplicate_labels" for w in rep.warnings)


def test_warn_theme_spans_many_entities() -> None:
    cfg = LayerBConfig()
    cfg.audit.max_entity_span = 3
    shelves = [_shelf("s1")]
    themes = [_theme("t1", "s1", signature=[f"FOODON:{i}" for i in range(6)])]
    fs = _fs_with(shelves, themes, [_chunk("c1")], [("c1", "s1")])
    fs.config.layer_b = cfg
    rep = build_quality_report(fs.chunk_store, fs.graph_store, cfg)
    assert any(w.kind == "theme_spans_many_entities" for w in rep.warnings)


def test_warn_theme_label_equals_parent() -> None:
    shelves = [_shelf("s1", label="Dairy", display="Dairy")]
    themes = [_theme("t1", "s1", label="dairy")]  # case-insensitive match
    fs = _fs_with(shelves, themes, [_chunk("c1")], [("c1", "s1")])
    rep = build_quality_report(fs.chunk_store, fs.graph_store, fs.config.layer_b)
    assert any(w.kind == "theme_label_equals_parent" for w in rep.warnings)


def test_clean_build_has_no_warnings() -> None:
    shelves = [_shelf("s1", direct=10, lifted=5, chunks=100)]
    themes = [
        _theme("t-merged", "s1", label="omega three fatty acids", pass_kind="merged"),
        _theme("t-rel", "s1", label="vitamin d status", pass_kind="relatedness"),
    ]
    chunks = [_chunk("c1", theme_ids=["t-merged"]), _chunk("c2", theme_ids=["t-rel"])]
    fs = _fs_with(shelves, themes, chunks, [("c1", "s1"), ("c2", "s1")])
    rep = build_quality_report(fs.chunk_store, fs.graph_store, fs.config.layer_b)
    assert rep.warnings == []
    assert "quality report" in str(rep).lower()
