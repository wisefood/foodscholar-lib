"""Layer B audit gates (§10 of the brief)."""

from __future__ import annotations

from foodscholar import FoodScholar
from foodscholar.io.chunk import Chunk
from foodscholar.io.graph import Theme
from foodscholar.layer_b.audit import audit_layer_b


def _chunk(cid: str, *, shelf_ids: list[str] | None = None, theme_ids: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=cid,
        source_doc_id="d",
        source_type="abstract",
        section_type="abstract",
        shelf_ids=shelf_ids or [],
        theme_ids=theme_ids or [],
    )


def _theme(tid: str, shelf_id: str, chunks: list[str], pass_kind: str = "merged") -> Theme:
    return Theme(
        theme_id=tid,
        label=tid,
        shelf_ids=[shelf_id],
        chunk_count=len(chunks),
        discovered_by="leiden",
        discovery_version="v0.1",
        facet="foods",
        discovery_pass=pass_kind,  # type: ignore[arg-type]
    )


def test_audit_passes_when_stores_agree() -> None:
    """Happy path: every chunk's theme_ids matches the THEME_OF edges in
    the graph store. parity = 1.0, no dangling, no empty themes."""
    fs = FoodScholar.in_memory()
    fs.upsert_chunks([_chunk("c1"), _chunk("c2")])
    fs.graph_store.upsert_themes([_theme("t1", "s1", ["c1", "c2"])])
    fs.graph_store.attach_chunks_to_themes_bulk(
        [("c1", "t1", True, 0.9), ("c2", "t1", False, 0.7)]
    )
    fs.chunk_store.bulk_set_theme_ids([("c1", ["t1"]), ("c2", ["t1"])])
    report = audit_layer_b(fs.chunk_store, fs.graph_store)
    assert report.passed is True
    assert report.parity == 1.0
    assert report.dangling_edges == 0
    assert report.empty_themes == 0
    assert report.n_themes == 1
    assert report.by_pass == {"merged": 1}


def test_audit_detects_parity_drift() -> None:
    """Graph store says c2 is in theme t1, but ES theme_ids only has c1.
    parity < 1.0."""
    fs = FoodScholar.in_memory()
    fs.upsert_chunks([_chunk("c1"), _chunk("c2")])
    fs.graph_store.upsert_themes([_theme("t1", "s1", ["c1", "c2"])])
    fs.graph_store.attach_chunks_to_themes_bulk(
        [("c1", "t1", True, 0.9), ("c2", "t1", False, 0.7)]
    )
    # Only c1 denormalized — c2 is in graph but missing from ES.
    fs.chunk_store.bulk_set_theme_ids([("c1", ["t1"])])
    report = audit_layer_b(fs.chunk_store, fs.graph_store)
    assert report.passed is False
    assert report.parity < 1.0


def test_audit_detects_dangling_edges_in_chunk_store() -> None:
    """Chunk has theme_ids pointing at a theme that doesn't exist in the
    graph store — a ghost from a previous run that wasn't cleared."""
    fs = FoodScholar.in_memory()
    fs.upsert_chunks([_chunk("c1", theme_ids=["t-ghost"])])
    # No theme upserted — t-ghost is dangling.
    report = audit_layer_b(fs.chunk_store, fs.graph_store)
    assert report.dangling_edges == 1
    assert report.passed is False


def test_audit_detects_empty_themes() -> None:
    """A theme node with chunk_count > 0 but zero attached chunks at audit
    time — indicates a broken persist run."""
    fs = FoodScholar.in_memory()
    fs.graph_store.upsert_themes([_theme("t1", "s1", ["c1", "c2"])])
    # No attach call → THEME_OF edges absent → audit finds the empty theme.
    report = audit_layer_b(fs.chunk_store, fs.graph_store)
    assert report.empty_themes == 1
    assert report.passed is False


def test_audit_canary_relatedness_zero_themes_records_pass_distribution() -> None:
    """The brief's '≥ 1 theme per pass' canary — when all themes come from
    one pass, the audit records that for tuning feedback. Doesn't fail
    `passed` (it's a WARN, not CRITICAL)."""
    fs = FoodScholar.in_memory()
    fs.upsert_chunks([_chunk("c1"), _chunk("c2")])
    # Both themes come from similarity pass — relatedness contributed 0
    fs.graph_store.upsert_themes(
        [
            _theme("t-sim-1", "s1", ["c1"], pass_kind="similarity"),
            _theme("t-sim-2", "s1", ["c2"], pass_kind="similarity"),
        ]
    )
    fs.graph_store.attach_chunks_to_themes_bulk(
        [("c1", "t-sim-1", True, 1.0), ("c2", "t-sim-2", True, 1.0)]
    )
    fs.chunk_store.bulk_set_theme_ids(
        [("c1", ["t-sim-1"]), ("c2", ["t-sim-2"])]
    )
    report = audit_layer_b(fs.chunk_store, fs.graph_store)
    assert report.passed is True  # CRITICAL invariants still hold
    assert report.by_pass.get("relatedness", 0) == 0
    assert report.by_pass.get("similarity", 0) == 2
    assert report.merged_rate == 0.0


def test_audit_canary_all_merged_records_high_merge_rate() -> None:
    """The inverse canary — if every theme is merged, Pass 2 isn't
    earning compute. Audit records merged_rate for tuning."""
    fs = FoodScholar.in_memory()
    fs.upsert_chunks([_chunk("c1"), _chunk("c2")])
    fs.graph_store.upsert_themes(
        [
            _theme("t-m-1", "s1", ["c1"], pass_kind="merged"),
            _theme("t-m-2", "s1", ["c2"], pass_kind="merged"),
        ]
    )
    fs.graph_store.attach_chunks_to_themes_bulk(
        [("c1", "t-m-1", True, 1.0), ("c2", "t-m-2", True, 1.0)]
    )
    fs.chunk_store.bulk_set_theme_ids(
        [("c1", ["t-m-1"]), ("c2", ["t-m-2"])]
    )
    report = audit_layer_b(fs.chunk_store, fs.graph_store)
    assert report.merged_rate == 1.0


def test_audit_empty_stores_is_vacuously_pass() -> None:
    """No themes, no chunks → no invariants to violate."""
    fs = FoodScholar.in_memory()
    report = audit_layer_b(fs.chunk_store, fs.graph_store)
    assert report.passed is True
    assert report.n_themes == 0


def test_audit_cross_shelf_theme_does_not_fail() -> None:
    """A theme belonging to two shelves (shelf_ids length >= 2) is valid and
    must not fail any CRITICAL gate."""
    fs = FoodScholar.in_memory()
    fs.upsert_chunks([_chunk("c1", theme_ids=["t-multi"])])
    # Build theme directly with two shelf_ids
    multi_shelf_theme = Theme(
        theme_id="t-multi",
        label="t-multi",
        shelf_ids=["s1", "s2"],
        chunk_count=1,
        discovered_by="leiden",
        discovery_version="v0.1",
        facet="foods",
        discovery_pass="global_similarity",
    )
    fs.graph_store.upsert_themes([multi_shelf_theme])
    fs.graph_store.attach_chunks_to_themes_bulk([("c1", "t-multi", True, 0.9)])
    fs.chunk_store.bulk_set_theme_ids([("c1", ["t-multi"])])
    report = audit_layer_b(fs.chunk_store, fs.graph_store)
    assert report.passed is True
    assert report.orphan_themes == 0
    # Both shelves should appear in themed_shelves count
    assert report.n_themed_shelves == 2


def test_audit_orphan_theme_fails() -> None:
    """A theme with shelf_ids=[] is unreachable from any shelf and must fail
    the audit via orphan_themes > 0."""
    fs = FoodScholar.in_memory()
    fs.upsert_chunks([_chunk("c1", theme_ids=["t-orphan"])])
    orphan_theme = Theme(
        theme_id="t-orphan",
        label="t-orphan",
        shelf_ids=[],
        chunk_count=1,
        discovered_by="leiden",
        discovery_version="v0.1",
        facet="foods",
        discovery_pass="similarity",
    )
    fs.graph_store.upsert_themes([orphan_theme])
    fs.graph_store.attach_chunks_to_themes_bulk([("c1", "t-orphan", True, 0.9)])
    fs.chunk_store.bulk_set_theme_ids([("c1", ["t-orphan"])])
    report = audit_layer_b(fs.chunk_store, fs.graph_store)
    assert report.orphan_themes == 1
    assert report.passed is False
    # An orphan theme contributes 0 to themed_shelves
    assert report.n_themed_shelves == 0
