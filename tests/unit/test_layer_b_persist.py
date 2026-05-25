"""Theme attachment + theme_ids denorm contract for Layer B.

These tests pin the three new storage protocol methods:
  - GraphStore.attach_chunks_to_themes_bulk — writes (:Chunk)-[:THEME_OF
    {primary, weight}]->(:Theme) edges
  - GraphStore.clear_themes — wipes (:Theme) + THEME_OF + HAS_THEME edges
  - ChunkStore.bulk_set_theme_ids — sets theme_ids without touching shelf_ids
    (avoids the read-then-overwrite race that bulk_update_attachments has)
"""

from __future__ import annotations

from foodscholar import FoodScholar
from foodscholar.io.chunk import Chunk
from foodscholar.io.graph import Theme


def _chunk(cid: str, *, shelf_ids: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=cid,
        source_doc_id="d",
        source_type="abstract",
        section_type="abstract",
        shelf_ids=shelf_ids or [],
    )


def _theme(tid: str, shelf_id: str, chunks: list[str]) -> Theme:
    return Theme(
        theme_id=tid,
        label=tid,
        shelf_ids=[shelf_id],
        chunk_count=len(chunks),
        discovered_by="leiden",
        discovery_version="v0.1",
        facet="foods",
        discovery_pass="similarity",
    )


# ----------------------------------------------------------------------------
# attach_chunks_to_themes_bulk
# ----------------------------------------------------------------------------


def test_attach_chunks_to_themes_bulk_writes_primary_and_weight() -> None:
    fs = FoodScholar.in_memory()
    fs.upsert_chunks([_chunk("c1"), _chunk("c2")])
    fs.graph_store.upsert_themes([_theme("t1", "s1", ["c1", "c2"])])
    fs.graph_store.attach_chunks_to_themes_bulk(
        [
            ("c1", "t1", True, 0.92),
            ("c2", "t1", False, 0.55),
        ]
    )
    assert set(fs.graph_store.get_chunks_for_theme("t1")) == {"c1", "c2"}


def test_attach_chunks_to_themes_bulk_empty_is_noop() -> None:
    fs = FoodScholar.in_memory()
    fs.graph_store.attach_chunks_to_themes_bulk([])
    # No crash, no leftover state.


def test_attach_chunks_to_themes_bulk_idempotent() -> None:
    fs = FoodScholar.in_memory()
    fs.upsert_chunks([_chunk("c1")])
    fs.graph_store.upsert_themes([_theme("t1", "s1", ["c1"])])
    fs.graph_store.attach_chunks_to_themes_bulk([("c1", "t1", True, 0.9)])
    fs.graph_store.attach_chunks_to_themes_bulk([("c1", "t1", True, 0.9)])
    assert fs.graph_store.get_chunks_for_theme("t1") == ["c1"]


# ----------------------------------------------------------------------------
# clear_themes
# ----------------------------------------------------------------------------


def test_clear_themes_drops_theme_nodes_and_edges() -> None:
    fs = FoodScholar.in_memory()
    fs.upsert_chunks([_chunk("c1")])
    fs.graph_store.upsert_themes([_theme("t1", "s1", ["c1"])])
    fs.graph_store.attach_chunks_to_themes_bulk([("c1", "t1", True, 1.0)])
    assert fs.graph_store.list_themes() != []

    fs.graph_store.clear_themes()
    assert fs.graph_store.list_themes() == []
    # Theme is gone; the chunk node + shelf attachments survive.
    assert fs.chunk_store.get("c1") is not None


def test_clear_themes_on_empty_store_is_noop() -> None:
    fs = FoodScholar.in_memory()
    fs.graph_store.clear_themes()  # no crash on a never-themed store


# ----------------------------------------------------------------------------
# bulk_set_theme_ids (chunk-side denorm)
# ----------------------------------------------------------------------------


def test_bulk_set_theme_ids_overwrites_theme_ids_only() -> None:
    """The critical contract — shelf_ids on the chunk MUST be preserved.
    This is the whole reason bulk_set_theme_ids exists vs reusing
    bulk_update_attachments."""
    fs = FoodScholar.in_memory()
    fs.upsert_chunks([_chunk("c1", shelf_ids=["s-foods", "s-dairy"])])
    fs.chunk_store.bulk_set_theme_ids([("c1", ["t1", "t2"])])
    c1 = fs.chunk_store.get("c1")
    assert c1 is not None
    assert c1.theme_ids == ["t1", "t2"]
    # Shelf ids untouched — the load-bearing invariant.
    assert c1.shelf_ids == ["s-foods", "s-dairy"]


def test_bulk_set_theme_ids_empty_list_clears_theme_ids() -> None:
    """Passing an empty theme_ids list is the explicit 'remove all themes
    from this chunk' signal — needed by clear-and-rebuild Layer B runs."""
    fs = FoodScholar.in_memory()
    fs.upsert_chunks([_chunk("c1", shelf_ids=["s-foods"])])
    fs.chunk_store.bulk_set_theme_ids([("c1", ["t1"])])
    fs.chunk_store.bulk_set_theme_ids([("c1", [])])
    c1 = fs.chunk_store.get("c1")
    assert c1 is not None
    assert c1.theme_ids == []
    assert c1.shelf_ids == ["s-foods"]


def test_bulk_set_theme_ids_empty_items_is_noop() -> None:
    fs = FoodScholar.in_memory()
    fs.chunk_store.bulk_set_theme_ids([])


def test_bulk_set_theme_ids_missing_chunk_silently_skipped() -> None:
    """A theme_id targeting a chunk that no longer exists shouldn't crash —
    in production this might happen if a chunk was deleted between the
    builder reading shelf attachments and persist running."""
    fs = FoodScholar.in_memory()
    fs.chunk_store.bulk_set_theme_ids([("c-missing", ["t1"])])
