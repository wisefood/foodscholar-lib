"""Tests for the pre-computed NER/NEL CSV loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from foodscholar.corpus.nel_loader import (
    iter_nel_rows,
    load_nel_dir,
    shorten_obo_uri,
)


def _write_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    lines = ["chunk_id,chunk_entities_ner,chunk_uri_nel"]
    for cid, surf, uris in rows:
        lines.append(f'"{cid}","{surf}","{uris}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_shorten_obo_uri_foodon() -> None:
    assert shorten_obo_uri("http://purl.obolibrary.org/obo/FOODON_00005147") == "FOODON:00005147"


def test_shorten_obo_uri_chebi_https() -> None:
    assert shorten_obo_uri("https://purl.obolibrary.org/obo/CHEBI_16526") == "CHEBI:16526"


def test_shorten_obo_uri_unknown_passthrough() -> None:
    # Already short → unchanged. Empty → empty.
    assert shorten_obo_uri("FOODON:03309927") == "FOODON:03309927"
    assert shorten_obo_uri("") == ""
    assert shorten_obo_uri("   ") == ""


def test_iter_nel_rows_basic(tmp_path: Path) -> None:
    p = tmp_path / "nel_demo.csv"
    _write_csv(
        p,
        [
            (
                "c1",
                "food;edible food",
                "http://purl.obolibrary.org/obo/FOODON_00005147;http://purl.obolibrary.org/obo/FOODON_00005147",
            ),
            (
                "c2",
                "China;food",
                "http://purl.obolibrary.org/obo/GAZ_00002845;http://purl.obolibrary.org/obo/FOODON_00005147",
            ),
        ],
    )
    rows = list(iter_nel_rows(p))
    assert [r[0] for r in rows] == ["c1", "c2"]
    _, mentions, links, foodon_ids = rows[0]
    assert {m.text for m in mentions} == {"food", "edible food"}
    assert [ln.ontology_id for ln in links] == ["FOODON:00005147", "FOODON:00005147"]
    assert foodon_ids == ["FOODON:00005147"]  # deduplicated


def test_iter_nel_rows_handles_nil_entries(tmp_path: Path) -> None:
    """Empty URI slots in the prototype output → no link, but mention kept."""
    p = tmp_path / "nel_nil.csv"
    _write_csv(
        p,
        [
            (
                "c1",
                "food;each year;edible food",
                "http://purl.obolibrary.org/obo/FOODON_00005147;;http://purl.obolibrary.org/obo/FOODON_00005147",
            ),
        ],
    )
    [(_cid, mentions, links, foodon_ids)] = list(iter_nel_rows(p))
    assert len(mentions) == 3  # all surfaces preserved as mentions
    assert len(links) == 2  # the NIL slot produced no EntityLink
    assert foodon_ids == ["FOODON:00005147"]


def test_iter_nel_rows_skips_chunks_with_no_id(tmp_path: Path) -> None:
    p = tmp_path / "blank_id.csv"
    _write_csv(p, [("", "food", "http://purl.obolibrary.org/obo/FOODON_00005147")])
    assert list(iter_nel_rows(p)) == []


def test_iter_nel_rows_rejects_missing_columns(tmp_path: Path) -> None:
    p = tmp_path / "bad_cols.csv"
    p.write_text("chunk_id,chunk_entities_ner\nc1,food\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        list(iter_nel_rows(p))


def test_load_nel_dir_merges_across_files(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "nel_a.csv",
        [("c1", "food", "http://purl.obolibrary.org/obo/FOODON_00005147")],
    )
    _write_csv(
        tmp_path / "nel_b.csv",
        [("c2", "olive oil", "http://purl.obolibrary.org/obo/FOODON_03309927")],
    )
    out = load_nel_dir(tmp_path)
    assert set(out.keys()) == {"c1", "c2"}
    assert out["c2"][2] == ["FOODON:03309927"]


def test_load_nel_dir_raises_when_path_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_nel_dir(tmp_path / "does_not_exist")
