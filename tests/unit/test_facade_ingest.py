"""Tests for `fs.ingest` — the single user-facing pipeline entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

from foodscholar import FoodScholar


def _write_corpus_csv(path: Path) -> None:
    path.write_text(
        "chunk_id,chunk_text,type,chunk_metadata\n"
        '"c1","Mediterranean diet rich in olive oil.","abstract","{}"\n'
        '"c2","An apple a day keeps the doctor away.","abstract","{}"\n',
        encoding="utf-8",
    )


def _write_nel_csv(path: Path) -> None:
    path.write_text(
        "chunk_id,chunk_entities_ner,chunk_uri_nel\n"
        '"c1","Mediterranean diet;olive oil","http://purl.obolibrary.org/obo/FOODON_00001234;http://purl.obolibrary.org/obo/FOODON_03309927"\n'
        '"c2","apple","http://purl.obolibrary.org/obo/FOODON_00001141"\n',
        encoding="utf-8",
    )


def _fs(tmp_path: Path, *, snapshot: Path | None = None) -> FoodScholar:
    cfg = {
        "corpus": {"chunks_path": str(tmp_path / "corpus")},
        "storage": {
            "chunk_store": {"backend": "memory"},
            "graph_store": {"backend": "memory"},
        },
    }
    if snapshot is not None:
        cfg["corpus"]["annotated_snapshot_path"] = str(snapshot)
    return FoodScholar.from_config(cfg)


def test_ingest_attaches_pre_computed_nel(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_corpus_csv(corpus / "chunks_foo.csv")
    nel_dir = tmp_path / "nel"
    nel_dir.mkdir()
    _write_nel_csv(nel_dir / "nel_chunks_foo.csv")

    fs = _fs(tmp_path)
    meta = fs.ingest(corpus, nel_dir=nel_dir)
    assert meta is not None
    assert meta.phase == "ingest"
    assert meta.record_count == 2

    c1 = fs.chunk_store.get("c1")
    assert c1 is not None
    assert {m.text for m in c1.mentions} == {"Mediterranean diet", "olive oil"}
    assert set(c1.foodon_ids) == {"FOODON:00001234", "FOODON:03309927"}
    # Ingest does NOT embed — vectors are populated by fs.embed() afterwards.
    assert c1.embedding is None
    assert c1.embedding_model is None
    assert c1.enrichment_version == "annotate-v2"


def test_ingest_handles_chunks_without_matching_nel_row(tmp_path: Path) -> None:
    """Chunks whose chunk_id isn't in the NEL output land empty-annotated, not skipped."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chunks_foo.csv").write_text(
        "chunk_id,chunk_text,type,chunk_metadata\n"
        '"c1","Mediterranean diet rich in olive oil.","abstract","{}"\n'
        '"orphan","Some text with no NEL row.","abstract","{}"\n',
        encoding="utf-8",
    )
    nel_dir = tmp_path / "nel"
    nel_dir.mkdir()
    (nel_dir / "nel_chunks_foo.csv").write_text(
        "chunk_id,chunk_entities_ner,chunk_uri_nel\n"
        '"c1","olive oil","http://purl.obolibrary.org/obo/FOODON_03309927"\n',
        encoding="utf-8",
    )
    fs = _fs(tmp_path)
    fs.ingest(corpus, nel_dir=nel_dir)
    orphan = fs.chunk_store.get("orphan")
    assert orphan is not None
    assert orphan.mentions == []
    assert orphan.foodon_ids == []
    # No embedding at ingest time; the chunk is still stored, just vector-less.
    assert orphan.embedding is None


def test_ingest_writes_snapshot_when_configured(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_corpus_csv(corpus / "chunks_foo.csv")
    nel_dir = tmp_path / "nel"
    nel_dir.mkdir()
    _write_nel_csv(nel_dir / "nel_chunks_foo.csv")
    snapshot = tmp_path / "snap.parquet"

    fs = _fs(tmp_path, snapshot=snapshot)
    fs.ingest(corpus, nel_dir=nel_dir)
    assert snapshot.exists()
    assert snapshot.stat().st_size > 0


def test_ingest_short_circuits_on_existing_snapshot(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_corpus_csv(corpus / "chunks_foo.csv")
    snapshot = tmp_path / "snap.parquet"
    snapshot.write_bytes(b"already-there")

    fs = _fs(tmp_path, snapshot=snapshot)
    # Returns None and does NOT load chunks.
    assert fs.ingest(corpus, nel_dir=tmp_path) is None
    assert fs.chunk_store.scan() == []


def test_ingest_without_nel_dir_calls_load_and_annotate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When nel_dir is omitted, fs.ingest must delegate to load_and_annotate
    (which runs GLiNER+HNSW). We verify by stubbing load_and_annotate.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_corpus_csv(corpus / "chunks_foo.csv")

    fs = _fs(tmp_path)
    called: dict = {}

    def fake_load_and_annotate(self, path, *, snapshot_path=None):  # type: ignore[no-untyped-def]
        called["path"] = Path(path)
        called["snapshot_path"] = snapshot_path
        return None

    monkeypatch.setattr(FoodScholar, "load_and_annotate", fake_load_and_annotate)
    fs.ingest(corpus)
    assert called["path"] == corpus
    assert called["snapshot_path"] is None
