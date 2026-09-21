"""Ingest-time chunk validation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from foodscholar.config import CorpusValidationConfig
from foodscholar.corpus.csv_reader import ChunkValidator, iter_csv_chunks

HEADER = ["chunk_id", "chunk_text", "type", "chunk_metadata"]


def write_corpus(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "corpus.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return path


def row(chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "chunk_text": text,
        "type": "guide",
        "chunk_metadata": "{'file': 'a.pdf'}",
    }


def test_oversized_chunk_warns_but_still_yields(tmp_path):
    """A corpus that trips the check is still a corpus."""
    path = write_corpus(tmp_path, [row("ok", "short"), row("big", "x " * 5000)])
    validator = ChunkValidator(CorpusValidationConfig())
    chunks = list(iter_csv_chunks(path, validator=validator))
    assert len(chunks) == 2
    assert validator.n_oversized == 1
    assert validator.worst[0] == "big"


def test_raise_mode_aborts(tmp_path):
    path = write_corpus(tmp_path, [row("big", "x " * 5000)])
    validator = ChunkValidator(CorpusValidationConfig(on_violation="raise"))
    with pytest.raises(ValueError, match="over the 512 limit"):
        list(iter_csv_chunks(path, validator=validator))


def test_disabled_checks_nothing(tmp_path):
    path = write_corpus(tmp_path, [row("big", "x " * 5000)])
    validator = ChunkValidator(CorpusValidationConfig(enabled=False))
    assert len(list(iter_csv_chunks(path, validator=validator))) == 1
    assert validator.n_checked == 0


def test_no_validator_is_the_old_behavior(tmp_path):
    path = write_corpus(tmp_path, [row("big", "x " * 5000)])
    assert len(list(iter_csv_chunks(path))) == 1


def test_summary_shape(tmp_path):
    path = write_corpus(tmp_path, [row("ok", "short")])
    validator = ChunkValidator(CorpusValidationConfig())
    list(iter_csv_chunks(path, validator=validator))
    assert validator.summary() == {
        "n_checked": 1,
        "n_oversized": 0,
        "worst_chunk_id": None,
        "worst_tokens": None,
    }


def test_one_validator_accumulates_across_files(tmp_path):
    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    p1 = write_corpus(tmp_path / "d1", [row("a1", "x " * 5000)])
    p2 = write_corpus(tmp_path / "d2", [row("b1", "x " * 5000)])
    validator = ChunkValidator(CorpusValidationConfig())
    list(iter_csv_chunks(p1, validator=validator))
    list(iter_csv_chunks(p2, validator=validator))
    assert validator.n_checked == 2
    assert validator.n_oversized == 2
