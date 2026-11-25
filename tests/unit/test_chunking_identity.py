"""Version ids track evidence; public ids persist; adjacency replaces overlap."""

from __future__ import annotations

from foodscholar.chunking import Chunker, Heading, Paragraph, PriorChunk, load_previous
from foodscholar.chunking.identity import version_id
from tests.unit.chunking_fixtures import COUNTER, config, ir


def _run(texts, previous=None, source_id="DOC", target=9):
    doc = ir([Heading(id="h", text="S", level=1), *(Paragraph(id=f"p{i}", text=t) for i, t in enumerate(texts))], source_id=source_id)
    return Chunker(config(min_tokens=0, target_tokens=target, max_tokens=40, overlap_max_tokens=8), token_counter=COUNTER).assemble(doc, previous=previous)


def _prior(result):
    return [PriorChunk(c.chunk_id, c.chunk_version_id, c.source_id, c.section_path, c.source_text) for c in result.chunks]


TEXTS = [
    "Eat five portions of vegetables and fruit every day.",
    "Choose wholemeal bread and brown rice when you can.",
    "Limit salt to six grams a day for adults.",
]


def test_version_id_ignores_reflowed_whitespace_but_not_words():
    assert version_id("d", ("s",), "a  b\nc") == version_id("d", ("s",), "a b c")
    assert version_id("d", ("s",), "a b c") != version_id("d", ("s",), "a b d")
    assert version_id("d", ("s",), "x") != version_id("e", ("s",), "x")
    assert version_id("d", ("s",), "x") != version_id("d", ("t",), "x")


def test_first_run_ids_equal_version_ids():
    res = _run(TEXTS)
    assert all(c.chunk_id == c.chunk_version_id for c in res.chunks)
    assert res.identity.summary() == {"reused": 0, "new": 3, "superseding": 0, "retired": 0}


def test_rerun_on_unchanged_text_reuses_every_id():
    first = _run(TEXTS)
    again = _run(TEXTS, previous=_prior(first))
    assert [c.chunk_id for c in again.chunks] == [c.chunk_id for c in first.chunks]
    assert again.identity.reused == 3 and again.identity.new == 0


def test_changed_text_gets_a_new_id_that_supersedes_the_old_one():
    first = _run(TEXTS)
    edited = [*TEXTS[:2], "Limit salt to five grams a day for adults."]
    again = _run(edited, previous=_prior(first))
    old, new = first.chunks[2], again.chunks[2]
    assert new.chunk_id != old.chunk_id
    assert new.supersedes == [old.chunk_id]
    assert again.identity.superseding == [(new.chunk_id, old.chunk_id)]
    assert again.identity.retired == []


def test_removed_text_is_reported_retired():
    first = _run(TEXTS)
    again = _run(TEXTS[:2], previous=_prior(first))
    assert again.identity.retired == [first.chunks[2].chunk_id]


def test_identical_repeats_in_a_section_get_distinct_ids():
    res = _run(["Wash your hands before cooking.", "Wash your hands before cooking."], target=5)
    assert len(res.chunks) == 2
    assert len({c.chunk_id for c in res.chunks}) == 2


def test_adjacency_links_follow_reading_order_within_a_document():
    res = _run(TEXTS)
    ids = [c.chunk_id for c in res.chunks]
    assert [c.prev_chunk_id for c in res.chunks] == [None, ids[0], ids[1]]
    assert [c.next_chunk_id for c in res.chunks] == [ids[1], ids[2], None]


def test_previous_run_round_trips_through_chunks_jsonl(tmp_path):
    first = _run(TEXTS)
    first.write(tmp_path)
    prior = load_previous(tmp_path / "chunks.jsonl")
    again = _run(TEXTS, previous=prior)
    assert again.identity.reused == 3
