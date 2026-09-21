"""The shared sliding window — no models, no tokenizer, no network.

The window is the whole chunking algorithm; these assertions pin the four
semantics that change the output if a port drifts.
"""

from __future__ import annotations

import pytest

from foodscholar.corpus.window import (
    FineUnit,
    build_overlapping_chunks,
    has_meaningful_text,
)


def wc(text: str) -> int:
    """Word count stands in for a tokenizer."""
    return len(text.split())


def unit(n_tokens: int, group: str = "", pages: tuple[int, ...] = ()) -> FineUnit:
    return FineUnit(text=" ".join(["w"] * n_tokens), group_key=group, page_numbers=pages)


def test_everything_fitting_emits_one_chunk_and_stops():
    chunks = build_overlapping_chunks(
        [unit(10), unit(10), unit(10)], count_tokens=wc, max_tokens=100, overlap=5
    )
    assert len(chunks) == 1
    assert chunks[0].num_sub_chunks == 3
    assert chunks[0].token_count == 30


def test_groupby_only_groups_consecutive_units():
    """A dict-grouping port would merge the two 'A' runs and emit fewer chunks."""
    chunks = build_overlapping_chunks(
        [unit(5, "A"), unit(5, "A"), unit(5, "B"), unit(5, "A")],
        count_tokens=wc,
        max_tokens=100,
        overlap=10,
    )
    assert [c.group_key for c in chunks] == ["A", "B", "A"]


def test_overlap_is_a_minimum_tail_not_a_target():
    """Consecutive windows share units, and the shared tail clears `overlap`."""
    chunks = build_overlapping_chunks(
        [unit(40) for _ in range(6)], count_tokens=wc, max_tokens=100, overlap=40
    )
    assert len(chunks) > 1
    # Each window holds 2 units (80 tokens) and advances by 1 -> 40 shared.
    assert all(c.num_sub_chunks == 2 for c in chunks)
    assert all(c.token_count == 80 for c in chunks)


def test_single_oversized_unit_is_emitted_alone():
    chunks = build_overlapping_chunks(
        [unit(500), unit(10)], count_tokens=wc, max_tokens=100, overlap=10
    )
    assert [c.token_count for c in chunks] == [500, 10]
    assert chunks[0].num_sub_chunks == 1


def test_always_makes_progress_on_pathological_input():
    """Many equal units at exactly the budget must terminate."""
    chunks = build_overlapping_chunks(
        [unit(100) for _ in range(20)], count_tokens=wc, max_tokens=100, overlap=99
    )
    assert len(chunks) == 20


def test_pages_union_and_first_page():
    chunks = build_overlapping_chunks(
        [unit(10, pages=(4, 5)), unit(10, pages=(5, 6))],
        count_tokens=wc,
        max_tokens=100,
        overlap=5,
    )
    assert chunks[0].page_numbers == (4, 5, 6)
    assert chunks[0].first_page == 4


def test_empty_input_is_empty_output():
    assert build_overlapping_chunks([], count_tokens=wc) == []


@pytest.mark.parametrize(
    ("max_tokens", "overlap"),
    [(0, 10), (100, -1), (100, 100), (100, 200)],
)
def test_invalid_parameters_raise(max_tokens, overlap):
    with pytest.raises(ValueError):
        build_overlapping_chunks([unit(5)], count_tokens=wc, max_tokens=max_tokens, overlap=overlap)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("hello", True), ("", False), ("   ", False), ("...", False), ("-- !!", False), ("a.", True)],
)
def test_has_meaningful_text(text, expected):
    assert has_meaningful_text(text) is expected
