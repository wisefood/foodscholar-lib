"""Stage 1: single-pass below threshold, map-reduce above it."""

from __future__ import annotations

from foodscholar.layer_c.base import BaseSummarizer
from foodscholar.layer_c.stage1 import run_stage1


class _FirstN(BaseSummarizer):
    """Deterministic stub: keep the first n sentences of the concatenation.

    Records how many times it was called so we can assert map vs reduce passes.
    """

    name = "firstn"

    def __init__(self, n: int = 2) -> None:
        self.n = n
        self.calls = 0

    def summarize(self, chunks: list[str]) -> str:
        self.calls += 1
        joined = " ".join(c for c in chunks if c.strip())
        parts = [p.strip() for p in joined.split(".") if p.strip()]
        return ". ".join(parts[: self.n]) + ("." if parts else "")


def test_single_pass_below_threshold() -> None:
    s = _FirstN(n=2)
    chunks = ["One. Two.", "Three."]
    out = run_stage1(chunks, s, map_reduce_threshold=100, group_char_budget=10_000)
    assert out.strategy == "single"
    assert out.n_groups == 1
    assert s.calls == 1
    assert out.n_input_chunks == 2


def test_mapreduce_above_threshold() -> None:
    s = _FirstN(n=2)
    # 6 chunks, each 3 real prose sentences = 18 sentences; threshold 5 forces
    # map-reduce. (Use real words — the splitter now drops non-prose fragments.)
    chunks = [
        f"Alpha grain number {i} is nutritious. "
        f"Beta cereal number {i} has fiber. "
        f"Gamma food number {i} tastes good."
        for i in range(6)
    ]
    out = run_stage1(chunks, s, map_reduce_threshold=5, group_char_budget=20)
    assert out.strategy == "mapreduce"
    assert out.n_groups >= 2
    # one call per group (map) + one reduce call
    assert s.calls == out.n_groups + 1
    assert out.n_input_chunks == 6


def test_empty_input() -> None:
    s = _FirstN(n=2)
    out = run_stage1([], s, map_reduce_threshold=5, group_char_budget=20)
    assert out.text == ""
    assert out.strategy == "single"
    assert out.n_input_chunks == 0
