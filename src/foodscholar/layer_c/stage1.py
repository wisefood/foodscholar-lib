"""Stage 1 — extractive compression of a theme's chunks.

Single pass when the input is small; map-reduce when it exceeds
`map_reduce_threshold` sentences (group by char budget → summarize each group →
summarize the concatenated group summaries). Nothing is dropped.
"""

from __future__ import annotations

from foodscholar.layer_c.base import BaseSummarizer, split_sentences
from foodscholar.layer_c.models import Stage1Output


def _group_by_chars(chunks: list[str], budget: int) -> list[list[str]]:
    """Greedily pack chunks into groups whose total chars stay near `budget`."""
    groups: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for c in chunks:
        clen = len(c)
        if cur and size + clen > budget:
            groups.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += clen
    if cur:
        groups.append(cur)
    return groups


def run_stage1(
    chunks: list[str],
    summarizer: BaseSummarizer,
    *,
    map_reduce_threshold: int,
    group_char_budget: int,
) -> Stage1Output:
    """Compress `chunks` into one extractive summary with provenance."""
    texts = [c for c in chunks if c and c.strip()]
    n_chunks = len(texts)
    n_chars = sum(len(c) for c in texts)

    if not texts:
        return Stage1Output(text="", n_input_chunks=0, n_input_chars=0,
                            strategy="single", n_groups=1)

    total_sentences = sum(len(split_sentences(c)) for c in texts)
    if total_sentences <= map_reduce_threshold:
        return Stage1Output(
            text=summarizer.summarize(texts),
            n_input_chunks=n_chunks, n_input_chars=n_chars,
            strategy="single", n_groups=1,
        )

    groups = _group_by_chars(texts, group_char_budget)
    group_summaries = [summarizer.summarize(g) for g in groups]
    reduced = summarizer.summarize(group_summaries)
    return Stage1Output(
        text=reduced, n_input_chunks=n_chunks, n_input_chars=n_chars,
        strategy="mapreduce", n_groups=len(groups),
    )
