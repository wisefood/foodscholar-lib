"""The shared sliding-window chunker.

All three corpus producers — guides, textbooks and abstracts — use the *same*
512/64 window. They differ only in how they produce the fine-grained units the
window slides over:

- guides / textbooks: Docling ``HybridChunker`` at ``fine_max_tokens=80``
- abstracts: NLTK sentences

So this module holds the algorithm once, source-agnostic and dependency-free,
and :mod:`foodscholar.corpus.chunker` supplies the producers. That split is
what makes the window unit-testable with a stub token counter and no models.

Ported from ``chunking_pipeline_guides_overlap.ipynb`` /
``chunking_pipeline_textbooks_overlap.ipynb`` /
``chunking_abstracts_split_check.ipynb``. The semantics below are deliberate
and load-bearing — see the notes on each.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import groupby

# A chunk that is empty or made only of punctuation carries no signal. The
# notebooks drop these after windowing; so do we.
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", flags=re.UNICODE)

DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP = 64
DEFAULT_FINE_MAX_TOKENS = 80


@dataclass(frozen=True)
class FineUnit:
    """One fine-grained unit the window slides over (a sub-chunk or sentence)."""

    text: str
    group_key: str = ""
    """Units are grouped by this before windowing, and a window never spans two
    groups. For PDFs it is the top-level heading, which is what stops overlap
    bleeding across chapters; for plain text it is a single constant group."""
    page_numbers: tuple[int, ...] = ()


@dataclass
class MergedChunk:
    """One emitted chunk: several `FineUnit`s merged."""

    text: str
    group_key: str = ""
    page_numbers: tuple[int, ...] = ()
    token_count: int = 0
    num_sub_chunks: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def first_page(self) -> int | None:
        """The chunk's first page.

        This is what the notebooks write to ``chunk_metadata["page_number"]``.
        It is *not* the chunk's span — see `page_numbers` and the warning on
        `foodscholar.io.chunk.ChunkProvenance`.
        """
        return self.page_numbers[0] if self.page_numbers else None


def has_meaningful_text(text: object) -> bool:
    """True when `text` is non-empty and not purely punctuation."""
    s = str(text or "").strip()
    return bool(s) and not _PUNCT_ONLY_RE.fullmatch(s)


def _merge(batch: Sequence[FineUnit], count_tokens: Callable[[str], int]) -> MergedChunk:
    merged_text = " ".join(u.text.strip() for u in batch)
    pages = sorted({p for u in batch for p in u.page_numbers})
    return MergedChunk(
        text=merged_text,
        group_key=batch[0].group_key if batch else "",
        page_numbers=tuple(pages),
        token_count=count_tokens(merged_text),
        num_sub_chunks=len(batch),
    )


def build_overlapping_chunks(
    fine_units: Sequence[FineUnit],
    *,
    count_tokens: Callable[[str], int],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[MergedChunk]:
    """Merge `fine_units` into ~`max_tokens` chunks with >= `overlap` tokens of overlap.

    Semantics, each of which changes the output if altered:

    1. **Grouping uses `itertools.groupby`**, which groups only *consecutive*
       units sharing a `group_key`. A dict-based grouping would merge
       non-adjacent occurrences of the same heading and produce different
       chunks. This is faithful to the notebooks.
    2. **If everything remaining in a group fits**, it is emitted as one chunk
       and the group ends — no redundant tail-only chunk.
    3. **`overlap` is a minimum tail, not a target.** `start` advances to the
       *furthest* unit that still leaves at least `overlap` tokens behind it,
       so the realized overlap is >= `overlap` and often more.
    4. **A single unit larger than `max_tokens`** is emitted alone; `start`
       advances by one so the loop always makes progress.

    `count_tokens` is injected so the window needs no tokenizer of its own —
    tests pass a word counter, production passes the bge-large tokenizer.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    result: list[MergedChunk] = []
    for _group_key, group_iter in groupby(fine_units, key=lambda u: u.group_key):
        group = list(group_iter)
        n = len(group)
        if n == 0:
            continue

        token_counts = [count_tokens(u.text) for u in group]
        cumsum = [0] * (n + 1)
        for i in range(n):
            cumsum[i + 1] = cumsum[i] + token_counts[i]

        start = 0
        while start < n:
            # (2) everything remaining fits -> one chunk, done with this group
            if cumsum[n] - cumsum[start] <= max_tokens:
                result.append(_merge(group[start:n], count_tokens))
                break

            # expand `end` as far as the budget allows
            end = start
            while end < n and (cumsum[end + 1] - cumsum[start]) <= max_tokens:
                end += 1

            if end == start:
                # (4) one oversized unit
                batch = group[start : start + 1]
                start += 1
            else:
                batch = group[start:end]
                # (3) advance to the furthest k still leaving >= overlap behind
                new_start = start
                for k in range(start + 1, end + 1):
                    if cumsum[end] - cumsum[k] >= overlap:
                        new_start = k
                    else:
                        break
                start = max(new_start, start + 1)

            result.append(_merge(batch, count_tokens))

    return result
