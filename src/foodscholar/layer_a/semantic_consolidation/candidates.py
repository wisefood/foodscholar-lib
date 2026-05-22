"""Candidate pair generation: cosine similarity + cheap pre-LLM filters.

For N≈230 shelves the all-pairs cosine is a single numpy matmul — faiss is
overkill and isn't a dependency. Vectors are L2-normalized here defensively so
the matmul *is* cosine regardless of whether the embedder normalized.

Three filters run before the (expensive) LLM judge, each recording why a pair
was excluded:

  - subtype collision  — parallel siblings ("turkey bacon" vs "bacon")
  - compound food       — one label contains the other plus real extra words
  - already merged       — the pair is already consolidated (idempotent re-runs)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from foodscholar.layer_a.semantic_consolidation.models import CandidatePair

if TYPE_CHECKING:
    from foodscholar.config import SemanticConsolidationConfig
    from foodscholar.io.graph import Shelf
    from foodscholar.layer_a.semantic_consolidation.models import ShelfEmbedding

# Qualifier tokens that don't make a label a distinct *compound* food — if the
# only extra words are these, the pair is still a merge candidate.
_QUALIFIERS = frozenset(
    {
        "product",
        "products",
        "food",
        "foods",
        "raw",
        "fresh",
        "whole",
        "the",
        "a",
        "an",
        "of",
    }
)


def find_candidates(
    embeddings: list[ShelfEmbedding],
    shelves_by_id: dict[str, Shelf],
    cfg: SemanticConsolidationConfig,
) -> tuple[list[CandidatePair], list[CandidatePair]]:
    """Return ``(candidates_for_judge, filtered_out)``.

    A pair lands in `filtered_out` (with `filtered_reason` set) when a
    pre-LLM filter excludes it; everything else is a candidate for the judge,
    capped per shelf.
    """
    if len(embeddings) < 2:
        return [], []

    matrix = np.asarray([e.embedding for e in embeddings], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix = matrix / norms
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, 0.0)

    candidates: list[CandidatePair] = []
    filtered: list[CandidatePair] = []
    rows, cols = np.where(sims > cfg.cosine_threshold)
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        if i >= j:
            continue  # dedup symmetric pairs (keep i < j)
        a, b = embeddings[i], embeddings[j]
        shelf_a, shelf_b = shelves_by_id[a.shelf_id], shelves_by_id[b.shelf_id]
        reason = (
            _subtype_collision(shelf_a.label, shelf_b.label, cfg)
            or _compound_food(shelf_a.label, shelf_b.label)
            or _already_merged(shelf_a, shelf_b)
        )
        pair = CandidatePair(
            shelf_a=a.shelf_id,
            shelf_b=b.shelf_id,
            cosine_similarity=float(sims[i, j]),
            filtered_reason=reason,
        )
        (filtered if reason else candidates).append(pair)

    candidates.sort(key=lambda p: p.cosine_similarity, reverse=True)
    return _cap_per_shelf(candidates, cfg.max_candidates_per_shelf), filtered


# ----------------------------------------------------------------- filters


def _subtype_collision(
    label_a: str, label_b: str, cfg: SemanticConsolidationConfig
) -> str | None:
    """Exclude if exactly one label starts with a configured subtype prefix.

    'turkey bacon' starts with 'turkey'; 'bacon' does not — they're parallel
    siblings, not duplicates. If *both* (or neither) start with a prefix, the
    rule says nothing.
    """
    a_pref = _starts_with_subtype(label_a, cfg.subtype_patterns)
    b_pref = _starts_with_subtype(label_b, cfg.subtype_patterns)
    if (a_pref is None) == (b_pref is None):
        return None
    prefix = a_pref or b_pref
    return f"subtype_collision:{prefix}"


def _starts_with_subtype(label: str, patterns: list[str]) -> str | None:
    low = label.lower().strip()
    for pat in patterns:
        p = pat.lower().strip()
        if low == p or low.startswith(p + " "):
            return pat
    return None


def _compound_food(label_a: str, label_b: str) -> str | None:
    """Exclude if one label whole-word-contains the other plus real extra words.

    'cream cheese' contains 'cream' and adds 'cheese' (not a qualifier) → a
    distinct compound food, not a duplicate. 'olive oil product' vs 'olive oil'
    differs only by the qualifier 'product' → still a candidate.
    """
    ta = label_a.lower().split()
    tb = label_b.lower().split()
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if not shorter or shorter == longer:
        return None
    if not _is_contiguous_sublist(shorter, longer):
        return None
    extra = [w for w in longer if w not in shorter and w not in _QUALIFIERS]
    if extra:
        return f"compound_food:+{' '.join(extra)}"
    return None


def _is_contiguous_sublist(needle: list[str], haystack: list[str]) -> bool:
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def _already_merged(shelf_a: Shelf, shelf_b: Shelf) -> str | None:
    """Exclude if the pair is already consolidated (foodon_id in see_also)."""
    if shelf_b.foodon_id and shelf_b.foodon_id in shelf_a.see_also:
        return "already_merged"
    if shelf_a.foodon_id and shelf_a.foodon_id in shelf_b.see_also:
        return "already_merged"
    return None


def _cap_per_shelf(
    candidates: list[CandidatePair], max_per_shelf: int
) -> list[CandidatePair]:
    """Keep at most `max_per_shelf` pairs touching any single shelf.

    Candidates are processed highest-similarity first (caller pre-sorts), so
    the strongest pairs survive the cap.
    """
    if max_per_shelf <= 0:
        return candidates
    counts: dict[str, int] = {}
    kept: list[CandidatePair] = []
    for pair in candidates:
        if (
            counts.get(pair.shelf_a, 0) >= max_per_shelf
            or counts.get(pair.shelf_b, 0) >= max_per_shelf
        ):
            continue
        counts[pair.shelf_a] = counts.get(pair.shelf_a, 0) + 1
        counts[pair.shelf_b] = counts.get(pair.shelf_b, 0) + 1
        kept.append(pair)
    return kept
