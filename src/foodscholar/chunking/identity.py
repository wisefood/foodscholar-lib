"""Chunk identity: a version id that tracks the evidence, a public id that persists.

Two ids, because one cannot do both jobs:

- ``chunk_version_id`` is a content hash over the source id, the section path
  and the normalised text. It changes whenever the evidence changes.
- ``chunk_id`` is what every relation, attachment and card cites. On a first
  extraction it equals the version id. On a re-extraction, a chunk whose
  version id already existed keeps the old ``chunk_id``; a chunk whose text
  changed gets a new id with a ``supersedes`` link to the chunk it most
  resembles in the same section. An old citation is never silently redirected
  to merely similar text — it keeps resolving to the snapshot it was made
  against, and the supersedes link says what replaced it.

This module also writes the adjacency links (``prev_chunk_id`` /
``next_chunk_id``) that replace overlap: a chunk's neighbours are one lookup
away instead of being copied into it.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from foodscholar.chunking.assembler import AssembledChunk

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_for_hash(text: str) -> str:
    """Whitespace-insensitive, case-preserving: reflowed text keeps its id."""
    return " ".join(text.split())


def version_id(source_id: str, section_path: Sequence[str], text: str, occurrence: int = 0) -> str:
    """Content hash of one chunk. `occurrence` separates identical repeats in a section."""
    parts = [source_id, "\x1e".join(section_path), normalize_for_hash(text)]
    if occurrence:
        parts.append(str(occurrence))
    digest = hashlib.sha1("\x1f".join(parts).encode(), usedforsecurity=False).hexdigest()
    return digest[:16]


@dataclass(frozen=True)
class PriorChunk:
    """What identity needs to know about a chunk from an earlier run."""

    chunk_id: str
    chunk_version_id: str
    source_id: str = ""
    section_path: tuple[str, ...] = ()
    text: str = ""


@dataclass
class IdentityReport:
    reused: int = 0
    new: int = 0
    superseding: list[tuple[str, str]] = field(default_factory=list)
    """``(new chunk_id, old chunk_id)`` pairs."""
    retired: list[str] = field(default_factory=list)
    """Old chunk ids with neither an unchanged copy nor a successor."""

    def summary(self) -> dict[str, int]:
        return {
            "reused": self.reused,
            "new": self.new,
            "superseding": len(self.superseding),
            "retired": len(self.retired),
        }


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def assign_identity(
    chunks: list[AssembledChunk],
    previous: Iterable[PriorChunk] | None = None,
    *,
    supersede_threshold: float = 0.5,
) -> IdentityReport:
    """Set version ids, public ids, supersedes links and adjacency, in place.

    `chunks` may span several documents; adjacency is linked per source id in
    ordinal order. `supersede_threshold` is the token-Jaccard similarity above
    which a changed chunk is recorded as the successor of an old one in the
    same section.
    """
    report = IdentityReport()
    seen: Counter[tuple[str, tuple[str, ...], str]] = Counter()
    for c in chunks:
        key = (c.source_id, tuple(c.section_path), normalize_for_hash(c.source_text))
        c.chunk_version_id = version_id(c.source_id, c.section_path, c.source_text, seen[key])
        seen[key] += 1

    prior = list(previous or [])
    by_version = {p.chunk_version_id: p for p in prior}
    matched: set[str] = set()
    unmatched: list[AssembledChunk] = []
    for c in chunks:
        old = by_version.get(c.chunk_version_id)
        if old is not None:
            c.chunk_id = old.chunk_id
            matched.add(old.chunk_id)
            report.reused += 1
        else:
            c.chunk_id = c.chunk_version_id
            unmatched.append(c)
            report.new += 1

    candidates = [p for p in prior if p.chunk_id not in matched]
    succeeded: set[str] = set()
    if candidates:
        cand_tokens = {p.chunk_id: _tokens(p.text) for p in candidates}
        for c in unmatched:
            mine = _tokens(c.source_text)
            best, best_score = None, supersede_threshold
            for p in candidates:
                if p.source_id != c.source_id or tuple(p.section_path) != tuple(c.section_path):
                    continue
                score = _jaccard(mine, cand_tokens[p.chunk_id])
                if score >= best_score:
                    best, best_score = p, score
            if best is not None:
                c.supersedes = [best.chunk_id]
                report.superseding.append((c.chunk_id, best.chunk_id))
                succeeded.add(best.chunk_id)
    report.retired = [p.chunk_id for p in candidates if p.chunk_id not in succeeded]

    link_adjacency(chunks)
    return report


def link_adjacency(chunks: list[AssembledChunk]) -> None:
    """Set `prev_chunk_id` / `next_chunk_id` within each document, by ordinal."""
    by_source: dict[str, list[AssembledChunk]] = {}
    for c in chunks:
        by_source.setdefault(c.source_id, []).append(c)
    for group in by_source.values():
        group.sort(key=lambda c: c.ordinal)
        for i, c in enumerate(group):
            c.prev_chunk_id = group[i - 1].chunk_id if i > 0 else None
            c.next_chunk_id = group[i + 1].chunk_id if i + 1 < len(group) else None
