"""Linker coverage evaluation per BRIEF §17.

Loads a JSONL gold set (one record per line, fields: `text`, `expected_id`,
optional `tier`) and reports linker coverage and accuracy. Records with
`expected_id == null` are negative cases — the linker should return None.

Returned `LinkerEvalReport` exposes both the headline numbers and the
per-tier breakdown so it's easy to see whether the dense fallback is
pulling its weight.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foodscholar.storage.protocols import Linker


@dataclass(frozen=True)
class GoldRecord:
    text: str
    expected_id: str | None
    tier: str = ""


@dataclass
class LinkerEvalReport:
    n_total: int = 0
    n_correct: int = 0
    n_linked: int = 0  # linker returned an id (regardless of correctness)
    by_tier_correct: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_tier_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    misses: list[tuple[str, str | None, str | None]] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of gold-positive cases the linker resolved (correct or not)."""
        positives = sum(self.by_tier_total[t] for t in self.by_tier_total if t != "miss")
        if positives == 0:
            return 0.0
        linked_positives = self.n_linked - self.by_tier_correct.get("miss-as-link", 0)
        return linked_positives / positives

    @property
    def accuracy(self) -> float:
        """Fraction of cases (positive + negative) where the linker returned the expected id (or None)."""
        if self.n_total == 0:
            return 0.0
        return self.n_correct / self.n_total

    def summary(self) -> dict[str, float | int]:
        return {
            "total": self.n_total,
            "correct": self.n_correct,
            "linked": self.n_linked,
            "coverage": round(self.coverage, 3),
            "accuracy": round(self.accuracy, 3),
            "by_tier": {
                t: f"{self.by_tier_correct[t]}/{self.by_tier_total[t]}"
                for t in sorted(self.by_tier_total)
            },
        }


def load_gold(path: str | Path) -> list[GoldRecord]:
    p = Path(path)
    out: list[GoldRecord] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.append(
            GoldRecord(
                text=row["text"],
                expected_id=row.get("expected_id"),
                tier=row.get("tier", ""),
            )
        )
    return out


def evaluate(linker: Linker, gold: list[GoldRecord]) -> LinkerEvalReport:
    from foodscholar.io.chunk import Mention

    report = LinkerEvalReport()
    for rec in gold:
        report.n_total += 1
        report.by_tier_total[rec.tier] += 1

        mention = Mention(
            text=rec.text,
            start=0,
            end=len(rec.text),
            score=1.0,
            ner_model_version="eval",
        )
        link = linker.link(mention)

        got_id = link.ontology_id if link else None
        if got_id is not None:
            report.n_linked += 1

        if got_id == rec.expected_id:
            report.n_correct += 1
            report.by_tier_correct[rec.tier] += 1
        else:
            report.misses.append((rec.text, rec.expected_id, got_id))
    return report
