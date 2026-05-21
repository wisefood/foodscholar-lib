"""Post-build graph audit.

Runs five independent checks against the chunk store, entity store, and graph
store after `fs.build()` (or any subset of phases). Each check produces an
`AuditCheck` with a structured `metric`/`threshold` pair so failures are
diffable across runs. The full `AuditReport` is JSON-serializable.

The point of audit is to make every cross-store invariant *visible*. Layer A's
projection chose which shelves survive; attach landed chunks on them. If those
two passes disagree, the graph is silently wrong and Layer B will build on top
of the wrongness. Audit catches the disagreement.

Five sections:

  A. Inventory             — node + edge counts. Snapshot, not a check.
  B. Coverage              — attached chunks / total chunks; orphan FOODON
                              entities concentrated on unattached chunks.
  C. Cross-store mention   — `foodon_ids` denorm on chunks (Elastic) vs
                              `MENTIONS` edges to Entity nodes (Neo4j). Two
                              writers, exhaustive comparison.
  D. Attach integrity      — `shelf_ids` denorm on chunks (Elastic) vs
                              `ATTACHED_TO` edges (Neo4j). Per-shelf
                              projection-time chunk_count vs actual edge count.
  E. Structural sanity     — dangling parent refs, cycles, orphan shelves.

What audit DOESN'T do
---------------------
- Judge attachment quality ("is chunk X really about shelf Y?"). That's a hand
  audit per BRIEF §17 and can't be automated honestly.
- Detect NEL drift. `link_blocklist` is the right tool for that.
- Performance regression checks. Separate concern.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from foodscholar.storage.protocols import ChunkStore, GraphStore


# ---------------------------------------------------------------- data shapes


Severity = Literal["critical", "warning", "info"]


class AuditCheck(BaseModel):
    """One audit invariant + its measured value."""

    model_config = ConfigDict(extra="forbid")

    name: str
    section: str  # "A. Inventory" etc — for grouped pretty-print
    severity: Severity = "info"
    passed: bool
    metric: float | int | None = None
    threshold: float | int | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    sample: list[dict[str, Any]] = Field(default_factory=list)  # capped examples

    def summary_line(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        sev = self.severity.upper()
        prefix = f"[{flag} {sev}]"
        body = self.name
        if self.metric is not None and self.threshold is not None:
            body += f" — metric={self.metric}, threshold={self.threshold}"
        elif self.metric is not None:
            body += f" — metric={self.metric}"
        return f"{prefix} {body}"


class AuditReport(BaseModel):
    """All checks plus enough context to diff across runs."""

    model_config = ConfigDict(extra="forbid")

    config_hash: str
    generated_at: datetime
    inventory: dict[str, int] = Field(default_factory=dict)
    checks: list[AuditCheck] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True iff zero critical checks failed. Warnings are allowed."""
        return not any(
            c.severity == "critical" and not c.passed for c in self.checks
        )

    @property
    def critical_failures(self) -> list[AuditCheck]:
        return [c for c in self.checks if c.severity == "critical" and not c.passed]

    @property
    def warnings(self) -> list[AuditCheck]:
        return [c for c in self.checks if c.severity == "warning" and not c.passed]

    def __str__(self) -> str:
        """Human-readable summary, grouped by section."""
        lines = [
            f"Audit report  (config_hash={self.config_hash},"
            f" generated_at={self.generated_at.isoformat()})",
            "",
            "Inventory:",
        ]
        for k, v in sorted(self.inventory.items()):
            lines.append(f"  {k:30s}  {v}")
        lines.append("")

        by_section: dict[str, list[AuditCheck]] = defaultdict(list)
        for c in self.checks:
            by_section[c.section].append(c)
        for section in sorted(by_section):
            lines.append(section)
            for check in by_section[section]:
                lines.append(f"  {check.summary_line()}")
        lines.append("")
        lines.append(
            f"Overall: {'PASS' if self.passed else 'FAIL'}"
            f"  ({len(self.critical_failures)} critical fail,"
            f" {len(self.warnings)} warning)"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------- runner


def audit(
    chunk_store: ChunkStore,
    graph_store: GraphStore,
    *,
    config_hash: str,
) -> AuditReport:
    """Run every check and produce a report. Read-only — no writes anywhere."""
    inventory = _inventory(chunk_store, graph_store)
    checks: list[AuditCheck] = []

    # Section B — coverage. Reads chunks once for the rest of the sections too,
    # so we pull them into memory here; the corpus is small enough (<100k
    # chunks per BRIEF §16) that this is fine.
    chunks = chunk_store.scan()
    checks.extend(_check_coverage(chunks, graph_store))

    # Section C — cross-store mention consistency. Exhaustive: every chunk's
    # FOODON ids in Elastic vs the MENTIONS edges in Neo4j.
    chunk_mentions_neo4j = graph_store.list_chunk_foodon_mentions()
    checks.extend(_check_cross_store_mentions(chunks, chunk_mentions_neo4j))

    # Section D — attach integrity. shelf_ids denorm vs ATTACHED_TO edges.
    chunk_shelf_edges = graph_store.list_chunk_shelf_attachments()
    checks.extend(_check_attach_integrity(chunks, chunk_shelf_edges, graph_store))

    # Section E — structural sanity.
    checks.extend(_check_structural(graph_store))

    return AuditReport(
        config_hash=config_hash,
        generated_at=datetime.now(UTC),
        inventory=inventory,
        checks=checks,
    )


# ---------------------------------------------------------------- A. inventory


def _inventory(chunk_store: ChunkStore, graph_store: GraphStore) -> dict[str, int]:
    chunks = chunk_store.scan()
    shelves = graph_store.list_shelves()
    by_facet: Counter = Counter(s.facet for s in shelves)
    return {
        "chunks_total": len(chunks),
        "shelves_total": len(shelves),
        **{f"shelves_{facet}": n for facet, n in sorted(by_facet.items())},
    }


# ---------------------------------------------------------------- B. coverage


def _check_coverage(chunks: list, graph_store: GraphStore) -> list[AuditCheck]:
    """How many chunks reached at least one shelf? How many had FOODON ids but
    didn't reach any? The latter is the "edible food" pitfall."""
    chunks_total = len(chunks)
    chunks_with_foodon = sum(1 for c in chunks if c.foodon_ids)
    chunks_attached = sum(1 for c in chunks if c.shelf_ids)
    chunks_with_foodon_attached = sum(
        1 for c in chunks if c.foodon_ids and c.shelf_ids
    )
    chunks_with_foodon_unattached = chunks_with_foodon - chunks_with_foodon_attached

    overall_pct = (
        chunks_attached / chunks_total if chunks_total > 0 else 0.0
    )
    foodon_attached_pct = (
        chunks_with_foodon_attached / chunks_with_foodon
        if chunks_with_foodon > 0 else 1.0
    )

    # Orphan FOODON-id pile-up: the top 10 FOODON ids appearing on unattached
    # chunks. High concentration = projection over-pruned that subtree.
    orphan_counter: Counter = Counter()
    for c in chunks:
        if c.foodon_ids and not c.shelf_ids:
            orphan_counter.update(c.foodon_ids)
    top_orphans = [
        {"foodon_id": fid, "n_unattached_chunks": n}
        for fid, n in orphan_counter.most_common(10)
    ]

    checks = [
        AuditCheck(
            name="overall chunk attachment rate",
            section="B. Coverage",
            severity="info",  # low rate is informative, not necessarily wrong
            passed=True,
            metric=round(overall_pct, 4),
            threshold=None,
            details={
                "chunks_total": chunks_total,
                "chunks_attached": chunks_attached,
                "chunks_unattached": chunks_total - chunks_attached,
            },
        ),
        AuditCheck(
            name="FOODON-linked chunks reaching a shelf",
            section="B. Coverage",
            # If chunks have FOODON ids but don't attach, projection is broken.
            # Threshold: 95% — below means the synthetic facet root isn't
            # catching orphans OR the resolver is skipping rows.
            severity="critical",
            passed=foodon_attached_pct >= 0.95,
            metric=round(foodon_attached_pct, 4),
            threshold=0.95,
            details={
                "chunks_with_foodon": chunks_with_foodon,
                "chunks_with_foodon_attached": chunks_with_foodon_attached,
                "chunks_with_foodon_unattached": chunks_with_foodon_unattached,
            },
            sample=top_orphans,
        ),
    ]
    return checks


# ---------------------------------------------------------------- C. cross-store


def _check_cross_store_mentions(
    chunks: list,
    mentions_by_chunk: dict[str, set[str]],
) -> list[AuditCheck]:
    """Per-chunk: do the FOODON ids in Elastic's `foodon_ids` match the FOODON
    `MENTIONS` edges in Neo4j? Exhaustive comparison across the whole corpus.

    Three buckets per chunk:
      - exact match (set equality)
      - mentions ⊂ foodon_ids (denorm has extras the entity graph missed)
      - foodon_ids ⊂ mentions (entity graph has extras the denorm missed)
      - other (disjoint or partial overlap both ways)
    """
    exact = 0
    denorm_has_extras = 0
    mentions_has_extras = 0
    partial = 0
    sample_mismatches: list[dict[str, Any]] = []

    for c in chunks:
        foodon_set = {fid for fid in c.foodon_ids if fid.startswith("FOODON:")}
        mention_set = mentions_by_chunk.get(c.chunk_id, set())
        if foodon_set == mention_set:
            exact += 1
            continue
        if mention_set < foodon_set:
            denorm_has_extras += 1
        elif foodon_set < mention_set:
            mentions_has_extras += 1
        else:
            partial += 1
        if len(sample_mismatches) < 10:
            sample_mismatches.append(
                {
                    "chunk_id": c.chunk_id,
                    "foodon_ids": sorted(foodon_set),
                    "mentions": sorted(mention_set),
                    "only_in_foodon_ids": sorted(foodon_set - mention_set),
                    "only_in_mentions": sorted(mention_set - foodon_set),
                }
            )

    total = len(chunks)
    exact_pct = exact / total if total > 0 else 1.0
    return [
        AuditCheck(
            name="foodon_ids (Elastic) ↔ MENTIONS (Neo4j) exact match",
            section="C. Cross-store consistency",
            # Anything < 95% means the two writers (fs.ingest and
            # fs.build_entities) are producing inconsistent state. Critical
            # because every downstream query is now ambiguous.
            severity="critical",
            passed=exact_pct >= 0.95,
            metric=round(exact_pct, 4),
            threshold=0.95,
            details={
                "exact": exact,
                "denorm_has_extras": denorm_has_extras,
                "mentions_has_extras": mentions_has_extras,
                "partial_overlap": partial,
                "total_chunks": total,
            },
            sample=sample_mismatches,
        )
    ]


# ---------------------------------------------------------------- D. attach


def _check_attach_integrity(
    chunks: list,
    chunk_shelf_edges: dict[str, set[str]],
    graph_store: GraphStore,
) -> list[AuditCheck]:
    """Two checks here:

    1. Per-chunk: does Elastic's `shelf_ids` match the ATTACHED_TO edges in
       Neo4j? The dedupe-bug we fixed would show up here as a mismatch.
    2. Per-shelf: does `shelf.chunk_count` (projection-time) match
       `count(ATTACHED_TO edges)` (actual)? Top divergences.
    """
    # 1. per-chunk parity
    match = 0
    mismatch = 0
    sample_mismatches: list[dict[str, Any]] = []
    for c in chunks:
        elastic = set(c.shelf_ids)
        neo4j = chunk_shelf_edges.get(c.chunk_id, set())
        if elastic == neo4j:
            match += 1
        else:
            mismatch += 1
            if len(sample_mismatches) < 10:
                sample_mismatches.append(
                    {
                        "chunk_id": c.chunk_id,
                        "elastic_shelf_ids": sorted(elastic),
                        "neo4j_attached_to": sorted(neo4j),
                        "only_in_elastic": sorted(elastic - neo4j),
                        "only_in_neo4j": sorted(neo4j - elastic),
                    }
                )

    parity_pct = match / len(chunks) if chunks else 1.0

    # 2. per-shelf support_direct drift.
    #
    # We compare `shelf.support_direct` (projection-time count of chunks that
    # named the shelf's own foodon_id directly) against the count of actual
    # ATTACHED_TO edges with empty lifted_from (i.e. direct attachments).
    # These two numbers SHOULD agree exactly — both measure "chunks linking
    # this shelf's term". A divergence flags either a resolver bug or a
    # projection-then-attach stale state.
    #
    # Notably we do NOT compare `chunk_count` (which is with-descendants in
    # the projection vs nearest-ancestor in attach — by-design different).
    shelves = graph_store.list_shelves()
    shelf_by_id = {s.shelf_id: s for s in shelves}

    direct_edges_per_shelf: Counter = Counter()
    for chunk in chunks:
        for shelf_id in chunk.shelf_ids:
            # A chunk's `shelf_ids` doesn't carry per-edge lifted_from; we
            # treat any chunk attached to a shelf whose foodon_id is in its
            # foodon_ids as "direct" for this check. This mirrors the
            # resolver's own definition of direct.
            shelf = shelf_by_id.get(shelf_id)
            if shelf is None or shelf.foodon_id is None:
                continue
            if shelf.foodon_id in chunk.foodon_ids:
                direct_edges_per_shelf[shelf_id] += 1

    foodon_shelves = [s for s in shelves if s.foodon_id is not None]
    drift_rows: list[dict[str, Any]] = []
    for shelf in foodon_shelves:
        projected = shelf.support_direct
        actual = direct_edges_per_shelf.get(shelf.shelf_id, 0)
        delta = actual - projected
        if abs(delta) > 0:
            drift_rows.append(
                {
                    "shelf_id": shelf.shelf_id,
                    "label": shelf.label,
                    "projected_support_direct": projected,
                    "actual_direct_edges": actual,
                    "delta": delta,
                }
            )
    drift_rows.sort(key=lambda r: -abs(r["delta"]))
    drift_threshold = max(1, int(0.05 * len(foodon_shelves)))

    return [
        AuditCheck(
            name="shelf_ids (Elastic) ↔ ATTACHED_TO (Neo4j) parity",
            section="D. Attach integrity",
            severity="critical",
            passed=parity_pct >= 0.99,
            metric=round(parity_pct, 4),
            threshold=0.99,
            details={"match": match, "mismatch": mismatch, "total": len(chunks)},
            sample=sample_mismatches,
        ),
        AuditCheck(
            name="projection support_direct ↔ actual direct edges",
            section="D. Attach integrity",
            # A small number of drifting shelves is suspect; many means
            # projection and attach disagree on what "direct" means.
            # Threshold is 5% of shelves with a foodon_id.
            severity="warning",
            passed=len(drift_rows) <= drift_threshold,
            metric=len(drift_rows),
            threshold=drift_threshold,
            details={"n_shelves_drifting": len(drift_rows)},
            sample=drift_rows[:10],
        ),
    ]


# ---------------------------------------------------------------- E. structural


def _check_structural(graph_store: GraphStore) -> list[AuditCheck]:
    shelves = graph_store.list_shelves()
    by_id = {s.shelf_id: s for s in shelves}

    # Dangling parent refs.
    dangling = [
        {"shelf_id": s.shelf_id, "label": s.label, "missing_parent": s.parent_shelf_id}
        for s in shelves
        if s.parent_shelf_id is not None and s.parent_shelf_id not in by_id
    ]

    # Cycles via DFS on PARENT_OF.
    cycles: list[list[str]] = []
    visited: set[str] = set()

    def _walk(sid: str, path: list[str]) -> None:
        if sid in path:
            cycle_start = path.index(sid)
            cycles.append([*path[cycle_start:], sid])
            return
        if sid in visited:
            return
        visited.add(sid)
        shelf = by_id.get(sid)
        if shelf is not None and shelf.parent_shelf_id is not None:
            _walk(shelf.parent_shelf_id, [*path, sid])

    for sid in by_id:
        _walk(sid, [])

    # Per-facet roots — should be exactly 1 per facet.
    roots_by_facet: dict[str, list[str]] = defaultdict(list)
    for s in shelves:
        if s.parent_shelf_id is None:
            roots_by_facet[s.facet].append(s.shelf_id)
    multi_root_facets = {f: rs for f, rs in roots_by_facet.items() if len(rs) > 1}

    return [
        AuditCheck(
            name="dangling parent references",
            section="E. Structural sanity",
            severity="critical",
            passed=not dangling,
            metric=len(dangling),
            threshold=0,
            sample=dangling[:10],
        ),
        AuditCheck(
            name="cycles in PARENT_OF",
            section="E. Structural sanity",
            severity="critical",
            passed=not cycles,
            metric=len(cycles),
            threshold=0,
            sample=[{"cycle": c} for c in cycles[:5]],
        ),
        AuditCheck(
            name="single root per facet",
            section="E. Structural sanity",
            severity="warning",
            passed=not multi_root_facets,
            metric=len(multi_root_facets),
            threshold=0,
            details={"facets_with_multiple_roots": multi_root_facets},
        ),
    ]


__all__ = ["AuditCheck", "AuditReport", "audit"]
