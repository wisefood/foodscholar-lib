"""Domain-expert quality report for the Layer A graph.

`fs.audit()` checks invariants ("is the graph correctly built?"). This module
asks a different question — "is the graph *good*?" — and packages the answer
for a nutritionist or food-science researcher to skim in ~30 minutes.

The output is intentionally NOT a single score. Quality is multidimensional
and collapsing it to one number hides the failures that matter. Instead the
report has five sections, each surfacing a different concern:

  1. Top shelves at a glance      — the navigable face of the graph
  2. Hierarchy walkthrough        — parent chains + sample descendants
  3. Suspicious shelves           — heuristic flags for likely-broken shelves
  4. Canonical vocabulary check   — does the graph contain the foods/nutrients
                                    /conditions an expert would expect?
  5. Random chunk sample          — concrete chunks + their shelves, formatted
                                    for hand audit (§17 sanity gate)

What this module deliberately does NOT do
-----------------------------------------
- Compute a composite "quality score". Multidimensional concerns don't reduce
  to one number without hiding failures.
- Run semantic similarity between chunks and shelves. Cosines aren't truth.
- Tie any section to pass/fail. Quality is a conversation with the expert,
  not an invariant.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from foodscholar.io.graph import Facet, Shelf
    from foodscholar.ontology import FoodOnAPI
    from foodscholar.storage.protocols import ChunkStore, GraphStore


# ---------------------------------------------------------------- vocabulary


# Canonical terms a nutritionist would expect to find as shelves in any
# nutrition-corpus graph. Mix of foods, nutrients, dietary patterns, and
# conditions. Override via `fs.quality_report(canonical_terms=...)`. Keep
# short (~30 items) — this is a probe, not a curriculum.
_DEFAULT_CANONICAL_TERMS: tuple[str, ...] = (
    # Foods
    "olive oil", "kale", "broccoli", "quinoa", "salmon", "walnut",
    "yogurt", "whole grain", "legume", "blueberry",
    # Nutrients
    "omega 3", "vitamin D", "fiber", "iron", "calcium", "vitamin B12",
    "antioxidant", "polyphenol",
    # Dietary patterns
    "mediterranean diet", "DASH diet", "vegetarian diet", "ketogenic diet",
    # Conditions / health concepts
    "cardiovascular disease", "diabetes", "obesity", "hypertension",
    "inflammation", "insulin resistance", "metabolic syndrome",
)


# ---------------------------------------------------------------- data shapes


class ShelfSnapshot(BaseModel):
    """Public per-shelf row a domain expert reads. No internal jargon."""

    model_config = ConfigDict(extra="forbid")

    label: str
    shelf_id: str
    depth: int
    facet: str
    chunk_count: int
    support_direct: int
    support_lifted: int
    direct_share: float  # support_direct / chunk_count, or 0 if no chunks
    sample_chunks: list[str] = Field(default_factory=list)  # up to 3 snippets


class SuspiciousShelf(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    shelf_id: str
    reason: str  # human-readable: "label contains EFSA code", etc.
    metric: str  # quick numeric description: "lifted_share=0.98"


class HierarchyEntry(BaseModel):
    """One root shelf with its parent chain + sample descendants."""

    model_config = ConfigDict(extra="forbid")

    shelf: ShelfSnapshot
    ancestor_chain: list[str] = Field(default_factory=list)  # root-first labels
    sample_descendants: list[str] = Field(default_factory=list)


class VocabHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: str
    status: Literal["found_as_shelf", "found_in_ontology_only", "not_found"]
    shelf_label: str | None = None
    shelf_id: str | None = None
    chunk_count: int = 0
    ontology_id: str | None = None  # populated when the term resolves in FoodOn


class ChunkAuditEntry(BaseModel):
    """One random chunk for hand review — the §17 sanity gate row."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text_snippet: str  # capped ~280 chars
    source_type: str
    attached_shelves: list[str]  # labels not ids — expert reads these


class QualityReport(BaseModel):
    """Bundle of all five sections + provenance for diffing across runs."""

    model_config = ConfigDict(extra="forbid")

    config_hash: str
    generated_at: datetime
    facet: str
    n_shelves: int

    top_shelves: list[ShelfSnapshot] = Field(default_factory=list)
    hierarchy_walkthrough: list[HierarchyEntry] = Field(default_factory=list)
    suspicious_shelves: list[SuspiciousShelf] = Field(default_factory=list)
    canonical_vocab_check: list[VocabHit] = Field(default_factory=list)
    chunk_sample: list[ChunkAuditEntry] = Field(default_factory=list)

    def __str__(self) -> str:
        """Markdown-style summary for in-notebook reading."""
        lines: list[str] = [
            f"# Layer A quality report — facet={self.facet!r}",
            "",
            f"_config_hash {self.config_hash}, generated {self.generated_at.isoformat()}_",
            f"_{self.n_shelves} shelves in this facet_",
            "",
        ]

        lines.append("## 1. Top shelves at a glance")
        lines.append("")
        for s in self.top_shelves:
            lines.append(
                f"### {s.label!r} (depth {s.depth}, {s.chunk_count} chunks, "
                f"direct {s.support_direct} / lifted {s.support_lifted})"
            )
            for snippet in s.sample_chunks:
                lines.append(f"  - {snippet}")
            lines.append("")

        lines.append("## 2. Hierarchy walkthrough")
        lines.append("")
        for h in self.hierarchy_walkthrough:
            chain = " > ".join(h.ancestor_chain) or "(root)"
            lines.append(f"### {h.shelf.label}")
            lines.append(f"  parents: {chain}")
            if h.sample_descendants:
                lines.append(
                    "  descendants: " + ", ".join(h.sample_descendants[:8])
                )
            lines.append("")

        lines.append("## 3. Suspicious shelves")
        lines.append("")
        if not self.suspicious_shelves:
            lines.append("_no suspicious shelves flagged_")
            lines.append("")
        for sus in self.suspicious_shelves:
            lines.append(f"- **{sus.label!r}** — {sus.reason} ({sus.metric})")
        lines.append("")

        lines.append("## 4. Canonical vocabulary check")
        lines.append("")
        for v in self.canonical_vocab_check:
            if v.status == "found_as_shelf":
                lines.append(
                    f"- ✓ {v.term!r} → shelf {v.shelf_label!r} "
                    f"({v.chunk_count} chunks)"
                )
            elif v.status == "found_in_ontology_only":
                lines.append(
                    f"- ~ {v.term!r} → exists in FoodOn as {v.ontology_id} "
                    f"but no surviving shelf"
                )
            else:
                lines.append(f"- ✗ {v.term!r} — not found")
        lines.append("")

        lines.append("## 5. Random chunk sample (for hand audit)")
        lines.append("")
        lines.append(
            "_Read each chunk and ask: does it really belong on the listed "
            "shelves? Mark yes/no manually._"
        )
        lines.append("")
        for i, c in enumerate(self.chunk_sample, 1):
            lines.append(f"### [{i}] {c.chunk_id} ({c.source_type})")
            lines.append(f"  shelves: {', '.join(c.attached_shelves) or '(none)'}")
            lines.append(f"  text: {c.text_snippet}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------- helpers


def _snippet(text: str, limit: int = 220) -> str:
    """Single-line preview of a chunk's text, capped + ellipsis-suffixed."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


@dataclass
class _ShelfIndex:
    """Internal: shelf-id-keyed maps + parent chain helper."""

    by_id: dict[str, Shelf]
    by_facet: dict[str, list[Shelf]]
    edges_by_shelf: dict[str, set[str]]  # shelf_id -> {chunk_id, ...}

    def ancestors(self, shelf_id: str) -> list[str]:
        out: list[str] = []
        current = self.by_id.get(shelf_id)
        seen: set[str] = set()
        while current is not None and current.parent_shelf_id is not None:
            if current.parent_shelf_id in seen:
                break  # defensive
            seen.add(current.parent_shelf_id)
            parent = self.by_id.get(current.parent_shelf_id)
            if parent is None:
                break
            out.append(parent.label)
            current = parent
        return list(reversed(out))  # root first

    def children(self, shelf_id: str) -> list[Shelf]:
        return [s for s in self.by_id.values() if s.parent_shelf_id == shelf_id]


# ---------------------------------------------------------------- runner


def quality_report(
    chunk_store: ChunkStore,
    graph_store: GraphStore,
    ontology: FoodOnAPI,
    *,
    config_hash: str,
    facet: Facet = "foods",
    top_n: int = 20,
    sample_size: int = 20,
    canonical_terms: tuple[str, ...] = _DEFAULT_CANONICAL_TERMS,
    seed: int = 0,
) -> QualityReport:
    """Build the five-section quality report. Read-only — no writes."""
    shelves = graph_store.list_shelves()
    edges = graph_store.list_chunk_shelf_attachments()

    # Invert: shelf_id -> set(chunk_id)
    chunks_by_shelf: dict[str, set[str]] = {}
    for chunk_id, shelf_ids in edges.items():
        for sid in shelf_ids:
            chunks_by_shelf.setdefault(sid, set()).add(chunk_id)

    index = _ShelfIndex(
        by_id={s.shelf_id: s for s in shelves},
        by_facet={
            f: [s for s in shelves if s.facet == f]
            for f in {s.facet for s in shelves}
        },
        edges_by_shelf=chunks_by_shelf,
    )

    facet_shelves = index.by_facet.get(facet, [])

    report = QualityReport(
        config_hash=config_hash,
        generated_at=datetime.now(UTC),
        facet=facet,
        n_shelves=len(facet_shelves),
    )
    report.top_shelves = _build_top_shelves(facet_shelves, chunk_store, index, top_n)
    report.hierarchy_walkthrough = _build_hierarchy(facet_shelves, index, top_n=5)
    report.suspicious_shelves = _flag_suspicious(facet_shelves, index)
    report.canonical_vocab_check = _check_canonical_vocab(
        canonical_terms, facet_shelves, ontology
    )
    report.chunk_sample = _build_chunk_sample(
        chunk_store, index, sample_size=sample_size, seed=seed
    )
    return report


# ---------------------------------------------------------------- 1. top shelves


def _build_top_shelves(
    facet_shelves: list[Shelf],
    chunk_store: ChunkStore,
    index: _ShelfIndex,
    top_n: int,
) -> list[ShelfSnapshot]:
    top = sorted(facet_shelves, key=lambda s: -s.chunk_count)[:top_n]
    out: list[ShelfSnapshot] = []
    for shelf in top:
        # Fetch up to 3 sample chunks via the chunk store's terms-filter on
        # shelf_ids. Skip cleanly if the shelf has no chunks attached.
        chunk_ids = list(index.edges_by_shelf.get(shelf.shelf_id, set()))[:3]
        chunks = chunk_store.get_many(chunk_ids) if chunk_ids else []
        snippets = [_snippet(c.text) for c in chunks]
        direct_share = (
            shelf.support_direct / shelf.chunk_count if shelf.chunk_count > 0 else 0.0
        )
        out.append(
            ShelfSnapshot(
                label=shelf.label,
                shelf_id=shelf.shelf_id,
                depth=shelf.depth,
                facet=shelf.facet,
                chunk_count=shelf.chunk_count,
                support_direct=shelf.support_direct,
                support_lifted=shelf.support_lifted,
                direct_share=round(direct_share, 3),
                sample_chunks=snippets,
            )
        )
    return out


# ---------------------------------------------------------------- 2. hierarchy


def _build_hierarchy(
    facet_shelves: list[Shelf],
    index: _ShelfIndex,
    top_n: int = 5,
) -> list[HierarchyEntry]:
    top = sorted(facet_shelves, key=lambda s: -s.chunk_count)[:top_n]
    out: list[HierarchyEntry] = []
    for shelf in top:
        children = index.children(shelf.shelf_id)
        # Sample up to 8 children, prefer deepest for "shows the depth"
        children_sorted = sorted(children, key=lambda s: (-s.depth, s.label))[:8]
        snap = ShelfSnapshot(
            label=shelf.label,
            shelf_id=shelf.shelf_id,
            depth=shelf.depth,
            facet=shelf.facet,
            chunk_count=shelf.chunk_count,
            support_direct=shelf.support_direct,
            support_lifted=shelf.support_lifted,
            direct_share=round(
                shelf.support_direct / shelf.chunk_count if shelf.chunk_count else 0.0,
                3,
            ),
        )
        out.append(
            HierarchyEntry(
                shelf=snap,
                ancestor_chain=index.ancestors(shelf.shelf_id),
                sample_descendants=[c.label for c in children_sorted],
            )
        )
    return out


# ---------------------------------------------------------------- 3. suspicious


def _flag_suspicious(
    facet_shelves: list[Shelf],
    index: _ShelfIndex,
) -> list[SuspiciousShelf]:
    """Conservative heuristics — only flag obvious junk.

    Each rule fires on a specific failure mode we've actually seen during
    development; together they target ≤ 5% of shelves on a healthy graph.
    """
    flagged: list[SuspiciousShelf] = []

    for shelf in facet_shelves:
        label_l = shelf.label.lower()

        # A. Label patterns that scream "this is not a human-readable term"
        if any(label_l.startswith(prefix) for prefix in ("0", "1", "2", "3", "4")) \
                and " - " in shelf.label:
            # e.g. "10210 - legumes (efsa foodex2)"
            flagged.append(SuspiciousShelf(
                label=shelf.label, shelf_id=shelf.shelf_id,
                reason="label has EFSA-style numeric code prefix",
                metric=f"label={shelf.label!r}",
            ))
            continue
        if "datum" in label_l or "data" in label_l.split():
            # FoodOn has "food calorie datum" etc — data records, not foods.
            flagged.append(SuspiciousShelf(
                label=shelf.label, shelf_id=shelf.shelf_id,
                reason="label refers to a 'datum'/'data' record, not a food",
                metric=f"label={shelf.label!r}",
            ))
            continue
        if label_l.endswith(" (raw)") or label_l.endswith(" (canned)"):
            # Processing-state labels — usually fine but flag for review since
            # the chunk text may not match that specific state.
            flagged.append(SuspiciousShelf(
                label=shelf.label, shelf_id=shelf.shelf_id,
                reason="label carries a processing-state qualifier",
                metric=f"label={shelf.label!r}",
            ))
            continue

        # B. Single-child shelf whose child is its only descendant — should
        #    have been collapsed. Flag with the survivor name.
        children = index.children(shelf.shelf_id)
        if len(children) == 1 and not index.children(children[0].shelf_id):
            flagged.append(SuspiciousShelf(
                label=shelf.label, shelf_id=shelf.shelf_id,
                reason=f"only child is {children[0].label!r}; collapse may have missed",
                metric=f"sole_child={children[0].shelf_id}",
            ))
            continue

        # C. Tiny shelves that survived via whitelist or depth-cap fluke
        if shelf.chunk_count == 0:
            flagged.append(SuspiciousShelf(
                label=shelf.label, shelf_id=shelf.shelf_id,
                reason="shelf has zero attached chunks",
                metric=f"chunk_count={shelf.chunk_count}",
            ))
            continue

    return flagged


# ---------------------------------------------------------------- 4. vocab


def _check_canonical_vocab(
    terms: tuple[str, ...],
    facet_shelves: list[Shelf],
    ontology: FoodOnAPI,
) -> list[VocabHit]:
    """For each canonical term: is there a shelf? Or only an ontology entry?

    Match priority: exact-label match on shelves → label-substring match →
    ontology `name_to_id` → ontology `search` substring → not found.
    """
    shelves_by_label_lower = {s.label.lower(): s for s in facet_shelves}
    shelves_by_foodon: dict[str, Shelf] = {
        s.foodon_id: s for s in facet_shelves if s.foodon_id is not None
    }

    out: list[VocabHit] = []
    for term in terms:
        term_l = term.lower()
        # 1. exact label match
        shelf = shelves_by_label_lower.get(term_l)
        if shelf is not None:
            out.append(VocabHit(
                term=term, status="found_as_shelf",
                shelf_label=shelf.label, shelf_id=shelf.shelf_id,
                chunk_count=shelf.chunk_count,
                ontology_id=shelf.foodon_id,
            ))
            continue
        # 2. resolve via ontology name lookup
        oid = ontology.name_to_id(term)
        if oid is not None and oid in shelves_by_foodon:
            shelf = shelves_by_foodon[oid]
            out.append(VocabHit(
                term=term, status="found_as_shelf",
                shelf_label=shelf.label, shelf_id=shelf.shelf_id,
                chunk_count=shelf.chunk_count,
                ontology_id=oid,
            ))
            continue
        if oid is not None:
            out.append(VocabHit(
                term=term, status="found_in_ontology_only",
                ontology_id=oid,
            ))
            continue
        # 3. ontology substring search — last-ditch (e.g. "DASH diet" -> "DASH")
        candidates = ontology.search(term, limit=1)
        if candidates and candidates[0] in shelves_by_foodon:
            shelf = shelves_by_foodon[candidates[0]]
            out.append(VocabHit(
                term=term, status="found_as_shelf",
                shelf_label=shelf.label, shelf_id=shelf.shelf_id,
                chunk_count=shelf.chunk_count,
                ontology_id=candidates[0],
            ))
        elif candidates:
            out.append(VocabHit(
                term=term, status="found_in_ontology_only",
                ontology_id=candidates[0],
            ))
        else:
            out.append(VocabHit(term=term, status="not_found"))
    return out


# ---------------------------------------------------------------- 5. chunk sample


def _build_chunk_sample(
    chunk_store: ChunkStore,
    index: _ShelfIndex,
    *,
    sample_size: int,
    seed: int,
) -> list[ChunkAuditEntry]:
    """Pick `sample_size` random *attached* chunks and format each for hand audit.

    Only sample chunks with at least one shelf — sampling unattached chunks
    here wouldn't tell the expert anything about Layer A quality. Use the
    audit module for unattached-chunk diagnostics.
    """
    all_chunks = chunk_store.scan()
    attached = [c for c in all_chunks if c.shelf_ids]
    if not attached:
        return []
    rng = random.Random(seed)
    picked = rng.sample(attached, k=min(sample_size, len(attached)))
    out: list[ChunkAuditEntry] = []
    for chunk in picked:
        labels = [
            index.by_id[sid].label
            for sid in chunk.shelf_ids
            if sid in index.by_id
        ]
        out.append(
            ChunkAuditEntry(
                chunk_id=chunk.chunk_id,
                text_snippet=_snippet(chunk.text, limit=280),
                source_type=chunk.source_type,
                attached_shelves=labels,
            )
        )
    return out


__all__ = [
    "ChunkAuditEntry",
    "HierarchyEntry",
    "QualityReport",
    "ShelfSnapshot",
    "SuspiciousShelf",
    "VocabHit",
    "quality_report",
]
