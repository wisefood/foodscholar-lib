"""Collapse an `ExtractedGraph` into `Relation` records.

Order is deliberate: **dedupe before grounding**. Dedup shrinks the distinct
surface set the linker must encode, and grounding then collapses further (two
distinct surfaces frequently share one ontology id). Reversing the order does
strictly more encoder work for the same result.

Everything here is deterministic given a fixed `ExtractedGraph` — the LLM is
the only nondeterministic step in the phase, and the determinism test asserts
this boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.io.relation import (
    RELATION_CHUNK_SAMPLE_CAP,
    Relation,
    make_relation_id,
)
from foodscholar.logging import get_logger
from foodscholar.relations.dedupe import DedupeResult, deduplicate_surfaces

if TYPE_CHECKING:
    from foodscholar.config import RelationsConfig
    from foodscholar.relations.extracted import ExtractedGraph
    from foodscholar.relations.ground import GroundingResult

_log = get_logger("foodscholar.relations.aggregate")


def dedupe_graph(
    graph: ExtractedGraph, cfg: RelationsConfig
) -> tuple[DedupeResult, DedupeResult]:
    """Dedup entity surfaces and predicates independently.

    Predicates get their own threshold: they are where an open extractor
    sprawls, and retrieval's PPR branch treats distinct predicates between the
    same pair as parallel edges (extra votes), so under-merging them biases
    ranking.
    """
    if not cfg.dedupe.enabled:
        identity_e = DedupeResult({s: s for s in graph.surfaces()}, version="off")
        identity_p = DedupeResult({p: p for p in graph.predicates()}, version="off")
        return identity_e, identity_p

    entities = deduplicate_surfaces(
        graph.surfaces(),
        threshold=cfg.dedupe.similarity_threshold,
        singularize=cfg.dedupe.singularize,
    )
    predicates = deduplicate_surfaces(
        graph.predicates(),
        threshold=cfg.dedupe.predicate_threshold,
        singularize=cfg.dedupe.singularize,
    )
    return entities, predicates


def build_relations(
    graph: ExtractedGraph,
    *,
    grounding: GroundingResult,
    entity_dedupe: DedupeResult,
    predicate_dedupe: DedupeResult,
    extractor_version: str,
    keep_nil: bool = True,
    sample_cap: int = RELATION_CHUNK_SAMPLE_CAP,
) -> list[Relation]:
    """Group triples by grounded ``(subject_id, predicate, object_id)``.

    Provenance unions across every triple that collapses into the group — the
    property the whole pipeline exists to preserve.
    """
    groups: dict[tuple[str, str, str], dict[str, object]] = {}

    for triple in sorted(graph.triples):
        subject_raw, predicate_raw, object_raw = triple
        subject_surface = entity_dedupe.resolve(subject_raw)
        object_surface = entity_dedupe.resolve(object_raw)
        predicate = predicate_dedupe.resolve(predicate_raw)

        subject_id = grounding.id_for(subject_surface)
        object_id = grounding.id_for(object_surface)
        subject_linked = grounding.is_linked(subject_surface)
        object_linked = grounding.is_linked(object_surface)

        if not keep_nil and not (subject_linked and object_linked):
            continue
        if subject_id == object_id:
            # Two surfaces grounded to the same ontology id — the edge is a
            # self-loop after grounding and carries no information.
            continue

        key = (subject_id, predicate, object_id)
        agg = groups.setdefault(
            key,
            {
                "subject_surfaces": set(),
                "object_surfaces": set(),
                "predicate_surfaces": set(),
                "chunk_ids": set(),
                "mention_count": 0,
                "subject_linked": subject_linked,
                "object_linked": object_linked,
            },
        )
        agg["subject_surfaces"].add(subject_raw)  # type: ignore[union-attr]
        agg["object_surfaces"].add(object_raw)  # type: ignore[union-attr]
        agg["predicate_surfaces"].add(predicate_raw)  # type: ignore[union-attr]
        agg["chunk_ids"] |= graph.triple_chunks.get(triple, set())  # type: ignore[operator]
        agg["mention_count"] = int(agg["mention_count"]) + graph.triple_mentions.get(
            triple, 1
        )

    out: list[Relation] = []
    for (subject_id, predicate, object_id), agg in sorted(groups.items()):
        chunk_ids = sorted(agg["chunk_ids"])  # type: ignore[call-overload]
        out.append(
            Relation(
                relation_id=make_relation_id(subject_id, predicate, object_id),
                subject_id=subject_id,
                predicate=predicate,
                object_id=object_id,
                subject_surfaces=tuple(sorted(agg["subject_surfaces"])),  # type: ignore[call-overload]
                object_surfaces=tuple(sorted(agg["object_surfaces"])),  # type: ignore[call-overload]
                predicate_surfaces=tuple(sorted(agg["predicate_surfaces"])),  # type: ignore[call-overload]
                chunk_ids=tuple(chunk_ids[:sample_cap]),
                chunk_count=len(chunk_ids),
                mention_count=int(agg["mention_count"]),
                subject_linked=bool(agg["subject_linked"]),
                object_linked=bool(agg["object_linked"]),
                extractor_version=extractor_version,
                dedupe_version=entity_dedupe.version,
            )
        )
    _log.info(
        "relations.aggregate.done",
        n_triples=len(graph.triples),
        n_relations=len(out),
        n_fully_grounded=sum(1 for r in out if r.is_fully_grounded),
    )
    return out
