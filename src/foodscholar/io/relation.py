"""Pydantic data carrier for Layer 0 — typed relations between entities.

A `Relation` is the edge counterpart to `io.entity.Entity`: one record per
distinct ``(subject_id, predicate, object_id)`` across the corpus, carrying the
surface forms that collapsed into it plus **the chunks it was extracted from**.

Layer 0 sits *under* the entity graph, not beside Layer C::

    Layer C   cards          (cited write-ups)
    Layer B   themes         (per-shelf communities)
    Layer A   shelves        (FoodOn backbone)
    ------------------------------------------------------------
    Layer 0   relations      (:Entity)-[:RELATED {predicate}]->(:Entity)
              entities       (:Entity)
              chunks         (:Chunk)

Built by `fs.build_relations()` from chunk text, after `fs.build_entities()`.
Read via the `fs.relations` namespace.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from foodscholar.io.chunk import ChunkId

# How many chunk ids to embed inline in `Relation.chunk_ids`. `chunk_count`
# always carries the true total. Mirrors ENTITY_CHUNK_SAMPLE_CAP.
RELATION_CHUNK_SAMPLE_CAP = 50

NIL_PREFIX = "NIL:"

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ID_SEP = "\x1f"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def nil_id(surface: str) -> str:
    """Sentinel id for an endpoint that did not link to the ontology.

    Unlinked endpoints are *kept*, not dropped: discarding them would silently
    delete every relation touching a concept FoodOn lacks, which for a
    nutrition corpus is most biomarkers, hormones and physiological processes.
    Consumers filter on `Relation.subject_linked` / `object_linked` rather than
    inspecting the id prefix.
    """
    slug = _SLUG_RE.sub("-", surface.strip().lower()).strip("-")
    return f"{NIL_PREFIX}{slug or 'unknown'}"


def is_nil(ontology_id: str) -> bool:
    return ontology_id.startswith(NIL_PREFIX)


def make_relation_id(subject_id: str, predicate: str, object_id: str) -> str:
    """Deterministic, content-addressed id.

    Not a UUID: a re-run over an unchanged corpus must produce identical ids so
    `RelationStore.upsert` is a true upsert and cross-store parity audits work.

    Each field is **length-prefixed** before hashing. A plain separator-joined
    payload is ambiguous — ``("A\x1fp", "x", "B")`` and ``("A", "p\x1fx", "B")``
    would serialize identically and collide. Predicates are free-text LLM
    output, so that input is not hypothetical.
    """
    payload = _ID_SEP.join(
        f"{len(part)}:{part}" for part in (subject_id, predicate, object_id)
    ).encode("utf-8")
    return f"rel:{hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:16]}"


class Relation(BaseModel):
    """A typed, corpus-grounded edge between two entities.

    One record per distinct ``(subject_id, predicate, object_id)``. Endpoints
    are ontology ids when the extracted surface linked (the common case), and a
    ``NIL:<slug>`` sentinel otherwise — always paired with the matching
    ``*_linked`` flag so consumers never string-sniff the id.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_id: str
    """Content-addressed; see `make_relation_id`."""

    subject_id: str
    predicate: str
    object_id: str

    subject_surfaces: tuple[str, ...] = Field(default_factory=tuple)
    """Every extracted surface that grounded to `subject_id` for this relation
    — the audit trail for the grounding step."""
    object_surfaces: tuple[str, ...] = Field(default_factory=tuple)
    predicate_surfaces: tuple[str, ...] = Field(default_factory=tuple)
    """Predicate variants collapsed into `predicate` by dedup."""

    chunk_ids: tuple[ChunkId, ...] = Field(default_factory=tuple)
    """PROVENANCE — the chunks this relation was extracted from, capped at
    `RELATION_CHUNK_SAMPLE_CAP`. Every id here must resolve in the chunk store;
    this is the invariant the whole integration exists to preserve."""
    chunk_count: int = 0
    """True number of distinct source chunks (may exceed `len(chunk_ids)`)."""
    mention_count: int = 0
    """Times the triple was extracted before dedup. A crude confidence proxy —
    validate it against a hand-scored sample before filtering on it."""

    subject_linked: bool = True
    object_linked: bool = True

    extractor_version: str = ""
    dedupe_version: str = ""
    last_seen: datetime = Field(default_factory=_utcnow)

    @property
    def is_fully_grounded(self) -> bool:
        """True when both endpoints resolved to real ontology ids."""
        return self.subject_linked and self.object_linked

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.subject_id, self.predicate, self.object_id)
