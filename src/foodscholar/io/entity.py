"""Pydantic data carrier for first-class linked entities.

An `Entity` is the dedup'd, ontology-resolved counterpart to the per-chunk
`EntityLink`: one record per distinct `ontology_id` across the corpus,
carrying the label / synonyms / ancestor chain (from the ontology API) plus
the corpus-side aggregate stats (`mention_count`, `chunk_count`, sample
`chunk_ids`). Stored in `fs.entity_store` (Elastic) and also exposed as
`(:Entity)` nodes in Neo4j.

Built by `fs.build_entities()` from the chunks already in the chunk store.
Read via the `fs.entities` namespace.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from foodscholar.io.graph import Facet
from foodscholar.io.ontology import OntologyId


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Entity(BaseModel):
    """A first-class entity discovered in the corpus.

    `ontology_id` is the canonical `PREFIX:LOCALID` (e.g. `FOODON:03309927`,
    `CHEBI:16526`). The `prefix` field is the OBO source ontology — callers
    filter on it to scope to FoodOn-only or any other vocabulary subset.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ontology_id: OntologyId
    prefix: str
    """The OBO source ontology (FOODON, CHEBI, GAZ, PATO, UBERON, …)."""

    label: str
    """Preferred label from the ontology, when available. Falls back to the
    surface form of the most-common mention otherwise."""

    synonyms: tuple[str, ...] = Field(default_factory=tuple)
    ancestor_ids: tuple[OntologyId, ...] = Field(default_factory=tuple)
    """Closed transitive ancestor set from the ontology (FoodOn only — other
    OBO prefixes ship empty ancestors since the loader is FoodOn-scoped)."""

    facet_hint: Facet | None = None
    """Best-effort mapping from `Mention.entity_type` to a Layer A facet.
    None when no mention carried a classifiable entity_type."""

    mention_count: int = 0
    """Total number of mentions across the corpus."""

    chunk_count: int = 0
    """Number of distinct chunks that mention this entity."""

    chunk_ids: tuple[str, ...] = Field(default_factory=tuple)
    """Sample of chunk ids that mention the entity. Capped (see
    `Entity.SAMPLE_CAP`) so the document stays bounded even for high-frequency
    entities like FOODON:00001020 ('food'). Use the chunk store's
    `terms`-filter on `foodon_ids` for the full set."""

    last_seen: datetime = Field(default_factory=_utcnow)


# How many chunk ids to embed inline in `Entity.chunk_ids`. `fs.build_entities`
# obeys this; callers wanting the full set should `terms`-filter the chunk
# store on `foodon_ids`.
ENTITY_CHUNK_SAMPLE_CAP = 50
