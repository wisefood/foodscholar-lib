"""Extended data model for KGGen with passage-aware graph representation.

ExtendedGraph adds to the base Graph:
  - passages: dict mapping passage_id (UUID) → {"text": str, "metadata": dict}
  - triplet_passages: dict mapping (subject, predicate, object) → set of passage_ids
  - entity_passages: dict mapping entity → set of passage_ids
"""

import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


# Serialization key for tuple keys in JSON
TRIPLE_SEP = "::"


def _triple_to_key(triple: Tuple[str, str, str]) -> str:
    """Serialize a triple tuple to a JSON-safe string key."""
    return TRIPLE_SEP.join(triple)


def _key_to_triple(key: str) -> Tuple[str, str, str]:
    """Deserialize a JSON-safe string key back to a triple tuple."""
    parts = key.split(TRIPLE_SEP)
    if len(parts) != 3:
        raise ValueError(f"Invalid triple key: {key}")
    return (parts[0], parts[1], parts[2])


def _loose_normalize(text: str) -> str:
    """Looser string normalization for stale provenance key repair."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"^(the|a|an)\s+", "", normalized)
    return normalized


class ExtendedGraph(BaseModel):
    """A knowledge graph with passage-level provenance tracking.

    Extends the original Graph model with:
    - passages: full registry of source passages (text + metadata)
    - triplet_passages: per-triplet provenance (which passages each triplet came from)
    - entity_passages: per-entity provenance (which passages each entity appeared in)

    After deduplication, triplet_passages can contain multiple passage_ids per triplet
    (when aliased entities or triplets collapse from different source passages).

    triplet_passages accepts both tuple keys (Tuple[str,str,str]) and string keys
    ('s::p::o'). Tuple keys are auto-serialized on construction.
    """

    entities: Set[str] = Field(
        ..., description="All entities including additional ones from response"
    )
    edges: Set[str] = Field(..., description="All edges")
    relations: Set[Tuple[str, str, str]] = Field(
        ..., description="List of (subject, predicate, object) triples"
    )
    entity_clusters: Optional[Dict[str, Set[str]]] = Field(
        default=None,
        description="Mapping from canonical entity → set of aliases",
    )
    edge_clusters: Optional[Dict[str, Set[str]]] = Field(
        default=None,
        description="Mapping from canonical edge → set of aliases",
    )
    entity_metadata: Optional[Dict[str, Set[str]]] = Field(
        default=None,
        description="Per-entity metadata (e.g., batch_id from extraction)",
    )

    # ── EXTENDED: Passage-aware fields ──
    passages: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Passage registry: passage_id (UUID) → {'text': str, 'metadata': dict}",
    )
    triplet_passages: Optional[Dict[str, Set[str]]] = Field(
        default=None,
        description="Triplet provenance: triple_key → set of passage_ids. "
        "Key format: 'subject::predicate::object' (TRIPLE_SEP-joined). "
        "After dedup, a triplet can have multiple passage_ids. "
        "Accepts both tuple and string keys on construction.",
    )
    entity_passages: Optional[Dict[str, Set[str]]] = Field(
        default=None,
        description="Entity provenance: entity_name → set of passage_ids",
    )

    @field_validator("triplet_passages", mode="before")
    @classmethod
    def _normalize_triplet_keys(cls, v: Any) -> Any:
        """Convert tuple keys in triplet_passages to string keys."""
        if v is None:
            return None
        if isinstance(v, dict):
            result: Dict[str, Set[str]] = {}
            for key, val in v.items():
                if isinstance(key, tuple):
                    result[_triple_to_key(key)] = set(val)
                else:
                    result[str(key)] = set(val)
            return result
        return v

    @model_validator(mode="after")
    def _sync_relations_with_entities_and_edges(self) -> "ExtendedGraph":
        """Ensure relation endpoints/predicates exist in the graph and provenance stays aligned.

        Some relation extractors can emit subject/object strings that were not returned
        by the entity extractor. We preserve the relation and add the missing endpoints
        into `entities`, mirroring the original kg-gen behavior. When triplet-level
        provenance exists, we also backfill entity-level provenance for those endpoints.

        This validator also normalizes stale provenance keys using cluster mappings.
        Older saved chunk graphs can contain deduplicated relations/entities while
        `triplet_passages` still references pre-dedup aliases. We repair those keys
        through `entity_clusters` and `edge_clusters` on load.
        """
        entity_alias_to_canonical: Dict[str, str] = {}
        if self.entity_clusters:
            for canonical, aliases in self.entity_clusters.items():
                entity_alias_to_canonical[canonical] = canonical
                for alias in aliases:
                    entity_alias_to_canonical[alias] = canonical

        edge_alias_to_canonical: Dict[str, str] = {}
        if self.edge_clusters:
            for canonical, aliases in self.edge_clusters.items():
                edge_alias_to_canonical[canonical] = canonical
                for alias in aliases:
                    edge_alias_to_canonical[alias] = canonical

        if self.entity_metadata:
            normalized_entity_metadata: Dict[str, Set[str]] = {}
            for entity_name, metadata_set in self.entity_metadata.items():
                canonical_entity = entity_alias_to_canonical.get(entity_name, entity_name)
                normalized_entity_metadata.setdefault(canonical_entity, set()).update(metadata_set)
            self.entity_metadata = normalized_entity_metadata

        if self.entity_passages:
            normalized_entity_passages: Dict[str, Set[str]] = {}
            for entity_name, passage_ids in self.entity_passages.items():
                canonical_entity = entity_alias_to_canonical.get(entity_name, entity_name)
                normalized_entity_passages.setdefault(canonical_entity, set()).update(passage_ids)
            self.entity_passages = normalized_entity_passages

        if self.triplet_passages:
            normalized_triplet_passages: Dict[str, Set[str]] = {}
            for triple_key, passage_ids in self.triplet_passages.items():
                subject, predicate, obj = _key_to_triple(triple_key)
                canonical_subject = entity_alias_to_canonical.get(subject, subject)
                canonical_predicate = edge_alias_to_canonical.get(predicate, predicate)
                canonical_object = entity_alias_to_canonical.get(obj, obj)
                canonical_key = _triple_to_key(
                    (canonical_subject, canonical_predicate, canonical_object)
                )
                normalized_triplet_passages.setdefault(canonical_key, set()).update(passage_ids)
            relation_candidates = list(self.relations)

            def _resolve_to_existing_relation(
                subject: str, predicate: str, obj: str
            ) -> Tuple[str, str, str]:
                candidate = (subject, predicate, obj)
                if candidate in self.relations:
                    return candidate

                loose_subject = _loose_normalize(subject)
                loose_object = _loose_normalize(obj)
                loose_predicate = _loose_normalize(predicate)

                exact_pair_matches = [
                    relation
                    for relation in relation_candidates
                    if relation[0] == subject and relation[2] == obj
                ]
                exact_pair_predicate_matches = [
                    relation
                    for relation in exact_pair_matches
                    if _loose_normalize(relation[1]) == loose_predicate
                ]
                if len(exact_pair_predicate_matches) == 1:
                    return exact_pair_predicate_matches[0]

                loose_pair_matches = [
                    relation
                    for relation in relation_candidates
                    if _loose_normalize(relation[0]) == loose_subject
                    and _loose_normalize(relation[2]) == loose_object
                ]
                loose_pair_predicate_matches = [
                    relation
                    for relation in loose_pair_matches
                    if _loose_normalize(relation[1]) == loose_predicate
                ]
                if len(loose_pair_predicate_matches) == 1:
                    return loose_pair_predicate_matches[0]

                if len(exact_pair_matches) == 1:
                    return exact_pair_matches[0]
                if len(loose_pair_matches) == 1:
                    return loose_pair_matches[0]

                predicate_matches = loose_pair_matches or exact_pair_matches
                if predicate_matches:
                    scored_matches = sorted(
                        predicate_matches,
                        key=lambda relation: SequenceMatcher(
                            None,
                            loose_predicate,
                            _loose_normalize(relation[1]),
                        ).ratio(),
                        reverse=True,
                    )
                    if scored_matches:
                        return scored_matches[0]

                return candidate

            aligned_triplet_passages: Dict[str, Set[str]] = {}
            for triple_key, passage_ids in normalized_triplet_passages.items():
                subject, predicate, obj = _key_to_triple(triple_key)
                resolved_relation = _resolve_to_existing_relation(subject, predicate, obj)
                resolved_key = _triple_to_key(resolved_relation)
                aligned_triplet_passages.setdefault(resolved_key, set()).update(passage_ids)

            self.triplet_passages = aligned_triplet_passages

        if self.triplet_passages is not None and self.entity_passages is None:
            self.entity_passages = {}

        for subject, predicate, obj in self.relations:
            self.entities.add(subject)
            self.entities.add(obj)
            self.edges.add(predicate)

            if self.entity_passages is not None and self.triplet_passages is not None:
                triple_key = _triple_to_key((subject, predicate, obj))
                passage_ids = self.triplet_passages.get(triple_key, set())
                if passage_ids:
                    self.entity_passages.setdefault(subject, set()).update(passage_ids)
                    self.entity_passages.setdefault(obj, set()).update(passage_ids)

        return self

    # ── Convenience accessors for triplet_passages with tuple keys ──

    def get_triplet_pids(self, triple: Tuple[str, str, str]) -> Set[str]:
        """Get passage_ids for a triple (tuple key)."""
        if self.triplet_passages is None:
            return set()
        return self.triplet_passages.get(_triple_to_key(triple), set())

    def set_triplet_pids(self, triple: Tuple[str, str, str], pids: Set[str]):
        """Set passage_ids for a triple (tuple key)."""
        if self.triplet_passages is None:
            self.triplet_passages = {}
        self.triplet_passages[_triple_to_key(triple)] = pids

    def add_triplet_pid(self, triple: Tuple[str, str, str], pid: str):
        """Add a single passage_id to a triple's provenance set."""
        if self.triplet_passages is None:
            self.triplet_passages = {}
        key = _triple_to_key(triple)
        if key not in self.triplet_passages:
            self.triplet_passages[key] = set()
        self.triplet_passages[key].add(pid)

    def iter_triplet_pids(self):
        """Iterate over (triple_tuple, pids_set) pairs."""
        if self.triplet_passages is None:
            return
        for key, pids in self.triplet_passages.items():
            yield (_key_to_triple(key), pids)

    def triple_has_multiple_passages(self, triple: Tuple[str, str, str]) -> bool:
        """Check if a triple has >1 passage_id."""
        return len(self.get_triplet_pids(triple)) > 1

    @staticmethod
    def from_file(file_path: str) -> "ExtendedGraph":
        """
        Load the graph from a JSON file.
        Fix graph entities and edges for missing ones defined in relations.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Deserialize string-keyed fields back to tuples/sets
        if "triplet_passages" in data and data["triplet_passages"] is not None:
            # Keys are already strings (stored as 's::p::o') — no conversion needed
            data["triplet_passages"] = {
                str(k): set(v) for k, v in data["triplet_passages"].items()
            }
        if "entity_passages" in data and data["entity_passages"] is not None:
            data["entity_passages"] = {
                k: set(v) for k, v in data["entity_passages"].items()
            }
        if "entity_metadata" in data and data["entity_metadata"] is not None:
            data["entity_metadata"] = {
                k: set(v) for k, v in data["entity_metadata"].items()
            }
        if "entity_clusters" in data and data["entity_clusters"] is not None:
            data["entity_clusters"] = {
                k: set(v) for k, v in data["entity_clusters"].items()
            }
        if "edge_clusters" in data and data["edge_clusters"] is not None:
            data["edge_clusters"] = {
                k: set(v) for k, v in data["edge_clusters"].items()
            }

        graph = ExtendedGraph.model_validate(data)

        # Fix graph entities and edges
        for relation in graph.relations:
            if relation[0] not in graph.entities:
                graph.entities.add(relation[0])
            if relation[1] not in graph.edges:
                graph.edges.add(relation[1])
            if relation[2] not in graph.entities:
                graph.entities.add(relation[2])

        return graph

    def to_file(self, file_path: str):
        """
        Save the graph to a JSON file.
        """
        data = self.model_dump(mode="python")

        # Serialize tuple-keyed fields (keys are already strings)
        if self.triplet_passages is not None:
            data["triplet_passages"] = {
                str(k): list(v)
                for k, v in self.triplet_passages.items()
            }

        # Convert sets to lists for JSON serialization
        data["entities"] = list(self.entities)
        data["edges"] = list(self.edges)
        data["relations"] = [list(r) for r in self.relations]

        if data.get("entity_metadata"):
            data["entity_metadata"] = {
                k: list(v) for k, v in data["entity_metadata"].items()
            }
        if data.get("entity_passages"):
            data["entity_passages"] = {
                k: list(v) for k, v in data["entity_passages"].items()
            }
        if data.get("entity_clusters"):
            data["entity_clusters"] = {
                k: list(v) for k, v in data["entity_clusters"].items()
            }
        if data.get("edge_clusters"):
            data["edge_clusters"] = {
                k: list(v) for k, v in data["edge_clusters"].items()
            }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def stats(self, name: Optional[str] = None):
        """
        Print the stats of the graph.
        """
        label = name or "ExtendedGraph"
        print(
            f"{label} with:\n"
            f"\t{len(self.entities)} entities\n"
            f"\t{len(self.edges)} edges\n"
            f"\t{len(self.relations)} relations\n"
            f"\t{len(self.passages)} passages"
        )
        if self.triplet_passages:
            multi = sum(
                1 for pids in self.triplet_passages.values() if len(pids) > 1
            )
            single = sum(
                1 for pids in self.triplet_passages.values() if len(pids) == 1
            )
            print(
                f"\t{single} triplets with 1 passage, "
                f"{multi} triplets with >1 passage"
            )
