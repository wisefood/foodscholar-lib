"""Semantic-hash based deduplication with passage-aware entity/triplet remapping."""
import unicodedata
from typing import Any, Collection, Dict, List, Optional, Set, Tuple, cast
from kggen_extended.models import ExtendedGraph, _triple_to_key, _key_to_triple
from semhash import SemHash
import inflect


class DeduplicateList:
    inflect_engine: 'inflect.engine'
    original_map: Dict[str, str]
    items_map: Dict[str, str]
    duplicates: Dict[str, str]
    deduplicated: List[str]

    # Stats values
    total_items: int
    deduplicated_items: int
    duplicate_items: int
    reduction: float

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.inflect_engine = inflect.engine()
        self.original_map = {}
        self.items_map = {}
        self.duplicates = {}
        self.deduplicated = []

    def normalize(self, text: str) -> str:
        """
        Normalize a text.
        """
        return unicodedata.normalize("NFKC", text)

    def singularize(self, text: str) -> str:
        """
        Singularize a text.
        """
        # singularize each token when it looks like a plural noun
        tokens = []
        for tok in text.split():
            sing = self.inflect_engine.singular_noun(cast(Any, tok))
            tokens.append(sing if isinstance(sing, str) and sing else tok)
        return " ".join(tokens).strip()

    def deduplicate(self, items: Collection[str]) -> None:
        """
        Deduplicate a list of items using semantic hashing.
        Before deduplication, items are normalized and singularized.

        Args:
            items: List of items to deduplicate

        Returns:
            List of deduplicated items
        """
        self.total_items = len(items)

        # Normalize and singularize each string
        normalized_items = set()
        for item in items:
            normalized = self.normalize(item)
            singular = self.singularize(normalized)
            self.original_map[item] = singular
            self.items_map[singular] = item
            normalized_items.add(singular)

        # Deduplicate the normalized strings
        semhash = SemHash.from_records(records=list(normalized_items))
        deduplication_result = semhash.self_deduplicate(threshold=self.threshold)

        self.deduplicated_items = len(deduplication_result.selected)
        self.duplicate_items = len(deduplication_result.duplicates)
        self.reduction = (self.duplicate_items / self.total_items) * 100

        # Map back to original strings
        duplicates = deduplication_result.duplicates
        for duplicate in duplicates:
            original = duplicate.record
            # Check if duplicates list is not empty before accessing
            if (
                duplicate.duplicates
                and len(duplicate.duplicates) > 0
                and len(duplicate.duplicates[0]) > 0
            ):
                duplicate_value = duplicate.duplicates[0][0]
                self.items_map[original] = self.items_map[duplicate_value]
                if not original in self.duplicates:
                    self.duplicates[original] = duplicate_value

        self.deduplicated = deduplication_result.selected

    def stats(self) -> str:
        return f"Total items: {self.total_items}; Deduplicated items: {self.deduplicated_items}; Duplicate items: {self.duplicate_items}; Reduction: {self.reduction:.1f}"


def run_semhash_deduplication(
    graph: ExtendedGraph,
    similarity_threshold: float = 0.95,
) -> ExtendedGraph:
    """
    Deduplicate the graph using semantic hashing, with full support
    for ExtendedGraph's passage-aware fields (entity_passages,
    triplet_passages, passages).

    Key extended behavior:
    - entity_passages: when alias entities collapse to a canonical entity,
      their passage_id sets are UNIONED.
    - triplet_passages: when distinct triplets collapse to the same
      canonical (s,p,o) triplet, their passage_id sets are UNIONED.
    - entity_clusters: populated from DeduplicateList.duplicates so
      downstream code has a reliable alias→canonical map.
    """
    # Deduplicate each graph component
    entities_dedup = DeduplicateList(similarity_threshold)
    entities_dedup.deduplicate(graph.entities)
    edges_dedup = DeduplicateList(similarity_threshold)
    edges_dedup.deduplicate(graph.edges)

    # ── Build alias→canonical lookup for entities ──
    # canonical_entity: the representative name for a cluster
    # aliases: set of all entity names that map to the same canonical
    alias_to_canonical: Dict[str, str] = {}

    # Normalize all entities to get the key
    for entity in graph.entities:
        normalized = entities_dedup.normalize(entity)
        singular = entities_dedup.singularize(normalized)
        if singular in entities_dedup.items_map:
            canonical = entities_dedup.items_map[singular]
            alias_to_canonical[entity] = canonical
        else:
            alias_to_canonical[entity] = entity

    def _get_canonical_entity(entity_name: str) -> str:
        """Resolve an entity name to its canonical form."""
        if entity_name in alias_to_canonical:
            return alias_to_canonical[entity_name]
        return entity_name

    def _get_canonical_edge(edge_name: str) -> str:
        """Resolve an edge/predicate name to its canonical form."""
        if edge_name in edges_dedup.original_map:
            return edges_dedup.items_map[edges_dedup.original_map[edge_name]]
        return edge_name

    def _get_relation(relation: Tuple[str, str, str]) -> Tuple[str, str, str]:
        """
        Get the transformed relation.
        """
        # Handle case where entity might not be in original_map due to normalization
        first_entity_original = relation[0]
        if first_entity_original in entities_dedup.original_map:
            first_entity = entities_dedup.items_map[
                entities_dedup.original_map[first_entity_original]
            ]
        else:
            # If not found, use the original entity (it might have been normalized differently)
            first_entity = first_entity_original

        second_entity_original = relation[2]
        if second_entity_original in entities_dedup.original_map:
            second_entity = entities_dedup.items_map[
                entities_dedup.original_map[second_entity_original]
            ]
        else:
            # If not found, use the original entity
            second_entity = second_entity_original

        edge_original = relation[1]
        if edge_original in edges_dedup.original_map:
            edge = edges_dedup.items_map[edges_dedup.original_map[edge_original]]
        else:
            # If not found, use the original edge
            edge = edge_original

        return (first_entity, edge, second_entity)

    # Deduplicate the graph
    new_entities = {
        entities_dedup.items_map[item] for item in entities_dedup.deduplicated
    }
    new_edges = {edges_dedup.items_map[item] for item in edges_dedup.deduplicated}
    new_relations = {_get_relation(relation) for relation in graph.relations}

    # ── EXTENDED: Remap entity_metadata ──
    # Update entity_metadata keys to match deduplicated entity names
    new_entity_metadata: Optional[Dict[str, Set[str]]] = None
    if graph.entity_metadata:
        new_entity_metadata = {}
        for original_entity, metadata_set in graph.entity_metadata.items():
            deduped_entity = _get_canonical_entity(original_entity)
            # Merge metadata sets when entities are deduplicated together
            if deduped_entity in new_entity_metadata:
                new_entity_metadata[deduped_entity].update(metadata_set)
            else:
                new_entity_metadata[deduped_entity] = metadata_set.copy()

    # ── EXTENDED: Remap entity_passages ──
    # When alias entities collapse to a canonical entity, union their passage_id sets.
    new_entity_passages: Dict[str, Set[str]] = {}
    if graph.entity_passages:
        for original_entity, passage_ids in graph.entity_passages.items():
            deduped_entity = _get_canonical_entity(original_entity)
            if deduped_entity in new_entity_passages:
                new_entity_passages[deduped_entity].update(passage_ids)
            else:
                new_entity_passages[deduped_entity] = set(passage_ids)

    # ── EXTENDED: Remap triplet_passages ──
    # When distinct triplets collapse to the same canonical (s,p,o), union their
    # passage_id sets. This is how a deduped triplet can connect to multiple passages.
    # Keys in triplet_passages are serialized strings ('s::p::o').
    new_triplet_passages: Dict[str, Set[str]] = {}
    if graph.triplet_passages:
        for tkey, passage_ids in graph.triplet_passages.items():
            s, p, o = _key_to_triple(tkey)
            canonical_s = _get_canonical_entity(s)
            canonical_p = _get_canonical_edge(p)
            canonical_o = _get_canonical_entity(o)
            canonical_tkey = _triple_to_key((canonical_s, canonical_p, canonical_o))

            if canonical_tkey in new_triplet_passages:
                new_triplet_passages[canonical_tkey].update(passage_ids)
            else:
                new_triplet_passages[canonical_tkey] = set(passage_ids)

    # ── EXTENDED: Build entity_clusters from DeduplicateList.duplicates ──
    # The original SEMHASH path doesn't return entity_clusters (LM path does).
    # We build them here so downstream code has a reliable alias→canonical map.
    entity_clusters: Dict[str, Set[str]] = {}
    for alias, canonical_norm in entities_dedup.duplicates.items():
        # Map the normalized key back to the actual entity string
        if canonical_norm in entities_dedup.items_map:
            canonical = entities_dedup.items_map[canonical_norm]
        else:
            canonical = canonical_norm

        if canonical not in entity_clusters:
            entity_clusters[canonical] = set()
        entity_clusters[canonical].add(alias)

    # Also add self-mappings for entities that weren't deduped
    for entity in new_entities:
        if entity not in entity_clusters:
            entity_clusters[entity] = set()

    edge_clusters: Dict[str, Set[str]] = {}
    for alias, canonical_norm in edges_dedup.duplicates.items():
        if canonical_norm in edges_dedup.items_map:
            canonical = edges_dedup.items_map[canonical_norm]
        else:
            canonical = canonical_norm

        if canonical not in edge_clusters:
            edge_clusters[canonical] = set()
        edge_clusters[canonical].add(alias)

    for edge in new_edges:
        if edge not in edge_clusters:
            edge_clusters[edge] = set()

    # ── Build the deduplicated ExtendedGraph ──
    return ExtendedGraph(
        entities=new_entities,
        edges=new_edges,
        relations=new_relations,
        entity_metadata=new_entity_metadata,
        entity_clusters=entity_clusters,
        edge_clusters=edge_clusters,
        # Passage-aware fields (already remapped above)
        passages=graph.passages,  # passage registry unchanged
        entity_passages=new_entity_passages if new_entity_passages else None,
        triplet_passages=new_triplet_passages if new_triplet_passages else None,
    )
