"""Read-only lookup API over a list of OntologyTerm.

All lookups are O(1) dict access. Built once from a term list (typically the
output of `load_ontology`), then queried by the linker, layer_a, and layer_c.
"""

from __future__ import annotations

from collections.abc import Iterator

from foodscholar.io.ontology import OntologyId, OntologyTerm


class FoodOnAPI:
    """O(1) lookup surface over OntologyTerm.

    Indexes:
      - id → term
      - lower-case label → id
      - lower-case exact synonym → id (synonyms can resolve to multiple ids)
      - id → set(child ids)   for descendants

    `obsolete` terms are loaded but excluded from name lookups so the linker
    never resolves to a deprecated FoodOn id.
    """

    def __init__(self, terms: list[OntologyTerm]) -> None:
        self._by_id: dict[OntologyId, OntologyTerm] = {t.id: t for t in terms}

        self._name_to_ids: dict[str, list[OntologyId]] = {}
        self._descendants: dict[OntologyId, set[OntologyId]] = {tid: set() for tid in self._by_id}

        for t in terms:
            if t.obsolete:
                continue
            self._index_name(t.label, t.id)
            for syn in t.synonyms:
                self._index_name(syn, t.id)
            for parent in t.ancestor_ids:
                if parent in self._descendants:
                    self._descendants[parent].add(t.id)

    def _index_name(self, name: str, term_id: OntologyId) -> None:
        key = name.strip().lower()
        if not key:
            return
        bucket = self._name_to_ids.setdefault(key, [])
        if term_id not in bucket:
            bucket.append(term_id)

    # -------------------------------------------------------------- lookups

    def __contains__(self, term_id: object) -> bool:
        return isinstance(term_id, str) and term_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[OntologyTerm]:
        return iter(self._by_id.values())

    def terms(self) -> list[OntologyTerm]:
        return list(self._by_id.values())

    def get(self, term_id: OntologyId) -> OntologyTerm | None:
        return self._by_id.get(term_id)

    def name_to_id(self, name: str) -> OntologyId | None:
        """Exact case-insensitive match against label or exact synonym.

        Returns None if no match. If a name maps to multiple ids (rare but
        possible with synonyms), returns the first deterministically. Use
        `name_to_ids` if you need all matches.
        """
        ids = self._name_to_ids.get(name.strip().lower())
        return ids[0] if ids else None

    def name_to_ids(self, name: str) -> list[OntologyId]:
        return list(self._name_to_ids.get(name.strip().lower(), ()))

    def id_to_label(self, term_id: OntologyId) -> str | None:
        t = self._by_id.get(term_id)
        return t.label if t else None

    def id_to_synonyms(
        self, term_id: OntologyId, *, include_related: bool = False
    ) -> list[str]:
        t = self._by_id.get(term_id)
        if t is None:
            return []
        out = list(t.synonyms)
        if include_related:
            out.extend(t.related_synonyms)
        return out

    def id_to_ancestors(self, term_id: OntologyId) -> list[OntologyId]:
        """Closed transitive set of ancestors. Empty if the term doesn't exist."""
        t = self._by_id.get(term_id)
        return list(t.ancestor_ids) if t else []

    def id_to_parents(self, term_id: OntologyId) -> list[OntologyId]:
        """Direct-parent ids only. Empty if the term doesn't exist or is a root."""
        t = self._by_id.get(term_id)
        return list(t.parent_ids) if t else []

    def id_to_descendants(self, term_id: OntologyId) -> list[OntologyId]:
        """Closed transitive set of descendants. Empty if the term doesn't exist."""
        return sorted(self._descendants.get(term_id, ()))

    def is_subclass_of(self, child_id: OntologyId, ancestor_id: OntologyId) -> bool:
        if child_id == ancestor_id:
            return True
        t = self._by_id.get(child_id)
        return t is not None and ancestor_id in t.ancestor_ids

    def search(self, query: str, *, limit: int = 25) -> list[OntologyId]:
        """Substring search over labels + exact synonyms (case-insensitive).

        Cheap and deterministic — it's the prefilter for the dense SapBERT
        fallback. Sorted: shortest match first, then alphabetical id.
        """
        q = query.strip().lower()
        if not q:
            return []
        hits: set[OntologyId] = set()
        for name, ids in self._name_to_ids.items():
            if q in name:
                hits.update(ids)
                if len(hits) >= limit * 4:  # gather some headroom for sorting
                    break
        ranked = sorted(
            hits,
            key=lambda tid: (len(self._by_id[tid].label), self._by_id[tid].id),
        )
        return ranked[:limit]
