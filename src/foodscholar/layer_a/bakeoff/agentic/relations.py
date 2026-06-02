"""Throwaway non-is-a relation index loaded straight from the FoodOn OWL.

The production ontology loader keeps is-a only; the agentic method needs FoodOn's
object-property relations (derives_from, member_of, has_ingredient, …) to bridge
gaps the is-a graph can't. This loads them once via pronto. Only FOODON→FOODON
edges are kept (targets like NCBITaxon/PATO are dropped). Prototype-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Relation:
    rel_id: str       # e.g. "RO:0001000"
    rel_name: str     # e.g. "derives from"
    target_id: str    # a FOODON id


def load_relation_index(
    owl_path: str | Path, *, keep_prefix: str = "FOODON:"
) -> dict[str, list[Relation]]:
    """Map each FOODON term id -> its non-is-a relations to other FOODON terms."""
    import pronto

    ont = pronto.Ontology(str(owl_path), import_depth=0)
    index: dict[str, list[Relation]] = {}
    for term in ont.terms():
        if term.id is None or not term.id.startswith(keep_prefix):
            continue
        rels: list[Relation] = []
        for rel, targets in (getattr(term, "relationships", None) or {}).items():
            rel_name = rel.name or rel.id
            for target in targets:
                if target.id is None or not target.id.startswith(keep_prefix):
                    continue
                rels.append(Relation(rel.id, rel_name, target.id))
        if rels:
            index[term.id] = rels
    return index
