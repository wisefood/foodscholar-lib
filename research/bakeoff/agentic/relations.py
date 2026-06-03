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


# Ontologies FoodOn cross-references that are useful as naming/judgment context.
# Targets in these namespaces are kept in the lens; everything else is dropped.
# These are LENS-ONLY: they never become tree nodes or membership edges.
DEFAULT_TARGET_PREFIXES = ("FOODON:", "CHEBI:", "NCBITaxon:", "PATO:", "UBERON:", "PO:")


def load_relation_index(
    owl_path: str | Path,
    *,
    source_prefix: str = "FOODON:",
    target_prefixes: tuple[str, ...] = DEFAULT_TARGET_PREFIXES,
) -> dict[str, list[Relation]]:
    """Map each FOODON term id -> its non-is-a relations.

    Sources are FoodOn terms; targets are kept when in `target_prefixes` — FoodOn
    plus the ontologies it references (CHEBI, NCBITaxon, PATO, …). Cross-ontology
    targets are lens context only; they never enter the tree as nodes."""
    import pronto

    ont = pronto.Ontology(str(owl_path), import_depth=0)
    index: dict[str, list[Relation]] = {}
    for term in ont.terms():
        if term.id is None or not term.id.startswith(source_prefix):
            continue
        rels: list[Relation] = []
        for rel, targets in (getattr(term, "relationships", None) or {}).items():
            rel_name = rel.name or rel.id
            for target in targets:
                if target.id is None or not target.id.startswith(target_prefixes):
                    continue
                rels.append(Relation(rel.id, rel_name, target.id))
        if rels:
            index[term.id] = rels
    return index
