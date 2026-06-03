from pathlib import Path

from bakeoff.agentic.relations import Relation, load_relation_index

_FIX = Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon_relations.obo"


def test_load_relation_index_keeps_foodon_to_foodon_edges():
    idx = load_relation_index(_FIX)
    rels = idx["FOODON:0000011"]
    assert rels == [Relation(rel_id="RO:0001000", rel_name="derives from",
                             target_id="FOODON:0000010")]


def test_load_relation_index_keeps_cross_ontology_targets_by_default():
    # CHEBI/NCBITaxon/PATO targets are kept as lens context (default prefixes).
    idx = load_relation_index(_FIX)
    assert "FOODON:0000012" in idx
    assert any(r.target_id.startswith("NCBITaxon:") for r in idx["FOODON:0000012"])


def test_load_relation_index_can_restrict_to_foodon_only():
    idx = load_relation_index(_FIX, target_prefixes=("FOODON:",))
    assert "FOODON:0000012" not in idx  # NCBITaxon target dropped when restricted
