from pathlib import Path

from foodscholar.layer_a.bakeoff.agentic.relations import Relation, load_relation_index

_FIX = Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon_relations.obo"


def test_load_relation_index_keeps_foodon_to_foodon_edges_only():
    idx = load_relation_index(_FIX)
    rels = idx["FOODON:0000011"]
    assert rels == [Relation(rel_id="RO:0001000", rel_name="derives from",
                             target_id="FOODON:0000010")]


def test_load_relation_index_drops_non_foodon_targets():
    idx = load_relation_index(_FIX)
    assert "FOODON:0000012" not in idx
