"""Grounding — the step that binds kggen's strings to foodscholar's ids."""

from __future__ import annotations

from foodscholar.relations.ground import ground_surfaces
from tests.unit.conftest_relations import StubLinker

TABLE = {
    "olive oil": ("FOODON:03301710", 0.95),
    "ldl cholesterol": ("CHEBI:39026", 0.80),
    "hba1c": ("CHEBI:35143", 0.55),
}


def test_links_in_a_single_batched_call():
    """Per-triple linking would be two orders of magnitude more encoder work."""
    linker = StubLinker(TABLE)
    ground_surfaces(["olive oil", "LDL cholesterol", "HbA1c"], linker=linker)
    assert linker.batch_calls == 1


def test_links_above_the_gate():
    result = ground_surfaces(["olive oil"], linker=StubLinker(TABLE), min_sim=0.70)
    assert result.id_for("olive oil") == "FOODON:03301710"
    assert result.is_linked("olive oil")


def test_below_the_gate_becomes_nil():
    result = ground_surfaces(["HbA1c"], linker=StubLinker(TABLE), min_sim=0.70)
    assert result.id_for("HbA1c") == "NIL:hba1c"
    assert not result.is_linked("HbA1c")


def test_unknown_surface_becomes_nil():
    result = ground_surfaces(["mystery thing"], linker=StubLinker(TABLE))
    assert result.id_for("mystery thing") == "NIL:mystery-thing"


def test_keep_nil_false_omits_rather_than_nils():
    result = ground_surfaces(["HbA1c"], linker=StubLinker(TABLE), keep_nil=False)
    assert "HbA1c" not in result.ids
    assert result.id_for("HbA1c") == "NIL:hba1c"  # id_for still synthesizes


def test_min_sim_is_independent_of_the_linker_threshold():
    """A stricter relations gate rejects a link the linker itself returned."""
    result = ground_surfaces(
        ["LDL cholesterol"], linker=StubLinker(TABLE), min_sim=0.90
    )
    assert not result.is_linked("LDL cholesterol")


def test_report_surfaces_the_ontology_gap():
    result = ground_surfaces(
        ["olive oil", "HbA1c", "mystery thing"], linker=StubLinker(TABLE)
    )
    report = result.report()
    assert report["n_linked"] == 1
    assert report["n_nil"] == 2
    assert "HbA1c" in report["top_nil_surfaces"]


def test_duplicate_and_blank_surfaces_are_collapsed():
    linker = StubLinker(TABLE)
    result = ground_surfaces(["olive oil", "olive oil", "", "  "], linker=linker)
    assert result.n_surfaces == 1


def test_empty_input_does_not_call_the_linker():
    linker = StubLinker(TABLE)
    assert ground_surfaces([], linker=linker).n_surfaces == 0
    assert linker.batch_calls == 0
