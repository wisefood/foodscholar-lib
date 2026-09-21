"""The `Relation` contract: id determinism, NIL endpoints, immutability."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from foodscholar.io.relation import (
    Relation,
    is_nil,
    make_relation_id,
    nil_id,
)


def test_relation_id_is_content_addressed_and_stable():
    a = make_relation_id("FOODON:1", "reduces", "CHEBI:2")
    b = make_relation_id("FOODON:1", "reduces", "CHEBI:2")
    assert a == b == make_relation_id("FOODON:1", "reduces", "CHEBI:2")
    assert a.startswith("rel:")


def test_relation_id_distinguishes_direction_and_predicate():
    base = make_relation_id("A", "p", "B")
    assert make_relation_id("B", "p", "A") != base
    assert make_relation_id("A", "q", "B") != base


def test_relation_id_cannot_collide_via_separator_injection():
    """A naive concat would make ('A:p', 'x', 'B') collide with ('A', 'p:x', 'B')."""
    assert make_relation_id("A\x1fp", "x", "B") != make_relation_id("A", "p\x1fx", "B")


def test_nil_id_slugs_and_round_trips():
    assert nil_id("LDL cholesterol!!") == "NIL:ldl-cholesterol"
    assert is_nil(nil_id("anything"))
    assert not is_nil("FOODON:123")


def test_nil_id_of_empty_surface_is_still_valid():
    assert nil_id("   ") == "NIL:unknown"


def _relation(**overrides) -> Relation:
    base = {
        "relation_id": make_relation_id("FOODON:1", "reduces", "CHEBI:2"),
        "subject_id": "FOODON:1",
        "predicate": "reduces",
        "object_id": "CHEBI:2",
    }
    return Relation(**{**base, **overrides})


def test_is_fully_grounded_reflects_both_flags():
    assert _relation().is_fully_grounded
    assert not _relation(object_linked=False).is_fully_grounded
    assert not _relation(subject_linked=False).is_fully_grounded


def test_triple_property():
    assert _relation().triple == ("FOODON:1", "reduces", "CHEBI:2")


def test_relation_is_frozen():
    with pytest.raises(ValidationError):
        _relation().predicate = "other"


def test_relation_forbids_extra_fields():
    with pytest.raises(ValidationError):
        _relation(unexpected="x")
