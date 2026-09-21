"""The `EntityType` vocabulary must cover both NER backends without collapsing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import pytest

from foodscholar.config import _GLINER_DEFAULT_LABELS, GLiner2Config
from foodscholar.io.chunk import GLINER2_TO_ENTITY_TYPE, EntityType
from foodscholar.layer_a.facet import ENTITY_TYPE_TO_FACET, facet_for_entity_type

VALID = frozenset(get_args(EntityType))

# Deliberately unmapped: they describe the study, not its subject — matching
# the pre-existing policy for Country / Measurement / Population.
UNFACETED = {"population", "life stage", "exercise", "measurement", "time expression", "country"}


def test_original_gliner_labels_are_still_members():
    """Renaming one would orphan every stored chunk carrying it."""
    for label in _GLINER_DEFAULT_LABELS:
        assert label in VALID


def test_every_gliner2_label_resolves_to_a_valid_entity_type():
    for label in GLiner2Config().labels:
        assert GLINER2_TO_ENTITY_TYPE.get(label, label) in VALID, label


def test_gliner2_labels_do_not_collapse_wholesale_to_other():
    """Mapping 23 of 27 to `other` would discard the point of the upgrade."""
    resolved = {GLINER2_TO_ENTITY_TYPE.get(x, x) for x in GLiner2Config().labels}
    assert "other" not in resolved
    assert len(resolved) >= 25


def test_alias_targets_are_valid_members():
    for target in GLINER2_TO_ENTITY_TYPE.values():
        assert target in VALID


def test_disease_folds_onto_the_established_member():
    assert GLINER2_TO_ENTITY_TYPE["disease"] == "medical condition"


def test_case_only_differences_fold_rather_than_duplicate():
    for lower, established in [
        ("country", "Country"),
        ("population", "Population"),
        ("measurement", "Measurement"),
        ("time expression", "Time expression"),
    ]:
        assert GLINER2_TO_ENTITY_TYPE[lower] == established
        assert lower not in VALID, f"{lower!r} must not become a second member"


@pytest.mark.parametrize(
    ("label", "facet"),
    [
        ("vitamin", "nutrients"),
        ("mineral", "nutrients"),
        ("amino acid", "nutrients"),
        ("lipid", "nutrients"),
        ("food additive", "foods"),
        ("disease", "health"),
        ("symptom", "health"),
        ("physiological process", "health"),
    ],
)
def test_new_types_route_to_a_facet(label, facet):
    assert facet_for_entity_type(GLINER2_TO_ENTITY_TYPE.get(label, label)) == facet


def test_every_gliner2_label_is_either_faceted_or_deliberately_not():
    for label in GLiner2Config().labels:
        resolved = GLINER2_TO_ENTITY_TYPE.get(label, label)
        if facet_for_entity_type(resolved) is None:
            assert label in UNFACETED, f"{label!r} routes nowhere and is not declared"


def test_facet_map_only_references_valid_types():
    for entity_type in ENTITY_TYPE_TO_FACET:
        # "allergen" predates the literal and is kept for forward compatibility.
        assert entity_type in VALID or entity_type == "allergen"


def test_default_labels_match_the_benchmark_fixture_verbatim():
    """Editing the descriptions invalidates the published numbers."""
    source = Path("kggen/graph_code/ner-nel/run_ner_nel_corpus_gliner2_sapbert.py")
    if not source.exists():
        pytest.skip("kggen reference not present in this checkout")
    text = source.read_text()
    start = text.index("GLINER2_CUSTOM_LABELS = {")
    block = text[start : text.index("\n}", start)]
    reference = dict(re.findall(r'"((?:[^"\\]|\\.)*)":\s*"((?:[^"\\]|\\.)*)"', block))
    assert GLiner2Config().labels == reference
