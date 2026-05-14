"""Tests for the FoodOn loader + FoodOnAPI lookups."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from foodscholar.io.ontology import OntologyTerm
from foodscholar.ontology import FoodOnAPI, load_ontology

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MINI_FOODON = FIXTURES / "mini_foodon.obo"


@pytest.fixture
def terms() -> list[OntologyTerm]:
    return load_ontology(MINI_FOODON)


@pytest.fixture
def api(terms: list[OntologyTerm]) -> FoodOnAPI:
    return FoodOnAPI(terms)


# ---------------------------------------------------------------- loader


def test_load_terms_count(terms: list[OntologyTerm]) -> None:
    assert len(terms) == 11  # 10 active + 1 obsolete


def test_load_term_fields(terms: list[OntologyTerm]) -> None:
    apple = next(t for t in terms if t.id == "TEST:0000006")
    assert apple.label == "apple"
    assert "Malus domestica" in apple.synonyms
    assert "eating apple" in apple.related_synonyms
    assert apple.parent_ids == ("TEST:0000004",)
    assert set(apple.ancestor_ids) == {"TEST:0000001", "TEST:0000002", "TEST:0000004"}
    assert apple.obsolete is False


def test_obsolete_loaded_but_marked(terms: list[OntologyTerm]) -> None:
    legacy = next(t for t in terms if t.id == "TEST:0000010")
    assert legacy.obsolete is True


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ontology(tmp_path / "nope.obo")


# ---------------------------------------------------------------- cache


def test_cache_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "fixture.obo"
    shutil.copy(MINI_FOODON, src)
    cache = tmp_path / "fixture.parquet"

    first = load_ontology(src, cache_path=cache)
    assert cache.exists()
    assert (cache.parent / "fixture.parquet.meta.json").exists()

    second = load_ontology(src, cache_path=cache)
    # Same data, sourced from cache the second time
    by_id_a = {t.id: t for t in first}
    by_id_b = {t.id: t for t in second}
    assert by_id_a == by_id_b


def test_cache_invalidates_on_source_change(tmp_path: Path) -> None:
    import time

    src = tmp_path / "fixture.obo"
    shutil.copy(MINI_FOODON, src)
    cache = tmp_path / "fixture.parquet"

    load_ontology(src, cache_path=cache)
    first_count = len(load_ontology(src, cache_path=cache))

    time.sleep(0.01)
    src.write_text(src.read_text() + "\n[Term]\nid: TEST:0000099\nname: extra\n")
    second_count = len(load_ontology(src, cache_path=cache))
    assert second_count == first_count + 1


# ---------------------------------------------------------------- API surface


def test_name_to_id_label(api: FoodOnAPI) -> None:
    assert api.name_to_id("apple") == "TEST:0000006"
    assert api.name_to_id("Apple") == "TEST:0000006"
    assert api.name_to_id("  APPLE  ") == "TEST:0000006"


def test_name_to_id_synonyms(api: FoodOnAPI) -> None:
    assert api.name_to_id("Malus domestica") == "TEST:0000006"
    assert api.name_to_id("EVOO") == "TEST:0000008"
    assert api.name_to_id("extra-virgin olive oil") == "TEST:0000008"


def test_name_to_id_obsolete_returns_none(api: FoodOnAPI) -> None:
    assert api.name_to_id("legacy term") is None


def test_name_to_id_unknown_returns_none(api: FoodOnAPI) -> None:
    assert api.name_to_id("quinoa") is None


def test_id_to_label(api: FoodOnAPI) -> None:
    assert api.id_to_label("TEST:0000008") == "olive oil"
    assert api.id_to_label("nope") is None


def test_id_to_synonyms_excludes_related_by_default(api: FoodOnAPI) -> None:
    syns = api.id_to_synonyms("TEST:0000006")
    assert syns == ["Malus domestica"]


def test_id_to_synonyms_can_include_related(api: FoodOnAPI) -> None:
    syns = api.id_to_synonyms("TEST:0000006", include_related=True)
    assert set(syns) == {"Malus domestica", "eating apple"}


def test_id_to_ancestors(api: FoodOnAPI) -> None:
    # olive oil -> olive -> fruit -> plant food -> food product
    ancestors = set(api.id_to_ancestors("TEST:0000008"))
    assert ancestors == {"TEST:0000007", "TEST:0000004", "TEST:0000002", "TEST:0000001"}


def test_id_to_parents_direct_only(api: FoodOnAPI) -> None:
    assert api.id_to_parents("TEST:0000008") == ["TEST:0000007"]
    assert api.id_to_parents("TEST:0000001") == []  # root


def test_id_to_descendants(api: FoodOnAPI) -> None:
    descendants = api.id_to_descendants("TEST:0000004")  # fruit
    assert set(descendants) == {"TEST:0000006", "TEST:0000007", "TEST:0000008"}


def test_is_subclass_of(api: FoodOnAPI) -> None:
    assert api.is_subclass_of("TEST:0000008", "TEST:0000001")  # olive oil ⊑ food product
    assert api.is_subclass_of("TEST:0000008", "TEST:0000008")  # reflexive
    assert not api.is_subclass_of("TEST:0000008", "TEST:0000003")  # not animal food
    assert not api.is_subclass_of("nope", "TEST:0000001")


def test_search_substring(api: FoodOnAPI) -> None:
    hits = api.search("olive")
    assert "TEST:0000007" in hits  # olive
    assert "TEST:0000008" in hits  # olive oil


def test_search_returns_shortest_first(api: FoodOnAPI) -> None:
    hits = api.search("olive")
    # "olive" (5 chars) should come before "olive oil" (9 chars)
    olive_idx = hits.index("TEST:0000007")
    olive_oil_idx = hits.index("TEST:0000008")
    assert olive_idx < olive_oil_idx


def test_search_empty_query(api: FoodOnAPI) -> None:
    assert api.search("") == []


def test_search_unknown(api: FoodOnAPI) -> None:
    assert api.search("zzz_no_such_term") == []


def test_membership_and_iteration(api: FoodOnAPI) -> None:
    assert "TEST:0000006" in api
    assert "nope" not in api
    assert len(api) == 11
    ids = {t.id for t in api}
    assert "TEST:0000006" in ids
