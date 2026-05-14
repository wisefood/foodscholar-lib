"""Tests for fs.ontology — lazy loading, attach, info."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from foodscholar import FoodScholar, FoodScholarConfig
from foodscholar.io.ontology import OntologyTerm
from foodscholar.ontology import FoodOnAPI

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MINI_FOODON = FIXTURES / "mini_foodon.obo"


def _config_with_ontology(tmp_path: Path) -> FoodScholarConfig:
    src = tmp_path / "fixture.obo"
    shutil.copy(MINI_FOODON, src)
    return FoodScholarConfig.model_validate(
        {
            "corpus": {"chunks_path": "data/chunks.parquet"},
            "ontology": {
                "foodon_path": str(src),
                "cache_path": str(tmp_path / "fixture.parquet"),
                "include_imports": False,
            },
            "storage": {
                "chunk_store": {"backend": "memory"},
                "graph_store": {"backend": "memory"},
            },
        }
    )


def test_in_memory_has_no_ontology_until_attached() -> None:
    fs = FoodScholar.in_memory()
    info = fs.info()
    assert info["ontology"] == "none"
    with pytest.raises(RuntimeError, match="no ontology section"):
        _ = fs.ontology


def test_attach_ontology_skips_loader() -> None:
    fs = FoodScholar.in_memory()
    api = FoodOnAPI(
        [OntologyTerm(id="X:1", label="thing")]
    )
    fs.attach_ontology(api)
    assert fs.ontology is api
    assert fs.info()["ontology"] == "loaded"


def test_lazy_load_from_config(tmp_path: Path) -> None:
    cfg = _config_with_ontology(tmp_path)
    fs = FoodScholar.from_config(cfg)
    # Lazy: not loaded until accessed
    assert fs.info()["ontology"] == "configured"
    api = fs.ontology
    assert isinstance(api, FoodOnAPI)
    assert api.name_to_id("apple") == "TEST:0000006"
    assert fs.info()["ontology"] == "loaded"


def test_load_ontology_eager(tmp_path: Path) -> None:
    cfg = _config_with_ontology(tmp_path)
    fs = FoodScholar.from_config(cfg)
    api = fs.load_ontology()
    assert isinstance(api, FoodOnAPI)
    assert len(api) > 0


def test_load_ontology_refresh(tmp_path: Path) -> None:
    cfg = _config_with_ontology(tmp_path)
    fs = FoodScholar.from_config(cfg)
    a = fs.load_ontology()
    b = fs.load_ontology(refresh=True)
    assert a is not b  # forced reload returned a new instance
    assert a.name_to_id("apple") == b.name_to_id("apple")


def test_lazy_load_caches_per_facade(tmp_path: Path) -> None:
    cfg = _config_with_ontology(tmp_path)
    fs = FoodScholar.from_config(cfg)
    a = fs.ontology
    b = fs.ontology
    assert a is b  # cached
