"""Facade exposes build_layer_c / benchmark_layer_c that delegate to layer_c."""

from __future__ import annotations

import foodscholar.facade as facade_mod


def test_build_layer_c_delegates(monkeypatch) -> None:
    called = {}

    def fake_build(fs, *, facet="foods", dry_run=False):
        called["facet"] = facet
        called["dry_run"] = dry_run
        return "report"

    monkeypatch.setattr("foodscholar.layer_c.builder.build_layer_c", fake_build)

    # Build a bare facade instance without running __init__ machinery.
    fs = object.__new__(facade_mod.FoodScholar)
    out = facade_mod.FoodScholar.build_layer_c(fs, facet="foods", dry_run=True)
    assert out == "report"
    assert called == {"facet": "foods", "dry_run": True}


def test_benchmark_layer_c_delegates(monkeypatch) -> None:
    called = {}

    def fake_bench(fs, *, facet="foods", themes=5, out=None):
        called["themes"] = themes
        return {"t1": []}

    monkeypatch.setattr("foodscholar.layer_c.benchmark.benchmark_facet", fake_bench)
    fs = object.__new__(facade_mod.FoodScholar)
    out = facade_mod.FoodScholar.benchmark_layer_c(fs, themes=3)
    assert out == {"t1": []}
    assert called["themes"] == 3
