"""Layer C internal models."""

from __future__ import annotations

from foodscholar.layer_c.models import LayerCReport, MethodResult, Stage1Output


def test_stage1_output_fields() -> None:
    o = Stage1Output(text="x", n_input_chunks=3, n_input_chars=10,
                     strategy="single", n_groups=1)
    assert o.strategy == "single"
    assert o.n_groups == 1


def test_method_result_roundtrip() -> None:
    r = MethodResult(method="lexrank", summary="s", input_chunks=243,
                     input_chars=184532, execution_time_ms=412,
                     summary_length_chars=1840)
    d = r.model_dump()
    assert d["method"] == "lexrank"
    assert d["input_chunks"] == 243
    assert d["summary_length_chars"] == 1840


def test_layer_c_report_defaults() -> None:
    rep = LayerCReport(n_themes=5, n_cards=4, n_skipped=1, n_failed=0)
    assert rep.strategy_counts == {}
    assert "5" in str(rep)  # __str__ mentions theme count
