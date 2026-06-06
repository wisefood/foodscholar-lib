"""CLI exposes build-layer-c (with --dry-run) and bench-layer-c."""

from __future__ import annotations

from typer.testing import CliRunner

from foodscholar.cli.main import app

runner = CliRunner()


def test_bench_layer_c_in_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "bench-layer-c" in result.output


def test_build_layer_c_has_dry_run_flag() -> None:
    result = runner.invoke(app, ["build-layer-c", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
    assert "--facet" in result.output
