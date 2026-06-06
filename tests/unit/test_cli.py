from pathlib import Path

from typer.testing import CliRunner

from foodscholar.cli.main import app

runner = CliRunner()


def _write_memory_config(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        "corpus:\n"
        "  chunks_path: data/chunks.parquet\n"
        "storage:\n"
        "  chunk_store:\n"
        "    backend: memory\n"
        "  graph_store:\n"
        "    backend: memory\n"
    )
    return p


def test_cli_info(tmp_path: Path) -> None:
    cfg = _write_memory_config(tmp_path)
    result = runner.invoke(app, ["info", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "foodscholar" in result.output
    assert "config_hash" in result.output


def test_cli_init_memory_backend_is_noop(tmp_path: Path) -> None:
    cfg = _write_memory_config(tmp_path)
    result = runner.invoke(app, ["init", "--config", str(cfg)])
    assert result.exit_code == 0, result.output


def test_cli_build_layer_c_runs_on_empty_stores(tmp_path: Path) -> None:
    # build-layer-c shipped in Layer C: it now runs (no themes → 0 cards),
    # rather than raising the old deferred-phase error.
    cfg = _write_memory_config(tmp_path)
    result = runner.invoke(app, ["build-layer-c", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "Layer C" in result.output


def test_cli_report_layer_b_runs_on_empty_stores(tmp_path: Path) -> None:
    """report-layer-b is read-only and works against an empty in-memory graph."""
    cfg = _write_memory_config(tmp_path)
    result = runner.invoke(app, ["report-layer-b", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "quality report" in result.output.lower()


def test_cli_sweep_layer_b_runs_on_empty_stores(tmp_path: Path) -> None:
    """sweep-layer-b with no shelves produces an (empty) ranked table without
    error — every config dry-run yields zero themes."""
    cfg = _write_memory_config(tmp_path)
    result = runner.invoke(app, ["sweep-layer-b", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "sweep" in result.output.lower()


def test_cli_version_command() -> None:
    from foodscholar import __version__

    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output
