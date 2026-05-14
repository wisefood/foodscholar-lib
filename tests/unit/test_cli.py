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


def test_cli_phase_command_is_deferred(tmp_path: Path) -> None:
    cfg = _write_memory_config(tmp_path)
    result = runner.invoke(app, ["build-layer-a", "--config", str(cfg)])
    assert result.exit_code == 1
    assert "not implemented yet" in result.output


def test_cli_version_command() -> None:
    from foodscholar import __version__

    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output
