import os
from pathlib import Path

from foodscholar.config import load_config


def test_load_example_config(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[2] / "config.example.yaml"
    os.environ["NEO4J_PASSWORD"] = "test-password"
    cfg = load_config(src)
    assert cfg.corpus.chunks_path == Path("data/chunks.parquet")
    assert cfg.layer_a.max_depth == 5
    assert cfg.storage.graph_store.password == "test-password"


def test_env_substitution_leaves_unknown_vars(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "corpus:\n"
        "  chunks_path: data/chunks.parquet\n"
        "storage:\n"
        "  graph_store:\n"
        "    backend: neo4j\n"
        "    user: ${UNSET_VAR_XYZ}\n"
    )
    cfg = load_config(p)
    assert cfg.storage.graph_store.user == "${UNSET_VAR_XYZ}"
