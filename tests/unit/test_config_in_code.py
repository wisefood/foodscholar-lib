"""In-code configuration tests.

Covers the three accepted shapes for `FoodScholar.from_config` /
`resolve_config`: YAML path, plain dict, and an already-validated
`FoodScholarConfig` instance — all without a sister YAML file on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foodscholar import FoodScholar, FoodScholarConfig
from foodscholar.config import resolve_config


def _memory_dict() -> dict:
    return {
        "corpus": {"chunks_path": "data/chunks.parquet"},
        "storage": {
            "chunk_store": {"backend": "memory"},
            "graph_store": {"backend": "memory"},
        },
    }


def test_resolve_config_accepts_dict() -> None:
    cfg = resolve_config(_memory_dict())
    assert isinstance(cfg, FoodScholarConfig)
    assert cfg.storage.chunk_store.backend == "memory"


def test_resolve_config_accepts_pydantic_instance_unchanged() -> None:
    cfg = FoodScholarConfig.model_validate(_memory_dict())
    out = resolve_config(cfg)
    assert out is cfg


def test_resolve_config_accepts_yaml_path(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "corpus:\n"
        "  chunks_path: data/chunks.parquet\n"
        "storage:\n"
        "  chunk_store:\n"
        "    backend: memory\n"
        "  graph_store:\n"
        "    backend: memory\n"
    )
    cfg = resolve_config(p)
    assert cfg.corpus.chunks_path == Path("data/chunks.parquet")


def test_resolve_config_rejects_unknown_type() -> None:
    with pytest.raises(TypeError, match="unsupported config type"):
        resolve_config(42)  # type: ignore[arg-type]


def test_dict_env_substitution_runs() -> None:
    """${ENV} placeholders in dict values are substituted, same as YAML."""
    import os

    os.environ["FS_TEST_PW"] = "secret-123"
    cfg = resolve_config(
        {
            "corpus": {"chunks_path": "data/chunks.parquet"},
            "storage": {
                "chunk_store": {"backend": "memory"},
                "graph_store": {
                    "backend": "neo4j",
                    "url": "bolt://localhost:7687",
                    "user": "neo4j",
                    "password": "${FS_TEST_PW}",
                },
            },
        }
    )
    assert cfg.storage.graph_store.password == "secret-123"


def test_in_memory_factory_with_dict_config() -> None:
    fs = FoodScholar.in_memory(
        config={
            "corpus": {"chunks_path": "data/chunks.parquet"},
            "annotate": {"batch_size": 32},
            "storage": {
                "chunk_store": {"backend": "memory"},
                "graph_store": {"backend": "memory"},
            },
        }
    )
    assert fs.config.annotate.batch_size == 32


def test_from_config_with_dict_no_yaml_file_needed() -> None:
    fs = FoodScholar.from_config(_memory_dict())
    info = fs.info()
    assert info["chunk_store"] == "memory"
    assert info["ner"] == "gliner"
    assert info["nel_backend"] == "hnsw"


def test_load_and_annotate_skips_existing_snapshot(tmp_path: Path) -> None:
    """Idempotency: if the snapshot exists and is non-empty, the call short-circuits."""
    snapshot = tmp_path / "annotated.parquet"
    snapshot.write_bytes(b"existing")
    chunks_path = tmp_path / "chunks.parquet"
    chunks_path.write_bytes(b"")  # never read because we short-circuit

    fs = FoodScholar.in_memory(
        config={
            "corpus": {
                "chunks_path": str(chunks_path),
                "annotated_snapshot_path": str(snapshot),
            },
            "storage": {
                "chunk_store": {"backend": "memory"},
                "graph_store": {"backend": "memory"},
            },
        }
    )
    assert fs.load_and_annotate(chunks_path) is None
