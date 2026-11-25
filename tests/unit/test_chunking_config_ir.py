"""ChunkingConfig loads alone; the IR round-trips; the old config name still works."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from foodscholar.chunking import (
    ChunkingConfig,
    DocumentIR,
    Figure,
    Heading,
    Paragraph,
    Region,
    Table,
    TableData,
)
from foodscholar.config import ChunkerConfig, FoodScholarConfig
from tests.unit.chunking_fixtures import ir


def test_old_name_is_the_same_class_and_the_full_config_still_validates():
    assert ChunkerConfig is ChunkingConfig
    cfg = FoodScholarConfig.model_validate({"corpus": {"chunks_path": "x"}, "chunking": {"max_tokens": 512}})
    assert isinstance(cfg.chunking, ChunkingConfig)
    assert cfg.chunking.profile == "legacy"


def test_standalone_yaml_with_env_substitution(tmp_path, monkeypatch):
    monkeypatch.setenv("FS_TEST_WORK", "/data/work")
    path = tmp_path / "chunking.yaml"
    path.write_text("profile: v2\nwork_dir: ${FS_TEST_WORK}\nsources:\n  guide:\n    genre: leaflet\n    assembly: {max_tokens: 256, target_tokens: 200}\n")
    cfg = ChunkingConfig.from_yaml(path)
    assert cfg.profile == "v2"
    assert str(cfg.work_dir) == "/data/work"
    assert cfg.profile_for("guide").assembly.max_tokens == 256
    assert set(cfg.sources) == {"guide"}  # an explicit mapping replaces the built-in one


def test_a_full_foodscholar_yaml_loads_only_its_chunking_section(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("corpus: {chunks_path: x}\nstorage: {anything: goes}\nchunking:\n  overlap: 32\n")
    assert ChunkingConfig.from_yaml(path).overlap == 32


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        ChunkingConfig.from_mapping({"max_tokenz": 10})


def test_unknown_source_type_names_the_known_ones():
    with pytest.raises(KeyError, match="abstract, guide, textbook"):
        ChunkingConfig().profile_for("filing")


def test_ir_round_trips_through_json(tmp_path):
    doc = ir(
        [
            Heading(id="h", text="Fats", level=1, regions=[Region(page=1)]),
            Paragraph(id="p", text="Use oil.", regions=[Region(page=1, bbox=(1, 2, 3, 4)), Region(page=2)]),
            Table(id="t", table=TableData(cells=[["a", "b"], ["1", "2"]], header_rows=1, units=[None, "g"])),
            Figure(id="f", figure_kind="chart", caption="Figure 1"),
        ]
    )
    path = doc.save(tmp_path / "doc.json")
    again = DocumentIR.load(path)
    assert again == doc
    assert again.elements[1].pages == (1, 2)
    assert json.loads(path.read_text())["elements"][3]["kind"] == "figure"


def test_ir_load_refuses_another_schema_version(tmp_path):
    path = ir([]).save(tmp_path / "doc.json")
    data = json.loads(path.read_text())
    data["schema_version"] = "0"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="IR schema"):
        DocumentIR.load(path)


def test_table_markdown_repeats_the_header_and_keeps_footnotes():
    t = TableData(cells=[["Food", "g"], ["oil", "10"], ["nuts", "30"]], header_rows=1, footnotes=["per day"])
    assert t.to_markdown([["nuts", "30"]]).splitlines() == ["| Food | g |", "| --- | --- |", "| nuts | 30 |", "[1] per day"]


def test_body_text_skips_furniture_and_includes_figure_captions():
    from foodscholar.chunking import Furniture

    doc = ir([Furniture(id="x", text="page 3"), Figure(id="f", caption="Fig 1 Plate"), Paragraph(id="p", text="Eat.")])
    assert doc.body_text() == "Fig 1 Plate\nEat."
