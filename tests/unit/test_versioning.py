from foodscholar.versioning import config_hash, make_artifact_meta, new_artifact_id


def test_config_hash_is_stable_under_key_reordering() -> None:
    a = {"a": 1, "b": [1, 2, 3], "c": {"x": True, "y": "z"}}
    b = {"c": {"y": "z", "x": True}, "b": [1, 2, 3], "a": 1}
    assert config_hash(a) == config_hash(b)


def test_config_hash_changes_on_value_change() -> None:
    a = {"a": 1}
    b = {"a": 2}
    assert config_hash(a) != config_hash(b)


def test_new_artifact_id_is_namespaced() -> None:
    aid = new_artifact_id("annotate")
    assert aid.startswith("annotate-")


def test_make_artifact_meta_populates_fields() -> None:
    meta = make_artifact_meta(phase="layer-a", config={"k": 1}, record_count=42)
    assert meta.phase == "layer-a"
    assert meta.record_count == 42
    assert len(meta.config_hash) == 16
