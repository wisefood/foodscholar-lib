"""Unit tests for the content-addressed annotation cache."""

from __future__ import annotations

import pytest

from foodscholar.annotate.cache import SCHEMA_VERSION, AnnotationCache, cache_key
from foodscholar.io.chunk import EntityLink, Mention


def _link(text: str = "olive oil", oid: str = "FOODON:03301826") -> EntityLink:
    m = Mention(text=text, start=0, end=len(text), score=1.0, ner_model_version="v0")
    return EntityLink(
        mention=m,
        ontology_id=oid,
        confidence=0.91,
        method="lexical_exact",
        linker_version="v0",
    )


# -- cache_key -------------------------------------------------------------


def test_cache_key_is_deterministic() -> None:
    args = dict(agent_model_id="m1", prompt_version="p1", ontology_hash="o1")
    assert cache_key("some text", **args) == cache_key("some text", **args)


def test_cache_key_changes_with_each_input() -> None:
    base = dict(
        chunk_text="text",
        agent_model_id="m1",
        prompt_version="p1",
        ontology_hash="o1",
    )
    k0 = cache_key(**base)
    assert cache_key(**{**base, "chunk_text": "other"}) != k0
    assert cache_key(**{**base, "agent_model_id": "m2"}) != k0
    assert cache_key(**{**base, "prompt_version": "p2"}) != k0
    assert cache_key(**{**base, "ontology_hash": "o2"}) != k0


def test_cache_key_has_no_concatenation_collision() -> None:
    # Length-prefixing means "ab"+"c" and "a"+"bc" must not collide.
    k1 = cache_key("ab", agent_model_id="c", prompt_version="p", ontology_hash="o")
    k2 = cache_key("a", agent_model_id="bc", prompt_version="p", ontology_hash="o")
    assert k1 != k2


# -- get / put -------------------------------------------------------------


def test_get_miss_returns_none() -> None:
    with AnnotationCache() as cache:
        assert cache.get("nonexistent") is None


def test_put_then_get_round_trips() -> None:
    with AnnotationCache() as cache:
        links = [_link("olive oil"), _link("apple", "FOODON:00002473")]
        cache.put("k1", "chunk-1", links)
        got = cache.get("k1")
        assert got is not None
        assert [link.ontology_id for link in got] == [link.ontology_id for link in links]
        assert got[0].mention.text == "olive oil"


def test_put_empty_links_round_trips() -> None:
    with AnnotationCache() as cache:
        cache.put("k-empty", "chunk-2", [])
        got = cache.get("k-empty")
        assert got == []  # a real hit, distinct from a miss (None)
        assert "k-empty" in cache


def test_put_is_idempotent_on_key() -> None:
    with AnnotationCache() as cache:
        cache.put("k1", "chunk-1", [_link("olive oil")])
        cache.put("k1", "chunk-1", [_link("apple", "FOODON:00002473")])
        got = cache.get("k1")
        assert got is not None
        assert len(got) == 1
        assert got[0].ontology_id == "FOODON:00002473"  # the second write wins
        assert len(cache) == 1


def test_contains_and_len() -> None:
    with AnnotationCache() as cache:
        assert len(cache) == 0
        assert "k1" not in cache
        cache.put("k1", "c1", [_link()])
        cache.put("k2", "c2", [_link()])
        assert len(cache) == 2
        assert "k1" in cache and "k2" in cache


# -- persistence -----------------------------------------------------------


def test_cache_persists_across_reopen(tmp_path) -> None:
    db = tmp_path / "nested" / "annotations.db"  # nested dir is created
    cache = AnnotationCache(db)
    cache.put("k1", "chunk-1", [_link("olive oil")])
    cache.close()

    reopened = AnnotationCache(db)
    got = reopened.get("k1")
    assert got is not None
    assert got[0].ontology_id == "FOODON:03301826"
    reopened.close()


def test_stale_schema_entry_is_treated_as_miss(tmp_path) -> None:
    db = tmp_path / "annotations.db"
    cache = AnnotationCache(db)
    cache.put("k1", "chunk-1", [_link()])
    # Simulate an entry written by an older schema version.
    cache._conn.execute(
        "UPDATE annotations SET schema_version = ? WHERE key = ?",
        (SCHEMA_VERSION - 1, "k1"),
    )
    cache._conn.commit()
    assert cache.get("k1") is None  # recomputed, not mis-deserialized
    cache.close()


def test_replay_is_a_pure_cache_hit() -> None:
    """The §4 idempotency contract: same key -> identical links, no recompute."""
    with AnnotationCache() as cache:
        key = cache_key(
            "Olive oil is heart-healthy.",
            agent_model_id="agent-v1",
            prompt_version="v1",
            ontology_hash="foodon-2024",
        )
        original = [_link("olive oil")]
        cache.put(key, "chunk-1", original)
        replay = cache.get(key)
        assert replay is not None
        assert replay[0].model_dump() == original[0].model_dump()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
