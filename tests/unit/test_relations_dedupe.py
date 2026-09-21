"""Dedup — provenance unions, determinism, graceful degradation."""

from __future__ import annotations

import subprocess
import sys
import textwrap

from foodscholar.config import RelationsConfig
from foodscholar.relations.aggregate import build_relations, dedupe_graph
from foodscholar.relations.dedupe import DedupeResult, deduplicate_surfaces
from foodscholar.relations.extracted import ExtractedGraph
from foodscholar.relations.ground import GroundingResult


def test_normalizes_case_and_whitespace():
    result = deduplicate_surfaces(["olive oil", "Olive Oil", "olive   oil"])
    assert len({result.resolve(s) for s in ["olive oil", "Olive Oil", "olive   oil"]}) == 1


def test_unrelated_surfaces_are_not_merged():
    result = deduplicate_surfaces(["olive oil", "LDL cholesterol"])
    assert result.resolve("olive oil") != result.resolve("LDL cholesterol")


def test_empty_input():
    assert deduplicate_surfaces([]).canonical == {}


def test_resolve_passes_through_unknown_surfaces():
    assert deduplicate_surfaces(["a b"]).resolve("never seen") == "never seen"


def test_provenance_unions_when_relations_collapse():
    """The property the whole pipeline exists to preserve."""
    graph = ExtractedGraph()
    graph.add_triple(("olive oil", "reduces", "LDL"), "c1")
    graph.add_triple(("Olive Oil", "lowers", "LDL"), "c2")

    entity_dedupe = DedupeResult(
        {"olive oil": "olive oil", "Olive Oil": "olive oil", "LDL": "LDL"}, "test"
    )
    predicate_dedupe = DedupeResult({"reduces": "reduces", "lowers": "reduces"}, "test")
    grounding = GroundingResult(
        ids={"olive oil": "FOODON:1", "LDL": "CHEBI:2"},
        linked={"olive oil", "LDL"},
    )

    relations = build_relations(
        graph,
        grounding=grounding,
        entity_dedupe=entity_dedupe,
        predicate_dedupe=predicate_dedupe,
        extractor_version="test",
    )
    assert len(relations) == 1
    relation = relations[0]
    assert set(relation.chunk_ids) == {"c1", "c2"}
    assert relation.chunk_count == 2
    assert relation.mention_count == 2
    assert set(relation.subject_surfaces) == {"olive oil", "Olive Oil"}
    assert set(relation.predicate_surfaces) == {"reduces", "lowers"}


def test_relations_grounding_to_the_same_id_are_dropped_as_self_loops():
    graph = ExtractedGraph()
    graph.add_triple(("olive oil", "same as", "olive-oil"), "c1")
    dedupe = DedupeResult({"olive oil": "olive oil", "olive-oil": "olive-oil"}, "t")
    grounding = GroundingResult(
        ids={"olive oil": "FOODON:1", "olive-oil": "FOODON:1"},
        linked={"olive oil", "olive-oil"},
    )
    assert (
        build_relations(
            graph,
            grounding=grounding,
            entity_dedupe=dedupe,
            predicate_dedupe=DedupeResult({"same as": "same as"}, "t"),
            extractor_version="t",
        )
        == []
    )


def test_keep_nil_false_drops_partially_grounded_relations():
    graph = ExtractedGraph()
    graph.add_triple(("olive oil", "reduces", "HbA1c"), "c1")
    dedupe = DedupeResult({"olive oil": "olive oil", "HbA1c": "HbA1c"}, "t")
    grounding = GroundingResult(
        ids={"olive oil": "FOODON:1", "HbA1c": "NIL:hba1c"}, linked={"olive oil"}
    )
    kwargs = dict(
        grounding=grounding,
        entity_dedupe=dedupe,
        predicate_dedupe=DedupeResult({"reduces": "reduces"}, "t"),
        extractor_version="t",
    )
    assert build_relations(graph, keep_nil=True, **kwargs)
    assert build_relations(graph, keep_nil=False, **kwargs) == []


def test_dedupe_disabled_is_identity():
    graph = ExtractedGraph()
    graph.add_triple(("a b", "p", "c d"), "c1")
    cfg = RelationsConfig()
    cfg.dedupe.enabled = False
    entities, predicates = dedupe_graph(graph, cfg)
    assert entities.version == "off"
    assert predicates.resolve("p") == "p"


def test_result_is_identical_across_python_hash_seeds():
    """Guards the upstream bug: iterating a `set` made the canonical pick
    depend on PYTHONHASHSEED, which would pass in-process and fail in CI."""
    script = textwrap.dedent(
        """
        from foodscholar.relations.dedupe import deduplicate_surfaces
        surfaces = ["vitamins", "vitamin", "Vitamin", "minerals", "mineral",
                    "olive oil", "Olive Oil", "whole grains", "whole grain"]
        r = deduplicate_surfaces(surfaces)
        print("RESULT " + "|".join(f"{k}={r.canonical[k]}" for k in sorted(r.canonical)))
        """
    )
    outputs = set()
    for seed in ("0", "1", "424242"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            check=True,
        )
        # The library logs to stdout, so pick the marked line rather than
        # comparing timestamps.
        line = next(
            ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")
        )
        outputs.add(line)
    assert len(outputs) == 1, f"nondeterministic across hash seeds: {outputs}"
