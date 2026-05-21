"""Tests for the post-build audit module.

Each test builds a small graph in a known state and verifies that the
corresponding audit section catches (or doesn't catch) the condition.
"""

from __future__ import annotations

from pathlib import Path

from foodscholar import FoodScholar
from foodscholar.config import FoodScholarConfig, LayerAConfig
from foodscholar.evaluation.audit import AuditReport, audit
from foodscholar.io.chunk import Chunk, EntityLink, Mention
from foodscholar.io.graph import Shelf
from foodscholar.layer_a import attach, build_layer_a
from foodscholar.ontology import FoodOnAPI, load_ontology
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore

# ---------------------------------------------------------------- fixtures


def _mini_foodon() -> FoodOnAPI:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"
    return FoodOnAPI(load_ontology(path), prefix_filter=None)


def _full_config(layer_a: LayerAConfig | None = None) -> FoodScholarConfig:
    data: dict = {
        "corpus": {"chunks_path": "data/chunks.parquet"},
        "storage": {
            "chunk_store": {"backend": "memory"},
            "graph_store": {"backend": "memory"},
        },
    }
    cfg = FoodScholarConfig.model_validate(data)
    if layer_a is not None:
        cfg = cfg.model_copy(update={"layer_a": layer_a})
    return cfg


def _chunk(
    chunk_id: str,
    foodon_ids: list[str] | None = None,
    shelf_ids: list[str] | None = None,
    *,
    entity_links: list[EntityLink] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="t",
        source_doc_id="d",
        source_type="abstract",
        section_type="abstract",
        foodon_ids=foodon_ids or [],
        shelf_ids=shelf_ids or [],
        entity_links=entity_links or [],
    )


def _link(
    ontology_id: str,
    *,
    confidence: float = 1.0,
    entity_type: str = "food",
    text: str = "x",
) -> EntityLink:
    return EntityLink(
        mention=Mention(
            text=text,
            start=0,
            end=1,
            score=confidence,
            ner_model_version="t",
            entity_type=entity_type,  # type: ignore[arg-type]
        ),
        ontology_id=ontology_id,
        confidence=confidence,
        method="dense",
        linker_version="t",
    )


def _build_real_graph() -> tuple[InMemoryChunkStore, InMemoryGraphStore, FoodOnAPI]:
    """End-to-end build: ingest chunks, build_layer_a, build_entities, attach."""
    chunk_store = InMemoryChunkStore()
    chunks = [
        _chunk("c1", foodon_ids=["TEST:0000008"],
               entity_links=[_link("TEST:0000008", text="olive oil")]),
        _chunk("c2", foodon_ids=["TEST:0000006"],
               entity_links=[_link("TEST:0000006", text="apple")]),
        _chunk("c3", foodon_ids=["TEST:0000008", "TEST:9999999"],
               entity_links=[_link("TEST:0000008", text="olive oil")]),
    ]
    chunk_store.upsert(chunks)
    ontology = _mini_foodon()
    graph_store = InMemoryGraphStore()
    cfg = LayerAConfig(
        min_support=2,
        max_depth=5,
        collapse_single_child_chains=False,
        facets=["foods"],
        umbrella_direct_share_max=0.0,
    )
    full_cfg = _full_config(cfg)
    build_layer_a(chunk_store, graph_store, ontology, config=cfg, full_config=full_cfg)

    # Mimic fs.build_entities's [:MENTIONS] writes — audit needs them to
    # compare against foodon_ids on chunks. Strip the "FOODON" check since
    # mini-fixture uses TEST: prefix, so route to a TEST-prefixed entity_chunks
    # entry. To keep tests aligned with audit's FOODON-only filter, we use
    # the FOODON prefix directly for the cross-store test.
    for chunk in chunk_store.scan():
        for link in chunk.entity_links:
            # use TEST as a placeholder; the cross-store test patches this
            graph_store.attach_chunks_to_entity(
                link.ontology_id, [(chunk.chunk_id, link.confidence, link.method)]
            )

    attach(chunk_store, graph_store, ontology, full_config=full_cfg)
    return chunk_store, graph_store, ontology


# ---------------------------------------------------------------- A. inventory


def test_audit_inventory_counts_match_stores() -> None:
    chunk_store, graph_store, _ = _build_real_graph()
    report = audit(chunk_store, graph_store, config_hash="test-hash")

    assert isinstance(report, AuditReport)
    assert report.config_hash == "test-hash"
    assert report.inventory["chunks_total"] == 3
    assert report.inventory["shelves_total"] == len(graph_store.list_shelves())
    # at least one foods shelf
    assert any(k.startswith("shelves_foods") for k in report.inventory)


# ---------------------------------------------------------------- B. coverage


def test_audit_passes_on_clean_attach() -> None:
    """Healthy graph: every chunk with foodon_ids reaches a shelf."""
    chunk_store, graph_store, _ = _build_real_graph()
    report = audit(chunk_store, graph_store, config_hash="t")

    assert report.passed, (
        f"clean build should pass audit:\n{report}"
    )
    coverage_check = next(
        c for c in report.checks if c.name.startswith("FOODON-linked chunks")
    )
    assert coverage_check.passed
    assert coverage_check.metric == 1.0


def test_audit_flags_foodon_chunks_that_didnt_attach() -> None:
    """The 'edible food' pitfall: a chunk has FOODON ids but didn't attach.
    Coverage check should fail and report the orphan id."""
    # Build a chunk with a foodon_id that doesn't exist in any shelf,
    # AND no synthetic facet root to catch it.
    chunk_store = InMemoryChunkStore()
    chunk_store.upsert([
        _chunk("c1", foodon_ids=["FOODON:99999999"]),  # orphan
    ])
    graph_store = InMemoryGraphStore()
    graph_store.upsert_shelves([
        Shelf(
            shelf_id="foodon:UNRELATED",
            label="unrelated",
            facet="foods",
            depth=1,
            foodon_id="FOODON:UNRELATED",
            parent_shelf_id=None,
            chunk_count=0,
        ),
    ])

    report = audit(chunk_store, graph_store, config_hash="t")

    assert not report.passed
    coverage_check = next(
        c for c in report.checks if c.name.startswith("FOODON-linked chunks")
    )
    assert not coverage_check.passed
    assert coverage_check.metric == 0.0
    # Sample should surface the orphan foodon_id.
    assert any(
        s["foodon_id"] == "FOODON:99999999" for s in coverage_check.sample
    )


# ---------------------------------------------------------------- C. cross-store


def test_audit_flags_mention_denorm_mismatch() -> None:
    """foodon_ids on chunk says X; MENTIONS edges say Y. Audit must flag."""
    chunk_store = InMemoryChunkStore()
    chunk_store.upsert([
        _chunk("c1", foodon_ids=["FOODON:00000001"]),
    ])
    graph_store = InMemoryGraphStore()
    # Write a MENTIONS edge to a DIFFERENT FOODON id.
    graph_store.attach_chunks_to_entity(
        "FOODON:00000002", [("c1", 1.0, "dense")]
    )

    report = audit(chunk_store, graph_store, config_hash="t")
    cross_check = next(
        c for c in report.checks if "foodon_ids" in c.name and "MENTIONS" in c.name
    )
    assert not cross_check.passed, "drift between writers must be flagged"
    assert cross_check.details.get("mismatch", True)
    assert cross_check.metric == 0.0


def test_audit_passes_when_mentions_match_foodon_ids() -> None:
    chunk_store = InMemoryChunkStore()
    chunk_store.upsert([
        _chunk("c1", foodon_ids=["FOODON:00000001"]),
    ])
    graph_store = InMemoryGraphStore()
    graph_store.attach_chunks_to_entity(
        "FOODON:00000001", [("c1", 1.0, "dense")]
    )

    report = audit(chunk_store, graph_store, config_hash="t")
    cross_check = next(
        c for c in report.checks if "foodon_ids" in c.name and "MENTIONS" in c.name
    )
    assert cross_check.passed
    assert cross_check.metric == 1.0


# ---------------------------------------------------------------- D. attach integrity


def test_audit_flags_shelf_ids_neo4j_parity_mismatch() -> None:
    """Elastic shelf_ids says X; Neo4j ATTACHED_TO says Y. Audit must flag."""
    chunk_store = InMemoryChunkStore()
    chunk_store.upsert([
        _chunk("c1", shelf_ids=["s-only-in-elastic"]),
    ])
    graph_store = InMemoryGraphStore()
    graph_store.upsert_shelves([
        Shelf(shelf_id="s-only-in-elastic", label="x", facet="foods", depth=1,
              foodon_id=None, parent_shelf_id=None),
        Shelf(shelf_id="s-only-in-neo4j", label="y", facet="foods", depth=1,
              foodon_id=None, parent_shelf_id=None),
    ])
    # Write the wrong edge — chunk attached to a shelf the denorm doesn't know about.
    graph_store.attach_chunks_to_shelf("s-only-in-neo4j", [("c1", [])])

    report = audit(chunk_store, graph_store, config_hash="t")
    parity = next(
        c for c in report.checks if "shelf_ids" in c.name and "ATTACHED_TO" in c.name
    )
    assert not parity.passed, "denorm/edge drift must be flagged"
    # Sample should show the mismatch.
    assert len(parity.sample) == 1
    s0 = parity.sample[0]
    assert "only_in_elastic" in s0 and s0["only_in_elastic"] == ["s-only-in-elastic"]


def test_audit_attach_integrity_passes_after_real_attach() -> None:
    """The end-to-end build path produces consistent denorm + edges."""
    chunk_store, graph_store, _ = _build_real_graph()
    report = audit(chunk_store, graph_store, config_hash="t")
    parity = next(
        c for c in report.checks if "shelf_ids" in c.name and "ATTACHED_TO" in c.name
    )
    assert parity.passed, (
        f"real attach should produce parity; report:\n{report}"
    )


# ---------------------------------------------------------------- E. structural


def test_audit_flags_dangling_parent_reference() -> None:
    chunk_store = InMemoryChunkStore()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_shelves([
        Shelf(shelf_id="s-child", label="child", facet="foods", depth=2,
              foodon_id=None, parent_shelf_id="s-ghost"),  # ghost doesn't exist
    ])

    report = audit(chunk_store, graph_store, config_hash="t")
    dangling = next(c for c in report.checks if c.name == "dangling parent references")
    assert not dangling.passed
    assert dangling.metric == 1
    assert dangling.sample[0]["missing_parent"] == "s-ghost"


def test_audit_flags_multi_root_facet() -> None:
    chunk_store = InMemoryChunkStore()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_shelves([
        Shelf(shelf_id="r1", label="r1", facet="foods", depth=0,
              foodon_id=None, parent_shelf_id=None),
        Shelf(shelf_id="r2", label="r2", facet="foods", depth=0,
              foodon_id=None, parent_shelf_id=None),
    ])

    report = audit(chunk_store, graph_store, config_hash="t")
    multi = next(c for c in report.checks if c.name == "single root per facet")
    assert not multi.passed
    assert "foods" in multi.details["facets_with_multiple_roots"]


# ---------------------------------------------------------------- report shape


def test_audit_report_pretty_print_includes_all_sections() -> None:
    chunk_store, graph_store, _ = _build_real_graph()
    report = audit(chunk_store, graph_store, config_hash="t")
    s = str(report)
    assert "Inventory:" in s
    assert "B. Coverage" in s
    assert "C. Cross-store consistency" in s
    assert "D. Attach integrity" in s
    assert "E. Structural sanity" in s
    assert "Overall:" in s


def test_audit_report_is_json_serializable() -> None:
    """For diffing across runs — report must round-trip through JSON."""
    chunk_store, graph_store, _ = _build_real_graph()
    report = audit(chunk_store, graph_store, config_hash="t")
    json_str = report.model_dump_json()
    restored = AuditReport.model_validate_json(json_str)
    assert restored.config_hash == report.config_hash
    assert len(restored.checks) == len(report.checks)


def test_audit_check_passed_vs_critical_failures() -> None:
    chunk_store, graph_store, _ = _build_real_graph()
    report = audit(chunk_store, graph_store, config_hash="t")
    # Healthy build -> no critical failures -> report.passed is True.
    assert report.passed
    assert report.critical_failures == []


def test_audit_facade_method_returns_report() -> None:
    """fs.audit() returns an AuditReport whose config_hash matches fs."""
    fs = FoodScholar.in_memory()
    # No data, but audit should still run cleanly.
    report = fs.audit()
    assert isinstance(report, AuditReport)
    assert report.config_hash == fs.config_hash
    assert report.inventory["chunks_total"] == 0
