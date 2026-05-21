"""Tests for Layer A attachment phase.

Resolver unit tests use a hand-built ShelfIndex so each mode (direct /
collapsed / lifted / synthetic-root / no-shelf) can be exercised in
isolation. The orchestrator + facade tests go end-to-end through real
stores so the denormalization + idempotency invariants are caught here.
"""

from __future__ import annotations

from pathlib import Path

from foodscholar.config import FacetConfig, LayerAConfig
from foodscholar.io.chunk import Chunk, EntityLink, Mention
from foodscholar.io.graph import Shelf
from foodscholar.layer_a import (
    ShelfIndex,
    attach,
    build_layer_a,
    resolve_chunk,
    shelf_id_for_foodon,
)
from foodscholar.ontology import FoodOnAPI, load_ontology
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore

# ---------------------------------------------------------------- fixtures


def _mini_foodon() -> FoodOnAPI:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"
    return FoodOnAPI(load_ontology(path), prefix_filter=None)


def _foods_shelf(
    foodon_id: str | None,
    *,
    depth: int = 1,
    see_also: list[str] | None = None,
    shelf_id: str | None = None,
    label: str = "shelf",
) -> Shelf:
    sid = shelf_id or (shelf_id_for_foodon(foodon_id) if foodon_id else "facet:foods")
    return Shelf(
        shelf_id=sid,
        label=label,
        facet="foods",
        depth=depth,
        foodon_id=foodon_id,
        parent_shelf_id=None,
        chunk_count=0,
        see_also=see_also or [],
    )


def _chunk_with_links(
    chunk_id: str,
    links: list[tuple[str, float, str]] | None = None,
    *,
    foodon_ids: list[str] | None = None,
) -> Chunk:
    entity_links = [
        EntityLink(
            mention=Mention(
                text=ontology_id.split(":")[-1],
                start=0,
                end=1,
                score=conf,
                ner_model_version="test",
                entity_type=entity_type,  # type: ignore[arg-type]
            ),
            ontology_id=ontology_id,
            confidence=conf,
            method="dense",
            linker_version="test",
        )
        for ontology_id, conf, entity_type in (links or [])
    ]
    return Chunk(
        chunk_id=chunk_id,
        text="fixture text",
        source_doc_id="fixture-doc",
        source_type="abstract",
        section_type="abstract",
        entity_links=entity_links,
        foodon_ids=foodon_ids or [],
    )


# ---------------------------------------------------------------- resolver


def test_resolve_direct_hit() -> None:
    """A chunk linking a shelf's own foodon_id attaches direct (empty lifted_from)."""
    ontology = _mini_foodon()
    index = ShelfIndex.from_shelves([_foods_shelf("TEST:0000008", label="olive oil")])
    chunk = _chunk_with_links("c1", foodon_ids=["TEST:0000008"])

    resolutions = resolve_chunk(chunk, index, ontology)

    assert resolutions == {shelf_id_for_foodon("TEST:0000008"): []}


def test_resolve_collapsed_via_see_also() -> None:
    """A chunk linking a term that was collapsed into a survivor attaches to
    the survivor with the collapsed id in lifted_from."""
    ontology = _mini_foodon()
    # Pretend `olive` (007) collapsed into `olive oil` (008) — its foodon_id
    # is on 008's see_also.
    index = ShelfIndex.from_shelves(
        [_foods_shelf("TEST:0000008", see_also=["TEST:0000007"], label="olive oil")]
    )
    chunk = _chunk_with_links("c1", foodon_ids=["TEST:0000007"])

    resolutions = resolve_chunk(chunk, index, ontology)

    assert resolutions == {shelf_id_for_foodon("TEST:0000008"): ["TEST:0000007"]}


def test_resolve_lifted_to_deepest_surviving_ancestor() -> None:
    """A chunk linking a pruned term lifts to its nearest surviving ancestor.

    Mini ontology: olive oil (008) -> olive (007) -> fruit (004) -> plant food (002)
                                                    -> food product (001).
    Both `fruit` and `plant food` survive; `olive` and `olive oil` don't. A
    chunk linking olive oil should lift to `fruit` (deepest), not `plant food`.
    """
    ontology = _mini_foodon()
    index = ShelfIndex.from_shelves(
        [
            _foods_shelf("TEST:0000004", depth=3, label="fruit"),
            _foods_shelf("TEST:0000002", depth=2, label="plant food"),
        ]
    )
    chunk = _chunk_with_links("c1", foodon_ids=["TEST:0000008"])

    resolutions = resolve_chunk(chunk, index, ontology)

    # Lifts to deepest (fruit), not both.
    assert resolutions == {shelf_id_for_foodon("TEST:0000004"): ["TEST:0000008"]}


def test_resolve_orphan_routes_to_synthetic_root() -> None:
    """A FOODON id whose ancestry doesn't intersect any surviving shelf goes to
    the synthetic facet root if present."""
    ontology = _mini_foodon()
    synthetic = Shelf(
        shelf_id="facet:foods",
        label="Foods",
        facet="foods",
        depth=0,
        foodon_id=None,
        parent_shelf_id=None,
    )
    # No real shelves — just the synthetic root. Olive oil has no surviving
    # ancestor on the index.
    index = ShelfIndex.from_shelves([synthetic])
    chunk = _chunk_with_links("c1", foodon_ids=["TEST:0000008"])

    resolutions = resolve_chunk(chunk, index, ontology)

    assert resolutions == {"facet:foods": ["TEST:0000008"]}


def test_resolve_orphan_dropped_when_no_synthetic_root() -> None:
    """Without a synthetic root, an orphan FOODON id silently drops — no edge."""
    ontology = _mini_foodon()
    index = ShelfIndex.from_shelves([_foods_shelf("TEST:0000011", label="dairy")])
    # Chunk's olive oil has no ancestry that reaches dairy.
    chunk = _chunk_with_links("c1", foodon_ids=["TEST:0000008"])

    resolutions = resolve_chunk(chunk, index, ontology)

    assert resolutions == {}


def test_resolve_dedupes_multiple_foodon_ids_to_same_shelf() -> None:
    """Two foodon_ids resolving to the same shelf merge into one edge."""
    ontology = _mini_foodon()
    index = ShelfIndex.from_shelves([_foods_shelf("TEST:0000004", depth=3, label="fruit")])
    # olive (007) and olive oil (008) both lift to fruit.
    chunk = _chunk_with_links("c1", foodon_ids=["TEST:0000007", "TEST:0000008"])

    resolutions = resolve_chunk(chunk, index, ontology)

    assert list(resolutions.keys()) == [shelf_id_for_foodon("TEST:0000004")]
    assert sorted(resolutions[shelf_id_for_foodon("TEST:0000004")]) == [
        "TEST:0000007",
        "TEST:0000008",
    ]


def test_resolve_direct_wins_over_lifted_when_both_apply() -> None:
    """When a chunk links both the survivor's own id (direct) and an
    unrelated id that lifts to the same shelf, the edge stays 'direct'
    (empty lifted_from)."""
    ontology = _mini_foodon()
    # `fruit` survives. Chunk links fruit directly AND olive oil (lifts to fruit).
    index = ShelfIndex.from_shelves([_foods_shelf("TEST:0000004", depth=3, label="fruit")])
    chunk = _chunk_with_links("c1", foodon_ids=["TEST:0000004", "TEST:0000008"])

    resolutions = resolve_chunk(chunk, index, ontology)

    # Direct hit takes precedence — lifted_from is empty even though 008 also lifted.
    assert resolutions == {shelf_id_for_foodon("TEST:0000004"): []}


def test_resolve_facet_routing_keeps_health_off_foods() -> None:
    """A mention with entity_type='medical condition' on a FOODON id (rare,
    but possible if NEL drifts) doesn't attach to foods — `route_link_to_facet`
    routes it to health. No health shelves -> no edge for that link.
    The foods foodon_ids denorm still drives a foods attachment."""
    ontology = _mini_foodon()
    foods_shelf = _foods_shelf("TEST:0000004", depth=3, label="fruit")
    health_shelf = Shelf(
        shelf_id="facet:health",
        label="Health",
        facet="health",
        depth=0,
        foodon_id=None,
        parent_shelf_id=None,
    )
    index = ShelfIndex.from_shelves([foods_shelf, health_shelf])

    # entity_links: one is medical condition with a FOODON id (routes to health).
    # No health shelf has any FOODON id, so the orphan goes to the synthetic
    # health root. foodon_ids denorm seeds foods independently.
    chunk = _chunk_with_links(
        "c1",
        links=[("FOODON:99999999", 0.9, "medical condition")],
        foodon_ids=["TEST:0000008"],
    )

    resolutions = resolve_chunk(chunk, index, ontology)

    assert resolutions == {
        shelf_id_for_foodon("TEST:0000004"): ["TEST:0000008"],
        "facet:health": ["FOODON:99999999"],
    }


def test_resolve_chunk_with_no_foodon_routing_returns_empty() -> None:
    """A chunk with only non-FOODON links and no foodon_ids denorm produces
    no shelves (Layer A only projects FoodOn for now)."""
    ontology = _mini_foodon()
    index = ShelfIndex.from_shelves([_foods_shelf("TEST:0000004", depth=3, label="fruit")])
    chunk = _chunk_with_links("c1", links=[("CHEBI:12345", 0.9, "nutrient")])

    resolutions = resolve_chunk(chunk, index, ontology)

    assert resolutions == {}


# ---------------------------------------------------------------- orchestrator


def _full_config(layer_a: LayerAConfig | None = None):
    from foodscholar.config import FoodScholarConfig

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


def _build_corpus_and_layer_a() -> tuple[InMemoryChunkStore, InMemoryGraphStore, FoodOnAPI]:
    """Mirror the test_layer_a setup so we exercise the real builder output."""
    chunk_store = InMemoryChunkStore()
    chunk_store.upsert(
        [
            _chunk_with_links("c1", foodon_ids=["TEST:0000008"]),  # olive oil
            _chunk_with_links("c2", foodon_ids=["TEST:0000006"]),  # apple
            _chunk_with_links("c3", foodon_ids=["TEST:0000008", "TEST:9999999"]),  # olive oil + orphan
        ]
    )
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
    build_layer_a(
        chunk_store, graph_store, ontology, config=cfg, full_config=full_cfg
    )
    return chunk_store, graph_store, ontology


def test_attach_writes_edges_and_denormalizes_shelf_ids() -> None:
    chunk_store, graph_store, ontology = _build_corpus_and_layer_a()
    meta = attach(
        chunk_store,
        graph_store,
        ontology,
        full_config=_full_config(),
    )

    assert meta.phase == "attach"
    assert meta.record_count > 0

    # Every chunk should have at least one shelf_id denormalized.
    for cid in ("c1", "c2", "c3"):
        chunk = chunk_store.get(cid)
        assert chunk is not None
        assert chunk.shelf_ids, f"{cid} has no shelf_ids after attach"

    # And the graph store should know about it from the other side.
    olive_oil_shelf = shelf_id_for_foodon("TEST:0000008")
    if graph_store.get_shelf(olive_oil_shelf) is not None:
        bucket = graph_store._shelf_chunks[olive_oil_shelf]
        assert "c1" in bucket
        # c1's olive oil resolves direct -> empty lifted_from
        assert bucket["c1"] == []


def test_attach_is_idempotent() -> None:
    chunk_store, graph_store, ontology = _build_corpus_and_layer_a()
    full_cfg = _full_config()
    meta1 = attach(chunk_store, graph_store, ontology, full_config=full_cfg)
    snapshot = {
        sid: dict(bucket) for sid, bucket in graph_store._shelf_chunks.items()
    }
    shelf_ids_snapshot = {
        cid: list(chunk_store.get(cid).shelf_ids)  # type: ignore[union-attr]
        for cid in ("c1", "c2", "c3")
    }

    meta2 = attach(chunk_store, graph_store, ontology, full_config=full_cfg)

    # record_count + bucket contents must match exactly.
    assert meta1.record_count == meta2.record_count
    assert snapshot == {
        sid: dict(bucket) for sid, bucket in graph_store._shelf_chunks.items()
    }
    for cid in ("c1", "c2", "c3"):
        assert (
            chunk_store.get(cid).shelf_ids  # type: ignore[union-attr]
            == shelf_ids_snapshot[cid]
        )


def test_facade_attach_runs_end_to_end_against_in_memory() -> None:
    from foodscholar import FoodScholar

    chunk_store, graph_store, ontology = _build_corpus_and_layer_a()
    fs = FoodScholar(
        _full_config(),
        chunk_store=chunk_store,
        graph_store=graph_store,
    )
    fs.attach_ontology(ontology)

    meta = fs.attach()

    assert meta.phase == "attach"
    assert meta.record_count > 0
    # Every olive-oil-mentioning chunk denormalized at least one shelf.
    for cid in ("c1", "c3"):
        chunk = chunk_store.get(cid)
        assert chunk is not None
        assert chunk.shelf_ids


def test_graph_view_attach_chunks_still_works_with_new_signature() -> None:
    """`fs.graph.attach_chunks(...)` is the manual path — it shouldn't carry
    provenance, but the underlying protocol now demands tuples. Forwarding
    via empty lifted_from preserves the old behavior."""
    from foodscholar import FoodScholar

    fs = FoodScholar.in_memory()
    fs.graph.add_shelf(
        shelf_id="s-test",
        label="Test",
        facet="foods",
        depth=1,
    )
    # Upsert a chunk so update_attachments has something to update.
    fs.chunk_store.upsert(
        [_chunk_with_links("c1", foodon_ids=[])]
    )

    fs.graph.attach_chunks(["c1"], shelf="s-test")

    assert "c1" in fs.graph_store._shelf_chunks["s-test"]
    assert fs.graph_store._shelf_chunks["s-test"]["c1"] == []
    assert fs.chunk_store.get("c1").shelf_ids == ["s-test"]  # type: ignore[union-attr]


def test_attach_clears_prior_edges_when_projection_changes() -> None:
    """A second attach run with a different projection must produce only the
    new shelf's edges — no ghosts from the prior projection."""
    chunk_store, graph_store, ontology = _build_corpus_and_layer_a()
    full_cfg = _full_config()

    # First run.
    attach(chunk_store, graph_store, ontology, full_config=full_cfg)
    first_run_edges = {
        sid: dict(bucket) for sid, bucket in graph_store._shelf_chunks.items()
    }
    assert first_run_edges, "first run should write edges"

    # Reproject with a different min_support — fewer shelves survive, so
    # some of the old edges target shelves that no longer exist.
    new_layer_cfg = LayerAConfig(
        min_support=3,  # higher than original 2 — fewer shelves
        max_depth=5,
        collapse_single_child_chains=False,
        facets=["foods"],
        umbrella_direct_share_max=0.0,
    )
    new_full_cfg = _full_config(new_layer_cfg)
    build_layer_a(
        chunk_store, graph_store, ontology, config=new_layer_cfg, full_config=new_full_cfg
    )
    # build_layer_a clears shelves, which also drops their _shelf_chunks
    # entries — but if there were edges to OTHER shelves they'd survive.
    # Either way, attach() must produce a clean state.
    attach(chunk_store, graph_store, ontology, full_config=new_full_cfg)
    second_run_edges = {
        sid: dict(bucket) for sid, bucket in graph_store._shelf_chunks.items()
    }

    # Every edge in the second run must target a shelf that exists in the
    # current projection. No ghost edges.
    surviving_shelf_ids = {s.shelf_id for s in graph_store.list_shelves()}
    for sid in second_run_edges:
        assert sid in surviving_shelf_ids, (
            f"ghost edge to pruned shelf {sid!r}"
        )

    # Chunk-side shelf_ids must reference only currently-surviving shelves.
    for cid in ("c1", "c2", "c3"):
        chunk = chunk_store.get(cid)
        assert chunk is not None
        for sid in chunk.shelf_ids:
            assert sid in surviving_shelf_ids, (
                f"chunk {cid} carries stale shelf_id {sid!r}"
            )


def test_attach_clears_chunk_side_when_no_longer_resolves() -> None:
    """A chunk that resolved to a shelf in run 1 but to nothing in run 2 must
    end run 2 with an empty `shelf_ids` — not the stale list from run 1."""
    chunk_store, graph_store, ontology = _build_corpus_and_layer_a()
    full_cfg = _full_config()
    attach(chunk_store, graph_store, ontology, full_config=full_cfg)

    c1_before = chunk_store.get("c1")
    assert c1_before is not None and c1_before.shelf_ids

    # Strip every link/foodon_id off c1 by re-upserting a clean copy. c1
    # now resolves to nothing on the next attach run.
    chunk_store.upsert([_chunk_with_links("c1", foodon_ids=[])])

    attach(chunk_store, graph_store, ontology, full_config=full_cfg)

    c1_after = chunk_store.get("c1")
    assert c1_after is not None
    assert c1_after.shelf_ids == [], (
        "stale shelf_ids survived a re-run that no longer resolved this chunk"
    )


def test_bulk_update_attachments_writes_all_items_in_memory() -> None:
    """Sanity check the in-memory bulk path matches per-chunk update."""
    chunk_store = InMemoryChunkStore()
    chunk_store.upsert(
        [
            _chunk_with_links("c1", foodon_ids=[]),
            _chunk_with_links("c2", foodon_ids=[]),
        ]
    )

    chunk_store.bulk_update_attachments(
        [
            ("c1", ["s-a", "s-b"], []),
            ("c2", ["s-c"], ["t-1"]),
            ("c-missing", ["s-d"], []),  # unknown chunk_id — silently ignored
        ],
        wait_for_refresh=True,
    )

    assert chunk_store.get("c1").shelf_ids == ["s-a", "s-b"]  # type: ignore[union-attr]
    assert chunk_store.get("c2").shelf_ids == ["s-c"]  # type: ignore[union-attr]
    assert chunk_store.get("c2").theme_ids == ["t-1"]  # type: ignore[union-attr]


def test_clear_attachments_drops_denorm_and_edges_only() -> None:
    """clear_attachments must wipe shelf_ids/theme_ids on chunks and shelf->chunk
    edges in the graph — but leave shelves themselves and chunk content alone."""
    chunk_store, graph_store, ontology = _build_corpus_and_layer_a()
    full_cfg = _full_config()
    attach(chunk_store, graph_store, ontology, full_config=full_cfg)

    n_shelves_before = len(graph_store.list_shelves())
    c1_text_before = chunk_store.get("c1").text  # type: ignore[union-attr]

    graph_store.clear_attachments()
    chunk_store.clear_attachments()

    # Shelves survive.
    assert len(graph_store.list_shelves()) == n_shelves_before
    # Edge map is empty.
    assert not graph_store._shelf_chunks
    # Chunk content (text, foodon_ids, links) survives — only shelf/theme go.
    c1 = chunk_store.get("c1")
    assert c1 is not None
    assert c1.text == c1_text_before
    assert c1.foodon_ids == ["TEST:0000008"]
    assert c1.shelf_ids == []
    assert c1.theme_ids == []


def test_attach_handles_per_facet_overrides_without_crashing() -> None:
    """Smoke test that the resolver works when the projection has stub roots
    for non-foods facets (the common state on the prototype corpus)."""
    chunk_store = InMemoryChunkStore()
    chunk_store.upsert([_chunk_with_links("c1", foodon_ids=["TEST:0000008"])])
    ontology = _mini_foodon()
    graph_store = InMemoryGraphStore()

    cfg = LayerAConfig(
        min_support=1,
        max_depth=5,
        collapse_single_child_chains=True,
        facets=["foods", "health"],
        facet_overrides={"foods": FacetConfig(min_support=1)},
        umbrella_direct_share_max=0.0,
    )
    full_cfg = _full_config(cfg)
    build_layer_a(chunk_store, graph_store, ontology, config=cfg, full_config=full_cfg)

    meta = attach(chunk_store, graph_store, ontology, full_config=full_cfg)
    assert meta.record_count > 0


def test_attach_record_count_matches_actual_edges() -> None:
    """The honest version of the bug we hit on the live corpus: meta.record_count
    must equal the actual count of (chunk_id, shelf_id) pairs in the graph, not
    the number of (chunk_id, shelf_id) tuples we *submitted* (the latter can be
    inflated by iter_chunks yielding the same chunk twice or by duplicate
    resolutions across batches). The dict-backed pending_edges buffer + the
    seen_chunks dedupe guarantee this invariant."""
    chunk_store, graph_store, ontology = _build_corpus_and_layer_a()
    meta = attach(chunk_store, graph_store, ontology, full_config=_full_config())

    actual_edges = sum(
        len(bucket) for bucket in graph_store._shelf_chunks.values()
    )
    assert meta.record_count == actual_edges, (
        f"record_count ({meta.record_count}) != actual edges in store ({actual_edges}) "
        "— the dedupe invariant broke"
    )


def test_attach_dedupes_chunk_yielded_twice_by_iter_chunks() -> None:
    """Defensive: if iter_chunks yields the same chunk twice (which ES `_doc`
    sort can do under concurrent writes mid-scan), attach() must still write
    each (chunk, shelf) edge exactly once and report n_edges correctly."""

    class _DoubleYieldStore(InMemoryChunkStore):
        def iter_chunks(self, batch_size: int = 1000):
            # Yield every chunk twice across two batches — simulating the
            # ES search_after duplicate that motivated the dedupe.
            chunks = list(self._chunks.values())
            yield chunks
            yield chunks  # same chunks again

    chunk_store = _DoubleYieldStore()
    chunk_store.upsert(
        [_chunk_with_links("c1", foodon_ids=["TEST:0000008"])]
    )
    ontology = _mini_foodon()
    graph_store = InMemoryGraphStore()
    cfg = LayerAConfig(
        min_support=1,
        max_depth=5,
        collapse_single_child_chains=False,
        facets=["foods"],
        umbrella_direct_share_max=0.0,
    )
    full_cfg = _full_config(cfg)
    build_layer_a(chunk_store, graph_store, ontology, config=cfg, full_config=full_cfg)

    meta = attach(chunk_store, graph_store, ontology, full_config=full_cfg)

    actual_edges = sum(
        len(bucket) for bucket in graph_store._shelf_chunks.values()
    )
    # c1 has one foodon_id resolving to one shelf -> exactly 1 edge,
    # despite c1 being yielded by iter_chunks twice.
    assert actual_edges == 1, f"expected 1 deduped edge, got {actual_edges}"
    assert meta.record_count == 1, (
        f"meta.record_count ({meta.record_count}) inflated past actual edges"
    )
