from pathlib import Path

from foodscholar.config import FacetConfig, LayerAConfig, LinkBlocklistEntry
from foodscholar.io.chunk import Chunk, EntityLink, Mention
from foodscholar.layer_a import build_layer_a, build_shelves, shelf_id_for_foodon
from foodscholar.ontology import FoodOnAPI, load_ontology
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore


def _mini_foodon() -> FoodOnAPI:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "mini_foodon.obo"
    return FoodOnAPI(load_ontology(path), prefix_filter=None)


def _chunk(chunk_id: str, foodon_ids: list[str]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="fixture text",
        source_doc_id="fixture-doc",
        source_type="abstract",
        section_type="abstract",
        foodon_ids=foodon_ids,
    )


def _store() -> InMemoryChunkStore:
    store = InMemoryChunkStore()
    store.upsert(
        [
            _chunk("c1", ["TEST:0000008"]),
            _chunk("c2", ["TEST:0000006"]),
            _chunk("c3", ["TEST:0000008", "TEST:9999999"]),
        ]
    )
    return store


def test_build_shelves_propagates_foodon_support_to_ancestors() -> None:
    # Collapse disabled — this test focuses on the propagation pass producing
    # the right counts; collapse is exercised separately below.
    shelves = build_shelves(
        _store(),
        _mini_foodon(),
        LayerAConfig(
            min_support=2,
            max_depth=5,
            collapse_single_child_chains=False,
            facets=["foods"],
            umbrella_direct_share_max=0.0,
        ),
    )

    by_id = {s.shelf_id: s for s in shelves}
    assert shelf_id_for_foodon("TEST:0000006") not in by_id
    assert by_id[shelf_id_for_foodon("TEST:0000001")].chunk_count == 3
    assert by_id[shelf_id_for_foodon("TEST:0000008")].chunk_count == 2
    assert by_id[shelf_id_for_foodon("TEST:0000008")].parent_shelf_id == shelf_id_for_foodon(
        "TEST:0000007"
    )
    assert [s.depth for s in shelves] == sorted(s.depth for s in shelves)


def test_build_shelves_honors_blacklist_and_reparents_to_nearest_included_ancestor() -> None:
    shelves = build_shelves(
        _store(),
        _mini_foodon(),
        LayerAConfig(
            min_support=1,
            max_depth=5,
            collapse_single_child_chains=False,
            blacklist_terms=["plant food"],
            facets=["foods"],
            umbrella_direct_share_max=0.0,
        ),
    )

    by_id = {s.shelf_id: s for s in shelves}
    assert shelf_id_for_foodon("TEST:0000002") not in by_id
    assert by_id[shelf_id_for_foodon("TEST:0000004")].parent_shelf_id == shelf_id_for_foodon(
        "TEST:0000001"
    )


def test_build_layer_a_upserts_shelves_and_returns_artifact_meta() -> None:
    from foodscholar import FoodScholar

    fs = FoodScholar.in_memory()
    fs.config.layer_a.min_support = 2
    fs.config.layer_a.collapse_single_child_chains = False
    fs.config.layer_a.facets = ["foods"]
    fs.config.layer_a.umbrella_direct_share_max = 0.0
    fs.attach_ontology(_mini_foodon())
    fs.upsert_chunks(_store().scan())

    meta = fs.build_layer_a()

    assert meta.phase == "build-layer-a"
    assert meta.record_count == 5
    assert fs.graph_store.get_shelf(shelf_id_for_foodon("TEST:0000008")) is not None


def test_build_layer_a_function_writes_to_graph_store() -> None:
    from foodscholar import FoodScholarConfig

    chunk_store = _store()
    graph_store = InMemoryGraphStore()
    config = FoodScholarConfig.model_validate(
        {
            "corpus": {"chunks_path": "data/chunks.parquet"},
            "storage": {
                "chunk_store": {"backend": "memory"},
                "graph_store": {"backend": "memory"},
            },
            "layer_a": {
                "min_support": 2,
                "collapse_single_child_chains": False,
                "facets": ["foods"],
                "umbrella_direct_share_max": 0.0,
            },
        }
    )

    meta = build_layer_a(
        chunk_store,
        graph_store,
        _mini_foodon(),
        config=config.layer_a,
        full_config=config,
    )

    assert meta.record_count == len(graph_store.list_shelves())
    assert graph_store.get_shelf(shelf_id_for_foodon("TEST:0000007")) is not None


# ---------------------------------------------------------------- new in 7.0


def _chunk_with_links(
    chunk_id: str,
    links: list[tuple[str, float, str]],
) -> Chunk:
    """Build a chunk with explicit EntityLinks. `links` = (ontology_id, conf, entity_type)."""
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
        for ontology_id, conf, entity_type in links
    ]
    return Chunk(
        chunk_id=chunk_id,
        text="fixture text",
        source_doc_id="fixture-doc",
        source_type="abstract",
        section_type="abstract",
        entity_links=entity_links,
        foodon_ids=[link.ontology_id for link in entity_links if link.ontology_id.startswith("FOODON:")],
    )


def test_single_child_collapse_fires_on_pure_chain() -> None:
    # Default config has collapse=True. The chain food product -> plant food
    # -> fruit -> olive -> olive oil is single-child after threshold=1 keeps
    # everything; only olive oil survives, with the rest in see_also.
    store = InMemoryChunkStore()
    store.upsert([_chunk("c1", ["TEST:0000008"])])

    shelves = build_shelves(
        store,
        _mini_foodon(),
        LayerAConfig(
            min_support=1,
            max_depth=10,
            facets=["foods"],
            umbrella_direct_share_max=0.0,
        ),
    )

    by_id = {s.shelf_id: s for s in shelves}
    leaf = by_id.get(shelf_id_for_foodon("TEST:0000008"))
    assert leaf is not None
    # All four ancestors should have collapsed into the leaf.
    assert shelf_id_for_foodon("TEST:0000007") not in by_id
    assert shelf_id_for_foodon("TEST:0000004") not in by_id
    assert shelf_id_for_foodon("TEST:0000002") not in by_id
    assert shelf_id_for_foodon("TEST:0000001") not in by_id
    # And recorded in see_also for provenance.
    assert "TEST:0000007" in leaf.see_also
    assert "TEST:0000001" in leaf.see_also


def test_single_child_collapse_does_not_fire_when_siblings_survive() -> None:
    # Add a chunk for peanut (TEST:0000009) — sibling of fruit under plant food.
    # Now plant food has two surviving children (fruit branch + peanut), so it
    # cannot collapse. Fruit only has olive under it (apple pruned), so the
    # olive chain still collapses.
    store = InMemoryChunkStore()
    store.upsert(
        [
            _chunk("c1", ["TEST:0000008"]),
            _chunk("c2", ["TEST:0000008"]),
            _chunk("c3", ["TEST:0000009"]),
            _chunk("c4", ["TEST:0000009"]),
        ]
    )

    shelves = build_shelves(
        store,
        _mini_foodon(),
        LayerAConfig(
            min_support=2,
            max_depth=10,
            facets=["foods"],
            umbrella_direct_share_max=0.0,
        ),
    )

    by_id = {s.shelf_id: s for s in shelves}
    # plant food has two children that survived — should remain.
    assert shelf_id_for_foodon("TEST:0000002") in by_id


def test_depth_cap_lifts_to_nearest_ancestor() -> None:
    # max_depth=2 should lift olive oil (depth 4 normally) up to live under
    # an ancestor at depth <= 2. With collapse off so we can see the lift
    # directly.
    store = InMemoryChunkStore()
    store.upsert([_chunk("c1", ["TEST:0000008"])])

    shelves = build_shelves(
        store,
        _mini_foodon(),
        LayerAConfig(
            min_support=1,
            max_depth=2,
            collapse_single_child_chains=False,
            facets=["foods"],
            umbrella_direct_share_max=0.0,
        ),
    )

    by_id = {s.shelf_id: s for s in shelves}
    leaf = by_id[shelf_id_for_foodon("TEST:0000008")]
    # Reported depth must respect the cap.
    assert leaf.depth <= 2
    # Parent must be one of the surviving ancestors at depth <= cap.
    if leaf.parent_shelf_id is not None:
        parent = by_id[leaf.parent_shelf_id]
        assert parent.depth <= 2


def test_whitelist_keeps_term_below_threshold() -> None:
    # Apple normally pruned at min_support=2. Whitelist keeps it.
    store = InMemoryChunkStore()
    store.upsert([_chunk("c1", ["TEST:0000006"])])

    cfg = LayerAConfig(
        min_support=2,
        max_depth=10,
        collapse_single_child_chains=False,
        facets=["foods"],
        facet_overrides={"foods": FacetConfig(whitelist=["TEST:0000006"])},
    )
    shelves = build_shelves(store, _mini_foodon(), cfg)

    by_id = {s.shelf_id: s for s in shelves}
    assert shelf_id_for_foodon("TEST:0000006") in by_id


def test_confidence_floor_filters_low_confidence_links() -> None:
    # A chunk with a FOODON link below the floor is excluded from support.
    store = InMemoryChunkStore()
    store.upsert(
        [
            _chunk_with_links("c1", [("FOODON:00001002", 0.65, "other")]),
            _chunk_with_links("c2", [("FOODON:00001002", 0.65, "other")]),
            _chunk_with_links("c3", [("FOODON:00001002", 0.65, "other")]),
        ]
    )
    # Build an ontology fixture that knows FOODON:00001002.
    # Use the mini fixture with prefix_filter=None and just rely on chunks not
    # finding the term — empty shelves expected.
    ontology = _mini_foodon()
    cfg = LayerAConfig(
        min_support=1,
        min_link_confidence=0.70,
        facets=["foods"],
        collapse_single_child_chains=False,
    )
    shelves = build_shelves(store, ontology, cfg)
    # FOODON:00001002 isn't in the mini fixture; foods facet ends with stub root.
    assert len(shelves) == 1
    assert shelves[0].shelf_id == "facet:foods"


def test_see_also_records_collapsed_foodon_ids() -> None:
    store = InMemoryChunkStore()
    store.upsert([_chunk("c1", ["TEST:0000008"])])

    shelves = build_shelves(
        store,
        _mini_foodon(),
        LayerAConfig(
            min_support=1,
            max_depth=10,
            facets=["foods"],
            umbrella_direct_share_max=0.0,
        ),
    )
    leaf = next(s for s in shelves if s.shelf_id == shelf_id_for_foodon("TEST:0000008"))
    # Every ancestor in the chain should be present in see_also.
    for expected in ["TEST:0000001", "TEST:0000002", "TEST:0000004", "TEST:0000007"]:
        assert expected in leaf.see_also, f"missing {expected} in {leaf.see_also}"


def test_per_facet_override_resolves_over_globals() -> None:
    cfg = LayerAConfig(
        min_support=20,
        max_depth=5,
        blacklist_terms=["global term"],
        facet_overrides={
            "foods": FacetConfig(min_support=25, blacklist_terms=["foods term"]),
        },
    )
    foods = cfg.resolve_facet("foods")
    health = cfg.resolve_facet("health")
    assert foods.min_support == 25
    assert foods.blacklist_terms == ["foods term"]
    assert foods.max_depth == 5  # falls back to global
    assert health.min_support == 20  # no override
    assert health.blacklist_terms == ["global term"]


def test_sustainability_emits_stub_root() -> None:
    # Sustainability has no entity_type mapped to it; always a stub root.
    store = _store()  # foods-only chunks
    cfg = LayerAConfig(facets=["sustainability"])
    shelves = build_shelves(store, _mini_foodon(), cfg)
    assert len(shelves) == 1
    stub = shelves[0]
    assert stub.facet == "sustainability"
    assert stub.shelf_id == "facet:sustainability"
    assert stub.foodon_id is None
    assert stub.parent_shelf_id is None
    assert stub.chunk_count == 0
    assert stub.depth == 0


def test_umbrella_rule_drops_organizational_classes() -> None:
    # `plant food` (TEST:0000002) is a pure organizational class in the mini
    # fixture — no chunk mentions it directly, but it accumulates lifted
    # support via olive oil + peanut chunks. The umbrella rule MUST catch it
    # by structure, not by name.
    store = InMemoryChunkStore()
    store.upsert(
        [
            _chunk("c1", ["TEST:0000008"]),  # olive oil (under plant food)
            _chunk("c2", ["TEST:0000008"]),
            _chunk("c3", ["TEST:0000008"]),
            _chunk("c4", ["TEST:0000009"]),  # peanut (also under plant food)
            _chunk("c5", ["TEST:0000009"]),
        ]
    )
    # Umbrella defaults (0.10 / 0.85) — `plant food` direct=0, lifted_share=1.0.
    # min_count guard is normally 100; lower for the fixture's small counts.
    cfg = LayerAConfig(
        min_support=1,
        max_depth=10,
        collapse_single_child_chains=False,
        facets=["foods"],
        umbrella_min_count=1,
    )
    shelves = build_shelves(store, _mini_foodon(), cfg)
    by_id = {s.shelf_id: s for s in shelves}
    assert shelf_id_for_foodon("TEST:0000002") not in by_id, (
        "plant food should have been caught by umbrella rule"
    )
    # Real leaves with direct mentions survive.
    assert shelf_id_for_foodon("TEST:0000008") in by_id
    assert shelf_id_for_foodon("TEST:0000009") in by_id


def test_umbrella_rule_keeps_term_with_meaningful_direct_support() -> None:
    # Same shape as above, but now plant food (TEST:0000002) gets direct
    # support from a chunk that mentions it explicitly. direct_share goes from
    # 0 to non-zero. If umbrella_direct_share_max is below that share, the
    # term survives.
    store = InMemoryChunkStore()
    store.upsert(
        [
            _chunk("c0", ["TEST:0000002"]),  # one chunk mentions plant food directly
            _chunk("c1", ["TEST:0000008"]),
            _chunk("c2", ["TEST:0000008"]),
            _chunk("c3", ["TEST:0000008"]),
            _chunk("c4", ["TEST:0000009"]),
            _chunk("c5", ["TEST:0000009"]),
        ]
    )
    # plant food: direct=1, count_wd=6, direct_share=0.167. Default
    # umbrella_direct_share_max=0.10 → 0.167 is NOT < 0.10 → survives.
    cfg = LayerAConfig(
        min_support=1,
        max_depth=10,
        collapse_single_child_chains=False,
        facets=["foods"],
        umbrella_min_count=1,
    )
    shelves = build_shelves(store, _mini_foodon(), cfg)
    by_id = {s.shelf_id: s for s in shelves}
    assert shelf_id_for_foodon("TEST:0000002") in by_id


def test_umbrella_rule_bypassed_by_whitelist() -> None:
    # Even when an umbrella-shaped term is detected, whitelisting it keeps it.
    # This is the "navigation anchor" override.
    store = InMemoryChunkStore()
    store.upsert(
        [
            _chunk("c1", ["TEST:0000008"]),
            _chunk("c2", ["TEST:0000008"]),
            _chunk("c3", ["TEST:0000009"]),
            _chunk("c4", ["TEST:0000009"]),
        ]
    )
    cfg = LayerAConfig(
        min_support=1,
        max_depth=10,
        collapse_single_child_chains=False,
        facets=["foods"],
        umbrella_min_count=1,  # so umbrella DOES try to fire
        facet_overrides={
            "foods": FacetConfig(whitelist=["TEST:0000002"]),
        },
    )
    shelves = build_shelves(store, _mini_foodon(), cfg)
    by_id = {s.shelf_id: s for s in shelves}
    # plant food was umbrella-shaped (direct=0, lifted=4) but whitelisted.
    assert shelf_id_for_foodon("TEST:0000002") in by_id


def test_umbrella_rule_disabled_when_direct_share_max_is_zero() -> None:
    # Setting umbrella_direct_share_max=0 disables the rule entirely.
    store = InMemoryChunkStore()
    store.upsert(
        [
            _chunk("c1", ["TEST:0000008"]),
            _chunk("c2", ["TEST:0000008"]),
        ]
    )
    cfg = LayerAConfig(
        min_support=1,
        max_depth=10,
        collapse_single_child_chains=False,
        facets=["foods"],
        umbrella_direct_share_max=0.0,
    )
    shelves = build_shelves(store, _mini_foodon(), cfg)
    by_id = {s.shelf_id: s for s in shelves}
    # All ancestors survive — including the umbrella-shaped `plant food`.
    assert shelf_id_for_foodon("TEST:0000002") in by_id
    assert shelf_id_for_foodon("TEST:0000001") in by_id


def test_umbrella_min_count_guard_protects_small_shelves() -> None:
    # When `umbrella_min_count` is raised above `min_support`, a small shelf
    # that matches direct-share + lifted-share is left alone — variance is
    # too high at that scale to confidently call it an umbrella. The default
    # is `umbrella_min_count = min_support` (== threshold survivors are all
    # umbrella-eligible), but this test forces the guard to fire by setting
    # min_count = 100 above the fixture's small chunk counts.
    store = InMemoryChunkStore()
    store.upsert(
        [
            _chunk("c1", ["TEST:0000008"]),
            _chunk("c2", ["TEST:0000008"]),
            _chunk("c3", ["TEST:0000008"]),
            _chunk("c4", ["TEST:0000009"]),
            _chunk("c5", ["TEST:0000009"]),
        ]
    )
    cfg = LayerAConfig(
        min_support=1,
        max_depth=10,
        collapse_single_child_chains=False,
        facets=["foods"],
        umbrella_min_count=100,  # forces guard ABOVE the fixture's counts
    )
    shelves = build_shelves(store, _mini_foodon(), cfg)
    by_id = {s.shelf_id: s for s in shelves}
    # plant food is umbrella-shaped (direct=0, lifted=5) but below min_count.
    assert shelf_id_for_foodon("TEST:0000002") in by_id


def test_synthetic_facet_root_injected_when_multiple_orphans() -> None:
    # Multiple disconnected branches in FoodOn → orphan roots → synthetic
    # facet root collects them all so the projection is one tree, not a forest.
    store = InMemoryChunkStore()
    store.upsert(
        [
            _chunk("c1", ["TEST:0000008"]),  # olive oil branch
            _chunk("c2", ["TEST:0000008"]),
            _chunk("c3", ["TEST:0000009"]),  # peanut branch
            _chunk("c4", ["TEST:0000009"]),
            _chunk("c5", ["TEST:0000011"]),  # dairy branch (separate sibling of plant food)
            _chunk("c6", ["TEST:0000011"]),
        ]
    )
    cfg = LayerAConfig(
        min_support=1,
        max_depth=10,
        collapse_single_child_chains=False,
        facets=["foods"],
        umbrella_min_count=1,  # let umbrella fire so org classes drop
    )
    shelves = build_shelves(store, _mini_foodon(), cfg)
    # Exactly one synthetic facet root at depth=0.
    roots = [s for s in shelves if s.parent_shelf_id is None]
    assert len(roots) == 1
    root = roots[0]
    assert root.shelf_id == "facet:foods"
    assert root.depth == 0
    assert root.foodon_id is None
    # Synthetic root has no foodon_id → no chunk can name it directly →
    # support_direct MUST be 0 and every chunk reaching it counts as lifted.
    assert root.support_direct == 0
    assert root.support_lifted == root.chunk_count
    # All other shelves got shifted down by 1.
    non_root = [s for s in shelves if s.parent_shelf_id is not None]
    assert all(s.depth >= 1 for s in non_root)


def test_synthetic_root_chunk_count_is_unique_chunks_not_sum_of_roots() -> None:
    # A chunk linking to two terms in different orphan branches must not be
    # double-counted on the synthetic root. Olive oil (TEST:0000008) and
    # peanut (TEST:0000009) live under different surviving sub-trees after
    # the umbrella rule kills plant food. With one chunk attached to BOTH,
    # the synthetic root's chunk_count should be 1 — not 2.
    store = InMemoryChunkStore()
    store.upsert(
        [
            _chunk("c1", ["TEST:0000008", "TEST:0000009"]),  # both in one chunk
            _chunk("c2", ["TEST:0000008"]),  # olive oil only
            _chunk("c3", ["TEST:0000009"]),  # peanut only
        ]
    )
    cfg = LayerAConfig(
        min_support=1,
        max_depth=10,
        collapse_single_child_chains=False,
        facets=["foods"],
        umbrella_min_count=1,  # force umbrella to fire on plant food
    )
    shelves = build_shelves(store, _mini_foodon(), cfg)
    root = next(s for s in shelves if s.shelf_id == "facet:foods")
    # Three unique chunks total reach the foods facet; sum-of-roots would have
    # given 4 (c1 counted under each of olive oil's and peanut's subtrees).
    assert root.chunk_count == 3, (
        f"synthetic root must count unique chunks (expected 3, got {root.chunk_count})"
    )
    assert root.support_direct == 0
    assert root.support_lifted == 3


def test_synthetic_root_not_injected_when_already_single_rooted() -> None:
    # If the projection naturally has one root, don't add a synthetic one.
    store = InMemoryChunkStore()
    store.upsert([_chunk("c1", ["TEST:0000008"])])

    cfg = LayerAConfig(
        min_support=1,
        max_depth=10,
        collapse_single_child_chains=True,
        facets=["foods"],
        umbrella_min_count=1,
    )
    shelves = build_shelves(store, _mini_foodon(), cfg)
    # Only the umbrella-collapsed leaf survives; it's the single root.
    roots = [s for s in shelves if s.parent_shelf_id is None]
    assert len(roots) == 1
    # No synthetic root injected.
    assert all(s.shelf_id != "facet:foods" for s in shelves)


def _chunk_with_links_text(
    chunk_id: str,
    links: list[tuple[str, float, str]],
    *,
    surface: str,
) -> Chunk:
    """Build a chunk where every link's mention.text == `surface`."""
    entity_links = [
        EntityLink(
            mention=Mention(
                text=surface,
                start=0,
                end=len(surface),
                score=conf,
                ner_model_version="test",
                entity_type=entity_type,  # type: ignore[arg-type]
            ),
            ontology_id=ontology_id,
            confidence=conf,
            method="dense",
            linker_version="test",
        )
        for ontology_id, conf, entity_type in links
    ]
    return Chunk(
        chunk_id=chunk_id,
        text="fixture text",
        source_doc_id="fixture-doc",
        source_type="abstract",
        section_type="abstract",
        entity_links=entity_links,
        # No foodon_ids denormalization — force support to come only from
        # entity_links, so the blocklist's effect is observable.
        foodon_ids=[],
    )


def test_link_blocklist_filters_propagation() -> None:
    # Three chunks link "fish" → TEST:0000009 (the surface-form-drift case).
    # Without blocklist: TEST:0000009 has direct=3. With blocklist matching
    # (fish, TEST:0000009): support is zero, shelf doesn't survive.
    store = InMemoryChunkStore()
    store.upsert(
        [
            _chunk_with_links_text("c1", [("TEST:0000009", 0.9, "food")], surface="fish"),
            _chunk_with_links_text("c2", [("TEST:0000009", 0.9, "food")], surface="fish"),
            _chunk_with_links_text("c3", [("TEST:0000009", 0.9, "food")], surface="fish"),
        ]
    )

    # Without blocklist: term 9 survives at min_support=1.
    cfg_no_block = LayerAConfig(
        min_support=1, max_depth=10, facets=["foods"],
        umbrella_min_count=1, collapse_single_child_chains=False,
        link_blocklist=[],
    )
    shelves_no_block = build_shelves(store, _mini_foodon(), cfg_no_block)
    by_id_no_block = {s.shelf_id: s for s in shelves_no_block}
    assert shelf_id_for_foodon("TEST:0000009") in by_id_no_block

    # With blocklist on (fish, TEST:0000009): term 9 has zero support → gone.
    cfg_with_block = LayerAConfig(
        min_support=1, max_depth=10, facets=["foods"],
        umbrella_min_count=1, collapse_single_child_chains=False,
        link_blocklist=[LinkBlocklistEntry(surface="fish", ontology_id="TEST:0000009")],
    )
    shelves_with_block = build_shelves(store, _mini_foodon(), cfg_with_block)
    by_id_with_block = {s.shelf_id: s for s in shelves_with_block}
    assert shelf_id_for_foodon("TEST:0000009") not in by_id_with_block


def test_link_blocklist_surface_matching_is_case_insensitive() -> None:
    store = InMemoryChunkStore()
    store.upsert([
        _chunk_with_links_text("c1", [("TEST:0000009", 0.9, "food")], surface="FISH"),
        _chunk_with_links_text("c2", [("TEST:0000009", 0.9, "food")], surface="Fish"),
    ])
    cfg = LayerAConfig(
        min_support=1, max_depth=10, facets=["foods"],
        umbrella_min_count=1, collapse_single_child_chains=False,
        link_blocklist=[LinkBlocklistEntry(surface="fish", ontology_id="TEST:0000009")],
    )
    shelves = build_shelves(store, _mini_foodon(), cfg)
    by_id = {s.shelf_id: s for s in shelves}
    assert shelf_id_for_foodon("TEST:0000009") not in by_id


def test_link_blocklist_ontology_id_is_specific() -> None:
    # Blocking (fish, peanut) must NOT affect "fish" linked to olive_oil
    # — the blocklist is a (surface, id) PAIR, not a surface or id alone.
    store = InMemoryChunkStore()
    store.upsert([
        _chunk_with_links_text("c1", [("TEST:0000008", 0.9, "food")], surface="fish"),
        _chunk_with_links_text("c2", [("TEST:0000008", 0.9, "food")], surface="fish"),
    ])
    cfg = LayerAConfig(
        min_support=1, max_depth=10, facets=["foods"],
        umbrella_min_count=1, collapse_single_child_chains=False,
        link_blocklist=[LinkBlocklistEntry(surface="fish", ontology_id="TEST:0000009")],
    )
    shelves = build_shelves(store, _mini_foodon(), cfg)
    by_id = {s.shelf_id: s for s in shelves}
    # Olive oil link survives — blocklist entry doesn't match its ontology_id.
    assert shelf_id_for_foodon("TEST:0000008") in by_id


def test_build_layer_a_clears_stale_shelves_on_rerun() -> None:
    # First run with a narrow blacklist — `plant food` survives.
    from foodscholar import FoodScholar

    fs = FoodScholar.in_memory()
    fs.config.layer_a.min_support = 1
    fs.config.layer_a.collapse_single_child_chains = False
    fs.config.layer_a.facets = ["foods"]
    fs.config.layer_a.umbrella_direct_share_max = 0.0
    fs.attach_ontology(_mini_foodon())
    fs.upsert_chunks(_store().scan())
    fs.build_layer_a()
    first = {s.shelf_id for s in fs.graph_store.list_shelves()}
    assert shelf_id_for_foodon("TEST:0000002") in first

    # Second run with `plant food` blacklisted: the stale shelf MUST disappear,
    # not just be hidden by a smaller new set.
    fs.config.layer_a.blacklist_terms = ["plant food"]
    fs.build_layer_a()
    second = {s.shelf_id for s in fs.graph_store.list_shelves()}
    assert shelf_id_for_foodon("TEST:0000002") not in second
    # And nothing from the first run that doesn't belong in the second survives.
    stale = first - second
    assert all(
        fs.graph_store.get_shelf(sid) is None for sid in stale
    ), f"ghost shelves survived re-build: {stale}"


def test_blacklist_runs_before_threshold() -> None:
    # Blacklist plant food. Its support comes mostly via descendants. With
    # blacklist BEFORE threshold, plant food is dropped and its descendants'
    # support still propagates to surviving ancestor (food product). With
    # collapse OFF so we can see the structure.
    store = InMemoryChunkStore()
    store.upsert(
        [
            _chunk("c1", ["TEST:0000008"]),  # plant food via olive chain
            _chunk("c2", ["TEST:0000008"]),
        ]
    )
    shelves = build_shelves(
        store,
        _mini_foodon(),
        LayerAConfig(
            min_support=1,
            max_depth=10,
            collapse_single_child_chains=False,
            blacklist_terms=["plant food"],
            facets=["foods"],
            umbrella_direct_share_max=0.0,
        ),
    )
    by_id = {s.shelf_id: s for s in shelves}
    assert shelf_id_for_foodon("TEST:0000002") not in by_id
    # food product should still have the chunks via lifting.
    food_product = by_id[shelf_id_for_foodon("TEST:0000001")]
    assert food_product.chunk_count == 2
    # And olive's parent skips plant food → food product.
    olive = by_id[shelf_id_for_foodon("TEST:0000007")]
    # Walk up through surviving parent chain — at minimum food product is reachable.
    assert olive.parent_shelf_id is not None
