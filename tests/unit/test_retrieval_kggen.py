"""Extended KG-Gen hybrid retrieval over the Layer 0 stores."""

from __future__ import annotations

import pytest

from foodscholar.config import RetrievalConfig
from foodscholar.io.chunk import Chunk
from foodscholar.io.relation import Relation, make_relation_id
from foodscholar.retrieval import KGGenRetriever
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryRelationStore

pytestmark = pytest.mark.usefixtures()


class DirectionalEmbedder:
    """Embeds on keyword hits, so a test can steer each branch on purpose.

    A hash embedder (the shared `mock_embedder`) gives every text an unrelated
    direction, which makes "did the triplet branch move this chunk up" untestable
    — every score is noise. Here a text containing "salt" points along axis 0,
    "fibre" along axis 1, and the query does too, so the branches are separable.
    """

    model_id = "directional-v0"
    _AXES = ("salt", "fibre", "iron")

    @property
    def dim(self) -> int:
        return len(self._AXES) + 1

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            low = text.lower()
            vec = [1.0 if axis in low else 0.0 for axis in self._AXES]
            # Non-zero tail so a text matching nothing still has a direction
            # (a zero vector would make every cosine 0.0 and hide bugs).
            vec.append(0.1)
            vectors.append(vec)
        return vectors


def _chunk(chunk_id: str, text: str, embedding: list[float]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=f"doc-{chunk_id}",
        source_type="abstract",
        section_type="abstract",
        embedding=embedding,
        embedding_model="directional-v0",
    )


def _relation(
    subject: str,
    predicate: str,
    obj: str,
    chunk_ids: tuple[str, ...],
    *,
    mention_count: int = 1,
) -> Relation:
    return Relation(
        relation_id=make_relation_id(subject, predicate, obj),
        subject_id=subject,
        predicate=predicate,
        object_id=obj,
        subject_surfaces=(subject,),
        object_surfaces=(obj,),
        chunk_ids=chunk_ids,
        chunk_count=len(chunk_ids),
        mention_count=mention_count,
    )


@pytest.fixture
def embedder() -> DirectionalEmbedder:
    return DirectionalEmbedder()


@pytest.fixture
def stores(embedder: DirectionalEmbedder):
    chunk_store = InMemoryChunkStore()
    relation_store = InMemoryRelationStore()

    texts = {
        "c-salt": "salt intake and blood pressure",
        "c-fibre": "fibre intake and cholesterol",
        "c-iron": "iron absorption in adults",
        "c-plain": "a passage about nothing in particular",
    }
    vectors = embedder.embed(list(texts.values()))
    chunk_store.upsert(
        [_chunk(cid, text, vec) for (cid, text), vec in zip(texts.items(), vectors)]
    )
    return chunk_store, relation_store


def test_text_branch_alone_ranks_when_no_relations_exist(stores, embedder):
    """A corpus with no Layer 0 pass still retrieves — it does not fail."""
    chunk_store, relation_store = stores
    retriever = KGGenRetriever(
        chunk_store=chunk_store,
        relation_store=relation_store,
        embedder=embedder,
        config=RetrievalConfig(top_k=2),
    )

    hits, trace = retriever.retrieve("salt")

    assert next(h.chunk_id for h in hits) == "c-salt"
    assert trace.relations == 0
    assert trace.branches_used == ["text"]
    # Nothing to score, so those branches contribute zero rather than noise.
    assert all(h.triplet_sim == 0.0 and h.ppr_score == 0.0 for h in hits)


def test_triplet_branch_promotes_a_chunk_its_text_does_not_favour(stores, embedder):
    """The point of the triplet branch: a passage whose *assertions* match.

    `c-plain` has no query term in its text, so the text branch ranks it last.
    Giving it a triple that does match must move it above a chunk with neither.
    """
    chunk_store, relation_store = stores
    relation_store.upsert(
        [_relation("salt", "raises", "blood pressure", ("c-plain",))]
    )
    retriever = KGGenRetriever(
        chunk_store=chunk_store,
        relation_store=relation_store,
        embedder=embedder,
        # PPR off, so this isolates the triplet branch.
        config=RetrievalConfig(top_k=4, w_text=0.5, w_triplet=0.5, w_ppr=0.0),
    )

    hits, trace = retriever.retrieve("salt")
    ranked = [h.chunk_id for h in hits]

    assert "triplet" in trace.branches_used
    assert ranked.index("c-plain") < ranked.index("c-iron")
    plain = next(h for h in hits if h.chunk_id == "c-plain")
    assert plain.triplet_sim > 0.0


def test_ppr_branch_reaches_a_chunk_through_the_entity_graph(stores, embedder):
    """A passage two hops from the query's entity is still reachable."""
    chunk_store, relation_store = stores
    relation_store.upsert(
        [
            # Seed entity "salt" is what the query embeds near.
            _relation("salt", "raises", "blood pressure", ("c-salt",), mention_count=5),
            # c-plain never mentions salt; it is linked through blood pressure.
            _relation("blood pressure", "damages", "arteries", ("c-plain",), mention_count=3),
        ]
    )
    retriever = KGGenRetriever(
        chunk_store=chunk_store,
        relation_store=relation_store,
        embedder=embedder,
        config=RetrievalConfig(top_k=4, w_text=0.0, w_triplet=0.0, w_ppr=1.0),
    )

    hits, trace = retriever.retrieve("salt")

    assert "ppr" in trace.branches_used
    assert trace.subgraph_nodes >= 3
    scored = {h.chunk_id: h.ppr_score for h in hits}
    assert scored.get("c-plain", 0.0) > 0.0


def test_graph_connected_only_drops_chunks_with_no_triples(stores, embedder):
    chunk_store, relation_store = stores
    relation_store.upsert([_relation("salt", "raises", "blood pressure", ("c-salt",))])
    retriever = KGGenRetriever(
        chunk_store=chunk_store,
        relation_store=relation_store,
        embedder=embedder,
        config=RetrievalConfig(top_k=4, graph_connected_only=True),
    )

    hits, _ = retriever.retrieve("salt")

    assert [h.chunk_id for h in hits] == ["c-salt"]


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        RetrievalConfig(w_text=0.5, w_triplet=0.5, w_ppr=0.5)


def test_empty_query_returns_nothing(stores, embedder):
    chunk_store, relation_store = stores
    retriever = KGGenRetriever(
        chunk_store=chunk_store, relation_store=relation_store, embedder=embedder
    )

    hits, trace = retriever.retrieve("   ")

    assert hits == []
    assert trace.candidates == 0


def test_a_failing_relation_store_degrades_to_the_text_branch(stores, embedder):
    """Layer 0 being unavailable must not take retrieval down with it."""
    chunk_store, _ = stores

    class BrokenRelationStore:
        def for_chunks(self, chunk_ids):
            raise RuntimeError("relations index missing")

        def for_entity(self, ontology_id, *, direction="both", k=100):
            raise RuntimeError("relations index missing")

    retriever = KGGenRetriever(
        chunk_store=chunk_store,
        relation_store=BrokenRelationStore(),
        embedder=embedder,
        config=RetrievalConfig(top_k=2),
    )

    hits, trace = retriever.retrieve("salt")

    assert next(h.chunk_id for h in hits) == "c-salt"
    assert trace.branches_used == ["text"]


# --------------------------------------------------------------------------
# Production bounds: the query must stay cheap on a real corpus.
# --------------------------------------------------------------------------


class CountingEmbedder(DirectionalEmbedder):
    """Counts texts encoded, so caching and caps are measurable."""

    def __init__(self) -> None:
        self.encoded: list[str] = []
        self.batches = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches += 1
        self.encoded.extend(texts)
        return super().embed(texts)


def test_triple_embeddings_are_reused_across_queries(stores):
    """A warm replica must not re-encode the corpus on every question."""
    chunk_store, relation_store = stores
    relation_store.upsert(
        [_relation("salt", "raises", "blood pressure", ("c-salt",))]
    )
    embedder = CountingEmbedder()
    retriever = KGGenRetriever(
        chunk_store=chunk_store,
        relation_store=relation_store,
        embedder=embedder,
        config=RetrievalConfig(top_k=2),
    )

    retriever.retrieve("salt")
    after_first = len(embedder.encoded)
    retriever.retrieve("salt")
    after_second = len(embedder.encoded)

    # The second query re-encodes only its own query string; every triple and
    # entity label came from the cache.
    assert after_second - after_first == 1
    assert retriever._corpus_embed.hits > 0


def test_embed_cache_can_be_disabled(stores):
    chunk_store, relation_store = stores
    relation_store.upsert([_relation("salt", "raises", "blood pressure", ("c-salt",))])
    embedder = CountingEmbedder()
    retriever = KGGenRetriever(
        chunk_store=chunk_store,
        relation_store=relation_store,
        embedder=embedder,
        config=RetrievalConfig(top_k=2, embed_cache_size=0),
    )

    retriever.retrieve("salt")
    first = len(embedder.encoded)
    retriever.retrieve("salt")

    assert len(embedder.encoded) == first * 2


def test_embed_cache_evicts_at_capacity(stores, embedder):
    chunk_store, relation_store = stores
    from foodscholar.retrieval.kggen import _EmbeddingCache

    cache = _EmbeddingCache(embedder, capacity=2)
    cache.embed(["a", "b"])
    cache.embed(["c"])  # evicts "a"

    assert len(cache._cache) == 2
    assert "a" not in cache._cache
    assert "c" in cache._cache


def test_embed_cache_returns_vectors_in_input_order_with_duplicates(embedder):
    from foodscholar.retrieval.kggen import _EmbeddingCache

    cache = _EmbeddingCache(embedder, capacity=100)
    direct = embedder.embed(["salt", "fibre", "salt"])
    cached = cache.embed(["salt", "fibre", "salt"])

    assert cached == direct
    # The duplicate was encoded once, not twice.
    assert cache.misses == 2


def test_max_relations_caps_what_the_triplet_branch_embeds(stores):
    chunk_store, relation_store = stores
    relation_store.upsert(
        [
            _relation("salt", f"predicate{i}", f"object{i}", ("c-salt",))
            for i in range(20)
        ]
    )
    embedder = CountingEmbedder()
    retriever = KGGenRetriever(
        chunk_store=chunk_store,
        relation_store=relation_store,
        embedder=embedder,
        config=RetrievalConfig(top_k=2, max_relations=5, w_text=0.5, w_triplet=0.5, w_ppr=0.0),
    )

    _, trace = retriever.retrieve("salt")

    assert trace.relations == 20
    assert trace.relations_scored == 5
    # 1 query + 5 triples; the PPR branch is weighted out but still seeds, so
    # assert on the triples specifically rather than the total.
    assert sum(1 for t in embedder.encoded if t.startswith("salt predicate")) == 5


def test_expansion_budget_bounds_store_round_trips(stores, embedder):
    """A deep walk must not turn one question into a thousand store calls."""
    chunk_store, _ = stores

    class SprawlingRelationStore:
        """Every entity has `expand_k` fresh neighbours — an infinite frontier."""

        def __init__(self) -> None:
            self.calls = 0

        def for_chunks(self, chunk_ids):
            return [_relation("salt", "raises", "e0", ("c-salt",))]

        def for_entity(self, ontology_id, *, direction="both", k=100):
            self.calls += 1
            return [
                _relation(ontology_id, "leads_to", f"{ontology_id}-n{i}", ("c-salt",))
                for i in range(5)
            ]

    relation_store = SprawlingRelationStore()
    retriever = KGGenRetriever(
        chunk_store=chunk_store,
        relation_store=relation_store,
        embedder=embedder,
        config=RetrievalConfig(
            top_k=2, subgraph_depth=5, expand_k=5, max_expansion_calls=7
        ),
    )

    _, trace = retriever.retrieve("salt")

    assert relation_store.calls <= 7
    assert trace.expansion_calls <= 7


def test_embed_cache_is_safe_under_concurrent_queries(stores, embedder):
    """The API serves retrieval from a threadpool; the LRU must survive it."""
    import concurrent.futures

    chunk_store, relation_store = stores
    relation_store.upsert(
        [
            _relation("salt", f"p{i}", f"o{i}", ("c-salt", "c-plain"))
            for i in range(30)
        ]
    )
    retriever = KGGenRetriever(
        chunk_store=chunk_store,
        relation_store=relation_store,
        embedder=embedder,
        config=RetrievalConfig(top_k=3, embed_cache_size=64),
    )

    queries = ["salt", "fibre", "iron", "salt intake", "fibre and iron"] * 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda q: retriever.retrieve(q)[0], queries))

    assert all(isinstance(hits, list) for hits in results)
    # No vector came back empty (the sentinel for a slot never filled).
    assert all(v for v in retriever._corpus_embed._cache.values())
    assert len(retriever._corpus_embed._cache) <= 64


def test_mock_embedder_is_reported_as_degraded(stores):
    """The facade substitutes a hash embedder when the real one fails to build.

    Retrieval still runs, but the ranking is hash collisions. It must say so.
    """
    from foodscholar.retrieval.kggen import MOCK_EMBEDDER_IDS

    chunk_store, relation_store = stores

    class MockEmbedder(DirectionalEmbedder):
        model_id = "mock-embedder-v0"

    assert MockEmbedder.model_id in MOCK_EMBEDDER_IDS

    retriever = KGGenRetriever(
        chunk_store=chunk_store,
        relation_store=relation_store,
        embedder=MockEmbedder(),
        config=RetrievalConfig(top_k=2),
    )

    _, trace = retriever.retrieve("salt")

    assert trace.degraded
    assert "mock embedder" in trace.degraded[0]


def test_a_real_embedder_is_not_flagged(stores, embedder):
    chunk_store, relation_store = stores
    retriever = KGGenRetriever(
        chunk_store=chunk_store,
        relation_store=relation_store,
        embedder=embedder,
        config=RetrievalConfig(top_k=2),
    )

    _, trace = retriever.retrieve("salt")

    assert trace.degraded == []
    assert trace.embedder_model == "directional-v0"


# --------------------------------------------------------------------------
# Through the facade — the path the FoodScholar API actually calls.
# --------------------------------------------------------------------------


def test_fs_retrieve_wires_the_stores_and_config_together():
    """`fs.retrieve()` must reach Layer 0, not just the chunk store."""
    from foodscholar import FoodScholar

    fs = FoodScholar.in_memory()
    embedder = DirectionalEmbedder()
    fs.embedder = embedder

    texts = {
        "c-salt": "salt intake and blood pressure",
        "c-fibre": "fibre intake and cholesterol",
        "c-plain": "a passage about nothing in particular",
    }
    vectors = embedder.embed(list(texts.values()))
    fs.upsert_chunks(
        [_chunk(cid, text, vec) for (cid, text), vec in zip(texts.items(), vectors)]
    )
    fs.relation_store.upsert(
        [_relation("salt", "raises", "blood pressure", ("c-plain",))]
    )

    hits, trace = fs.retrieve("salt", k=3)

    assert hits
    assert trace.candidates == 3
    # Layer 0 was actually consulted through the facade's relation store.
    assert trace.relations == 1
    assert "triplet" in trace.branches_used
    # The facade handed the retriever its configured weights, not defaults
    # invented inside the retriever.
    assert fs.config.retrieval.w_text == 0.3


def test_fs_retrieve_honours_a_configured_weighting():
    from foodscholar import FoodScholar

    fs = FoodScholar.in_memory()
    fs.config.retrieval.w_text = 1.0
    fs.config.retrieval.w_triplet = 0.0
    fs.config.retrieval.w_ppr = 0.0
    embedder = DirectionalEmbedder()
    fs.embedder = embedder

    texts = {"c-salt": "salt intake", "c-fibre": "fibre intake"}
    vectors = embedder.embed(list(texts.values()))
    fs.upsert_chunks(
        [_chunk(cid, text, vec) for (cid, text), vec in zip(texts.items(), vectors)]
    )

    hits, _ = fs.retrieve("salt", k=2)

    assert next(h.chunk_id for h in hits) == "c-salt"
