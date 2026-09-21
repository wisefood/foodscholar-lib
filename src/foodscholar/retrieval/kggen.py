"""Extended KG-Gen hybrid passage retrieval, read from the library's stores.

The scoring is the one benchmarked in `kggen/graph_code/retrieval/` (vendored
as provenance)::

    query
     |
     +-- cosine(query, passage text)                    weight 0.3
     +-- mean cosine(query, triples of a passage)       weight 0.3
     +-- PPR on the k-hop subgraph of the top-k
         entities, propagated back onto passages        weight 0.4

What differs from the reference is where the data comes from. The reference
loaded a GraphML file plus a `passages.json` and cached ~520MB of embeddings
next to them, so every deployment had to mount a matching pair of artifacts.
Here each branch reads the store that already owns the data:

    branch 1    `ChunkStore.knn_search_chunks`   (chunk embeddings)
    branch 2    `RelationStore.for_chunks`       (Layer 0 triples)
    branch 3    `RelationStore.for_entity`       (entity-entity edges)

so the retriever inherits whatever backend the storage config produced, and a
graph rebuild is visible to it immediately with nothing to re-sync.

The cost model is different too, and worth stating. The reference scored every
passage in the corpus because it had them all in memory. This scores a
*candidate pool*: `cfg.retrieval.candidate_k` chunks from the text branch,
which the other two branches then re-rank. A chunk outside that pool cannot be
retrieved however well it would have scored on the graph, which is the price of
not holding the corpus in the process.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from foodscholar.config import RetrievalConfig
from foodscholar.io.chunk import Chunk, ChunkId
from foodscholar.io.relation import Relation
from foodscholar.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from foodscholar.storage.protocols import ChunkStore, Embedder, RelationStore

log = get_logger("foodscholar.retrieval")

# Embedder ids that are deterministic toys, not semantic models. Retrieval
# with one of these returns a confidently-ranked list of nonsense, and the
# facade substitutes one whenever the real embedder fails to build — so the
# failure mode is a service that answers questions from hash collisions and
# says nothing about it. Mirrors `facade._MOCK_EMBEDDING_MODELS`.
MOCK_EMBEDDER_IDS: frozenset[str] = frozenset({"mock-embedder-v0", "hash-embedder-v0"})


@dataclass
class RetrievalHit:
    """One retrieved passage with the branch scores that ranked it.

    The breakdown is kept rather than collapsed into `score` because the three
    branches answer different questions ("is this text about the query", "does
    this passage assert something about the query", "is this passage in the
    right neighbourhood of the graph") and a caller debugging a bad ranking
    needs to see which one carried it.
    """

    chunk: Chunk
    score: float
    text_sim: float = 0.0
    triplet_sim: float = 0.0
    ppr_score: float = 0.0

    @property
    def chunk_id(self) -> ChunkId:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass
class RetrievalTrace:
    """What the retriever did, for logging and the `/qa` status payload."""

    candidates: int = 0
    relations: int = 0
    seed_entities: list[str] = field(default_factory=list)
    subgraph_nodes: int = 0
    subgraph_edges: int = 0
    branches_used: list[str] = field(default_factory=list)
    expansion_calls: int = 0
    """`RelationStore.for_entity` round-trips spent on this query."""
    relations_scored: int = 0
    """Triples that reached the triplet branch, after `max_relations`."""
    embedder_model: str = ""
    degraded: list[str] = field(default_factory=list)
    """Reasons this ranking is less trustworthy than a healthy one — surfaced
    so a caller can show it rather than presenting the result as normal."""


def _normalize(scores: dict[str, float], keys: Sequence[str]) -> dict[str, float]:
    """Min-max a branch to [0, 1] **over the whole candidate pool**.

    The three branches are on incomparable scales — a cosine in [-1, 1], a mean
    of cosines, and a PageRank mass that sums to 1 over the subgraph — so the
    configured weights only mean what they say once each branch is normalized.

    Normalizing over `keys` rather than over the scored subset is not a detail.
    The triplet and PPR branches only produce entries for candidates they can
    say something about, and a candidate they are silent on scores 0. Taking
    the min-max over just the entries present would rescale that silent
    majority away: with one chunk scored, min == max and the single chunk that
    *did* match collapses to 0.0, scoring exactly like the chunks with no
    triples at all — the branch's one real signal, deleted.

    A branch where every candidate genuinely ties still collapses to 0.0
    rather than 1.0: a branch that separates nothing should contribute
    nothing, not hand its full weight to everyone equally.
    """
    if not keys:
        return {}
    dense = {key: scores.get(key, 0.0) for key in keys}
    lo = min(dense.values())
    hi = max(dense.values())
    if hi - lo < 1e-12:
        return dict.fromkeys(dense, 0.0)
    span = hi - lo
    return {k: (v - lo) / span for k, v in dense.items()}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _triple_text(relation: Relation) -> str:
    """The triple as a sentence, for embedding.

    Surface forms are preferred over the ontology ids: `FOODON_03301710` does
    not embed into anything near the query, and the surfaces are exactly the
    strings the extractor read out of the corpus.
    """
    subject = relation.subject_surfaces[0] if relation.subject_surfaces else relation.subject_id
    obj = relation.object_surfaces[0] if relation.object_surfaces else relation.object_id
    return f"{subject} {relation.predicate} {obj}"


def _entity_label(ontology_id: str, surfaces: dict[str, str]) -> str:
    return surfaces.get(ontology_id) or ontology_id


class _EmbeddingCache:
    """Process-local LRU in front of an embedder, keyed by exact text.

    The triplet and PPR branches embed triples and entity labels on every
    query, and those strings come from a fixed corpus — the same few thousand
    triples answer question after question. Without a cache, the embedding
    step dominates the query; with one, a warm replica pays it only for text
    it has not seen.

    Only cacheable-by-construction text goes through here: triples and entity
    labels, which are corpus artifacts. The query itself is deliberately NOT
    routed through it — every query is new, so it would only evict useful
    entries.

    Thread-safe: the API serves retrieval from a threadpool, and an
    `OrderedDict` reordered from two threads at once corrupts.
    """

    def __init__(self, embedder: Embedder, capacity: int) -> None:
        self._embedder = embedder
        self._capacity = max(0, capacity)
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeddings for `texts`, in input order.

        `misses` counts texts actually sent to the embedder, so it tracks real
        work: a text repeated within one batch is encoded once and the repeats
        score as hits.
        """
        if self._capacity == 0:
            return self._embedder.embed(texts)

        out: list[list[float] | None] = [None] * len(texts)
        misses: list[str] = []
        miss_slots: dict[str, list[int]] = {}

        with self._lock:
            for i, text in enumerate(texts):
                cached = self._cache.get(text)
                if cached is not None:
                    self._cache.move_to_end(text)
                    out[i] = cached
                    self.hits += 1
                elif text in miss_slots:
                    # A duplicate within this batch: served from the one
                    # encode below, so it cost nothing and counts as a hit.
                    miss_slots[text].append(i)
                    self.hits += 1
                else:
                    miss_slots[text] = [i]
                    misses.append(text)
                    self.misses += 1

        if misses:
            # Outside the lock: this is the slow call, and holding the lock
            # across it would serialize every concurrent query on the model.
            vectors = self._embedder.embed(misses)
            with self._lock:
                for text, vec in zip(misses, vectors, strict=False):
                    for slot in miss_slots[text]:
                        out[slot] = vec
                    self._cache[text] = vec
                    self._cache.move_to_end(text)
                while len(self._cache) > self._capacity:
                    self._cache.popitem(last=False)

        return [vec if vec is not None else [] for vec in out]


class KGGenRetriever:
    """Hybrid retrieval over Layer 0. Stateless across queries.

    Holds no index and no cache: every query is a bounded number of store
    round-trips. That is the deliberate trade against the reference
    implementation, which paid a multi-minute warm-up and ~520MB of resident
    cache to make each query fast. A service that answers a question every few
    seconds does not amortize that, and a stateless retriever is one that a
    replica can serve immediately after boot.
    """

    def __init__(
        self,
        *,
        chunk_store: ChunkStore,
        relation_store: RelationStore,
        embedder: Embedder,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.chunk_store = chunk_store
        self.relation_store = relation_store
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        # Corpus text (triples, entity labels) goes through the cache; the
        # query never does. See `_EmbeddingCache`.
        self._corpus_embed = _EmbeddingCache(embedder, self.config.embed_cache_size)

    # ---------------------------------------------------------------- query
    def retrieve(
        self,
        query: str,
        *,
        k: int | None = None,
    ) -> tuple[list[RetrievalHit], RetrievalTrace]:
        """Rank passages for `query`, best first.

        Returns the hits and a trace of how they were found. The trace is
        returned rather than logged-and-dropped because the caller (a QA
        service) surfaces it as retrieval status.
        """
        cfg = self.config
        top_k = k or cfg.top_k
        trace = RetrievalTrace()

        trace.embedder_model = str(getattr(self.embedder, "model_id", "") or "")
        if trace.embedder_model in MOCK_EMBEDDER_IDS:
            # Not fatal — `in_memory()` demos and tests run on toy embedders
            # deliberately — but in a deployment it means the real embedder
            # failed to build and every branch here is scoring hashes.
            trace.degraded.append(f"mock embedder ({trace.embedder_model})")
            log.warning(
                "retrieval.mock_embedder",
                model=trace.embedder_model,
                note="rankings are not semantically meaningful; "
                "install 'foodscholar[annotate]' or attach a real embedder",
            )

        query = (query or "").strip()
        if not query:
            return [], trace

        q_vec = self.embedder.embed([query])[0]

        # -- branch 1: text similarity ----------------------------------
        # Also the candidate generator: everything downstream re-ranks this
        # pool rather than scanning the corpus.
        knn = self.chunk_store.knn_search_chunks(q_vec, k=cfg.candidate_k)
        if not knn:
            return [], trace

        text_raw = {cid: score for cid, score in knn}
        candidate_ids = list(text_raw)
        trace.candidates = len(candidate_ids)
        trace.branches_used.append("text")

        # -- branch 2: mean triplet similarity --------------------------
        relations = self._relations_for(candidate_ids)
        trace.relations = len(relations)
        triplet_raw = self._score_triplets(q_vec, relations, candidate_ids)
        if triplet_raw:
            trace.branches_used.append("triplet")

        # -- branch 3: PPR over the entity subgraph ---------------------
        ppr_raw = self._score_ppr(q_vec, relations, candidate_ids, trace)
        if ppr_raw:
            trace.branches_used.append("ppr")

        # -- combine ----------------------------------------------------
        text_n = _normalize(text_raw, candidate_ids)
        triplet_n = _normalize(triplet_raw, candidate_ids)
        ppr_n = _normalize(ppr_raw, candidate_ids)

        chunks_with_triples = {
            cid for rel in relations for cid in rel.chunk_ids
        }

        final: dict[str, float] = {}
        for cid in candidate_ids:
            if cfg.graph_connected_only and cid not in chunks_with_triples:
                continue
            final[cid] = (
                cfg.w_text * text_n.get(cid, 0.0)
                + cfg.w_triplet * triplet_n.get(cid, 0.0)
                + cfg.w_ppr * ppr_n.get(cid, 0.0)
            )

        ranked = sorted(
            final,
            key=lambda c: (
                final[c],
                text_n.get(c, 0.0),
                triplet_n.get(c, 0.0),
                ppr_n.get(c, 0.0),
            ),
            reverse=True,
        )[:top_k]

        # One store call for the payloads, and only for what survived the cut.
        chunks = {c.chunk_id: c for c in self.chunk_store.get_many(list(ranked))}

        hits = [
            RetrievalHit(
                chunk=chunks[cid],
                score=final[cid],
                text_sim=text_n.get(cid, 0.0),
                triplet_sim=triplet_n.get(cid, 0.0),
                ppr_score=ppr_n.get(cid, 0.0),
            )
            for cid in ranked
            if cid in chunks
        ]

        trace.relations_scored = min(trace.relations, cfg.max_relations)
        log.info(
            "retrieval.kggen",
            query_chars=len(query),
            candidates=trace.candidates,
            relations=trace.relations,
            relations_scored=trace.relations_scored,
            subgraph_nodes=trace.subgraph_nodes,
            expansion_calls=trace.expansion_calls,
            embed_hits=self._corpus_embed.hits,
            embed_misses=self._corpus_embed.misses,
            hits=len(hits),
            branches=",".join(trace.branches_used),
        )
        return hits, trace

    # ------------------------------------------------------------ branches
    def _relations_for(self, candidate_ids: list[ChunkId]) -> list[Relation]:
        """Layer 0 triples extracted from the candidate chunks.

        `for_chunks` is contractually one round-trip, which is why the pool is
        passed whole rather than looped over.
        """
        try:
            return list(self.relation_store.for_chunks(candidate_ids))
        except Exception as exc:  # pragma: no cover - backend shape
            # A corpus with no Layer 0 pass is a normal deployment state, not
            # an error: the text branch alone still ranks. Degrade, don't fail.
            log.warning("retrieval.relations_unavailable", error=str(exc))
            return []

    def _score_triplets(
        self,
        q_vec: list[float],
        relations: list[Relation],
        candidate_ids: list[ChunkId],
    ) -> dict[str, float]:
        """Mean cosine between the query and each chunk's triples.

        Triple embeddings are computed per query rather than stored. The brief
        left "do relations get embeddings, and where" open precisely so this
        call site could answer it, and on a candidate pool the answer is that
        they do not need to be: this embeds at most the triples of
        `candidate_k` chunks, in one batch, and adds no field to the relation
        index that a rebuild would then have to maintain.
        """
        if not relations:
            return {}

        candidates = set(candidate_ids)
        # `for_chunks` returns best-supported first, so the cap keeps the
        # triples that carry the most corpus evidence and drops the
        # single-mention tail that would not move a ranking anyway.
        scored = relations[: self.config.max_relations]
        texts = [_triple_text(rel) for rel in scored]
        try:
            vectors = self._corpus_embed.embed(texts)
        except Exception as exc:  # pragma: no cover - model shape
            log.warning("retrieval.triplet_embed_failed", error=str(exc))
            return {}

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for rel, vec in zip(scored, vectors, strict=False):
            sim = _cosine(q_vec, vec)
            for cid in rel.chunk_ids:
                if cid not in candidates:
                    continue
                sums[cid] = sums.get(cid, 0.0) + sim
                counts[cid] = counts.get(cid, 0) + 1

        return {cid: sums[cid] / counts[cid] for cid in sums}

    def _score_ppr(
        self,
        q_vec: list[float],
        relations: list[Relation],
        candidate_ids: list[ChunkId],
        trace: RetrievalTrace,
    ) -> dict[str, float]:
        """Personalized PageRank over the entity graph, propagated to chunks.

        This is the branch that makes the retrieval graph-aware rather than a
        two-headed vector search: a passage scores because it sits near the
        query's entities in the relation graph, even when its own text does
        not read like the query.
        """
        if not relations:
            return {}

        import networkx as nx

        # Surface labels for every endpoint we have seen, so seeds can be
        # picked by embedding something human-readable.
        surfaces: dict[str, str] = {}
        for rel in relations:
            if rel.subject_surfaces:
                surfaces.setdefault(rel.subject_id, rel.subject_surfaces[0])
            if rel.object_surfaces:
                surfaces.setdefault(rel.object_id, rel.object_surfaces[0])

        cfg = self.config

        # Seeds: the entities of the candidate pool nearest the query.
        entity_ids = sorted({rel.subject_id for rel in relations} | {rel.object_id for rel in relations})
        if not entity_ids:
            return {}

        labels = [_entity_label(eid, surfaces) for eid in entity_ids]
        try:
            entity_vecs = self._corpus_embed.embed(labels)
        except Exception as exc:  # pragma: no cover - model shape
            log.warning("retrieval.entity_embed_failed", error=str(exc))
            return {}

        scored_entities = sorted(
            ((eid, _cosine(q_vec, vec)) for eid, vec in zip(entity_ids, entity_vecs, strict=False)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        seeds = [eid for eid, score in scored_entities[: cfg.seed_entities] if score > 0.0]
        if not seeds:
            return {}
        trace.seed_entities = [_entity_label(eid, surfaces) for eid in seeds]

        # The subgraph: the candidate pool's relations, plus `subgraph_depth`
        # hops out from the seeds. The expansion is what lets a passage be
        # reached through an entity the query never mentioned.
        edges = list(relations)
        edges.extend(self._expand(seeds, depth=cfg.subgraph_depth, trace=trace))

        graph = nx.DiGraph()
        chunk_links: dict[str, set[str]] = {}
        for rel in edges:
            weight = float(max(rel.mention_count, 1))
            if graph.has_edge(rel.subject_id, rel.object_id):
                graph[rel.subject_id][rel.object_id]["weight"] += weight
            else:
                graph.add_edge(rel.subject_id, rel.object_id, weight=weight)
            for cid in rel.chunk_ids:
                chunk_links.setdefault(rel.subject_id, set()).add(cid)
                chunk_links.setdefault(rel.object_id, set()).add(cid)

        trace.subgraph_nodes = graph.number_of_nodes()
        trace.subgraph_edges = graph.number_of_edges()
        if graph.number_of_nodes() == 0:
            return {}

        personalization = {node: 0.0 for node in graph.nodes}
        for eid in seeds:
            if eid in personalization:
                personalization[eid] = 1.0
        if not any(personalization.values()):
            return {}

        try:
            ranks = nx.pagerank(
                graph,
                alpha=cfg.ppr_alpha,
                personalization=personalization,
                max_iter=cfg.ppr_max_iter,
                weight="weight",
            )
        except nx.PowerIterationFailedConvergence:
            # A ranking that did not converge is not a reason to drop the
            # whole query — the other two branches still carry it.
            log.warning("retrieval.ppr_did_not_converge", nodes=graph.number_of_nodes())
            return {}

        # Propagate entity mass onto the candidate passages that evidence them.
        candidates = set(candidate_ids)
        chunk_scores: dict[str, float] = {}
        for eid, mass in ranks.items():
            for cid in chunk_links.get(eid, ()):
                if cid in candidates:
                    chunk_scores[cid] = chunk_scores.get(cid, 0.0) + mass
        return chunk_scores

    def _expand(
        self,
        seeds: list[str],
        *,
        depth: int,
        trace: RetrievalTrace,
    ) -> list[Relation]:
        """Relations within `depth` hops of the seeds.

        One store call per frontier entity per hop, which is why
        `cfg.subgraph_depth` defaults to 1 rather than the reference's 5: the
        reference walked an in-memory graph where a hop was free, and here
        each hop is a round-trip.

        `cfg.max_expansion_calls` is the ceiling that makes a deeper walk safe
        to configure. Hitting it stops the walk and scores whatever subgraph
        has been gathered — a slightly worse PPR branch is the right failure
        here, not a query that issues a thousand store calls.
        """
        if depth <= 0:
            return []

        budget = self.config.max_expansion_calls
        found: dict[str, Relation] = {}
        seen: set[str] = set(seeds)
        frontier = list(seeds)

        for _ in range(depth):
            next_frontier: list[str] = []
            for eid in frontier:
                if trace.expansion_calls >= budget:
                    log.info(
                        "retrieval.expansion_budget_exhausted",
                        budget=budget,
                        relations=len(found),
                    )
                    return list(found.values())
                trace.expansion_calls += 1
                try:
                    neighbours = self.relation_store.for_entity(
                        eid, direction="both", k=self.config.expand_k
                    )
                except Exception as exc:  # pragma: no cover - backend shape
                    log.warning("retrieval.expand_failed", entity=eid, error=str(exc))
                    continue
                for rel in neighbours:
                    found[rel.relation_id] = rel
                    for endpoint in (rel.subject_id, rel.object_id):
                        if endpoint not in seen:
                            seen.add(endpoint)
                            next_frontier.append(endpoint)
            frontier = next_frontier
            if not frontier:
                break

        return list(found.values())
