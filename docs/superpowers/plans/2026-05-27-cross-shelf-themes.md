# Cross-Shelf Themes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-shelf Layer B pipeline with a hybrid global/per-shelf design. Pass 1 (similarity) runs once over the whole attached corpus so themes are first-class cross-shelf objects; Pass 2 (relatedness) stays per-shelf since entity coherence is sharper within a shelf's scope.

**Architecture:** A new `build_global_similarity_candidates(chunk_store, attached_chunks, cfg)` runs Pass 1 over ~6,290 attached chunks using ES kNN search (one call per chunk, ES already indexes the vectors). Pass 2 stays per-shelf via the existing `build_shelf_relatedness_candidates`. A new `merge_global_and_local_candidates` step assembles cross-shelf themes from the union, with `shelf_ids = union of member chunks' shelf_ids`. `build_layer_b` is rewritten around this two-tier flow. The `Theme` schema needs no field changes (`shelf_ids` is already a list); we add a new `discovery_pass` literal `"global_similarity"` to distinguish.

**Tech Stack:** Python 3.11, python-igraph (Leiden clustering), pydantic v2, Elasticsearch 9.x kNN search, Neo4j 5.x for theme persistence, pytest.

---

## Background reading (for the implementing engineer)

Before starting, read these in order — total ~30 minutes:

1. `layer_b_construction_brief.md` — the original v1 design. §5 explicitly defers cross-shelf to v2; §6.4 is the per-shelf orchestrator we're replacing; §10 is the audit invariants you must keep passing.
2. `src/foodscholar/layer_b/builder.py` — the existing per-shelf orchestrator. Read top-to-bottom. The current `build_layer_b` at L322 is the function to rewrite.
3. `src/foodscholar/layer_b/semantic_graph.py` — Pass 1 (similarity). Today it does in-memory kNN on a shelf's chunks. We're keeping the *output* shape (igraph) but feeding it ES-kNN edges instead.
4. `src/foodscholar/layer_b/relatedness_graph.py` — Pass 2 (relatedness). Unchanged behavior, still runs per-shelf.
5. `src/foodscholar/layer_b/merge.py` — Existing merge step. Today it pairs sim and rel candidates within one shelf; we'll generalize so a global-similarity candidate can pair with multiple per-shelf relatedness candidates.
6. `src/foodscholar/storage/elastic.py` — ES backend. We'll add a `knn_search_chunks(...)` method that issues an ES kNN query and returns `(chunk_id, score)` tuples.

## File structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/foodscholar/storage/protocols.py` | Modify | Add `knn_search_chunks` to `ChunkStore` protocol |
| `src/foodscholar/storage/elastic.py` | Modify | Implement `knn_search_chunks` using ES `knn` search |
| `src/foodscholar/storage/in_memory.py` | Modify | Implement `knn_search_chunks` using numpy (used by tests) |
| `src/foodscholar/io/graph.py` | Modify | Add `"global_similarity"` to `Theme.discovery_pass` literal |
| `src/foodscholar/layer_b/models.py` | Modify | Add `"global_similarity"` to `DiscoveryPass` and to `ThemeCandidate.pass_name` |
| `src/foodscholar/layer_b/semantic_graph.py` | Modify | Add `build_global_similarity_graph(chunk_ids, chunk_store, cfg)` — builds graph from ES kNN |
| `src/foodscholar/layer_b/builder.py` | Modify | Replace `build_layer_b` orchestrator; add `build_global_similarity_candidates` and `merge_global_and_local_candidates` |
| `src/foodscholar/layer_b/merge.py` | Modify | Generalize `merge_candidates` to handle one sim-candidate-set crossed with many rel-candidate-sets |
| `src/foodscholar/layer_b/persist.py` | No change | Already handles `themes[i].shelf_ids: list` correctly |
| `src/foodscholar/layer_b/audit.py` | Modify | Update audit invariants for cross-shelf (shelf-coverage gate, themed-shelves counting) |
| `src/foodscholar/config.py` | Modify | Add `LayerBConfig.global_similarity_max_chunks` knob (safety cap) |
| `tests/unit/test_layer_b_semantic_graph.py` | Modify | Add tests for `build_global_similarity_graph` |
| `tests/unit/test_layer_b_builder.py` | Modify | Update orchestrator tests for new flow |
| `tests/unit/test_layer_b_merge.py` | Modify | Update merge tests for the global-cross-local case |
| `tests/unit/test_chunk_store_knn.py` | Create | Unit-test the new `knn_search_chunks` against `InMemoryChunkStore` |
| `tests/integration/test_layer_b_pipeline.py` | Modify | Update the integration test to verify cross-shelf shelf_ids on themes |

## Notes on the test approach

- Existing tests use `InMemoryChunkStore` to avoid ES dependency. We add the same `knn_search_chunks` implementation there so unit tests can drive the new flow without ES.
- The ES-side `knn_search_chunks` gets one integration test against the real ES (gated by the existing integration-test fixture).
- All new logic is exercised by unit tests; the integration test verifies wiring only.

---

## Task 1: Add `knn_search_chunks` to the ChunkStore protocol

**Files:**
- Modify: `src/foodscholar/storage/protocols.py`

- [ ] **Step 1: Write the failing test**

Test the protocol surface, not an implementation — we just verify the method exists on the protocol.

Create or extend `tests/unit/test_chunk_store_knn.py`:

```python
"""Tests for ChunkStore.knn_search_chunks across implementations."""
from __future__ import annotations

import pytest

from foodscholar.storage.protocols import ChunkStore


def test_chunk_store_protocol_has_knn_search_chunks():
    """ChunkStore protocol must expose knn_search_chunks for Layer B global pass."""
    assert hasattr(ChunkStore, "knn_search_chunks")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_chunk_store_knn.py::test_chunk_store_protocol_has_knn_search_chunks -v
```
Expected: FAIL with `AssertionError: assert False` (or AttributeError).

- [ ] **Step 3: Add the protocol method**

In `src/foodscholar/storage/protocols.py`, find the `ChunkStore` protocol class. Add this method alongside `search`:

```python
    def knn_search_chunks(
        self,
        query_vector: list[float],
        *,
        k: int,
        exclude_ids: list[ChunkId] | None = None,
        candidate_ids: list[ChunkId] | None = None,
    ) -> list[tuple[ChunkId, float]]:
        """Return the top-k cosine-nearest chunks to `query_vector`.

        - `exclude_ids`: chunks to omit from the result (typically `[query_id]`).
        - `candidate_ids`: if provided, restrict the search to these ids
          (used to constrain the global similarity pass to attached chunks).

        Returns `[(chunk_id, cosine_score), ...]` sorted by score descending.
        Implementations may skip the score sort if their backend returns it
        unordered (callers re-sort).
        """
        ...
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_chunk_store_knn.py::test_chunk_store_protocol_has_knn_search_chunks -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_chunk_store_knn.py src/foodscholar/storage/protocols.py
git commit -m "M5 (cross-shelf/p1): add knn_search_chunks to ChunkStore protocol"
```

---

## Task 2: Implement `knn_search_chunks` for `InMemoryChunkStore`

**Files:**
- Modify: `src/foodscholar/storage/in_memory.py`
- Modify: `tests/unit/test_chunk_store_knn.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_chunk_store_knn.py`:

```python
import numpy as np

from foodscholar.io.chunk import Chunk
from foodscholar.storage.in_memory import InMemoryChunkStore


def _make_chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=f"text for {cid}",
        source_doc_id="doc1",
        source_type="textbook",
        section_type="body",
        embedding=vec,
        embedding_model="test-model",
    )


def test_in_memory_knn_search_returns_top_k_by_cosine():
    """Three chunks: query is closest to A, then B, then C. k=2 returns A,B."""
    store = InMemoryChunkStore()
    store.upsert([
        _make_chunk("A", [1.0, 0.0, 0.0]),
        _make_chunk("B", [0.8, 0.6, 0.0]),
        _make_chunk("C", [0.0, 0.0, 1.0]),
    ])
    result = store.knn_search_chunks(
        query_vector=[1.0, 0.0, 0.0], k=2, exclude_ids=None, candidate_ids=None,
    )
    assert [cid for cid, _ in result] == ["A", "B"]
    # Cosine scores should be in [-1, 1] and sorted desc.
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_in_memory_knn_search_excludes_ids():
    """exclude_ids drops the query chunk from results."""
    store = InMemoryChunkStore()
    store.upsert([
        _make_chunk("A", [1.0, 0.0, 0.0]),
        _make_chunk("B", [0.8, 0.6, 0.0]),
    ])
    result = store.knn_search_chunks(
        query_vector=[1.0, 0.0, 0.0], k=2, exclude_ids=["A"], candidate_ids=None,
    )
    assert [cid for cid, _ in result] == ["B"]


def test_in_memory_knn_search_restricts_to_candidate_ids():
    """candidate_ids restricts the search universe."""
    store = InMemoryChunkStore()
    store.upsert([
        _make_chunk("A", [1.0, 0.0, 0.0]),
        _make_chunk("B", [0.8, 0.6, 0.0]),
        _make_chunk("C", [0.0, 0.0, 1.0]),
    ])
    result = store.knn_search_chunks(
        query_vector=[1.0, 0.0, 0.0], k=10,
        exclude_ids=None, candidate_ids=["B", "C"],
    )
    ids = [cid for cid, _ in result]
    assert "A" not in ids
    assert set(ids) == {"B", "C"}


def test_in_memory_knn_search_handles_chunks_without_embeddings():
    """Chunks with embedding=None are silently skipped."""
    store = InMemoryChunkStore()
    store.upsert([
        _make_chunk("A", [1.0, 0.0, 0.0]),
        Chunk(
            chunk_id="B", text="no vec", source_doc_id="doc1",
            source_type="textbook", section_type="body",
            embedding=None, embedding_model=None,
        ),
    ])
    result = store.knn_search_chunks(
        query_vector=[1.0, 0.0, 0.0], k=10, exclude_ids=None, candidate_ids=None,
    )
    assert [cid for cid, _ in result] == ["A"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_chunk_store_knn.py -v
```
Expected: 4 FAILS with `AttributeError: 'InMemoryChunkStore' object has no attribute 'knn_search_chunks'`.

- [ ] **Step 3: Implement on InMemoryChunkStore**

Open `src/foodscholar/storage/in_memory.py`, find the `InMemoryChunkStore` class, add this method (the imports `import numpy as np` and the `ChunkId` type may already be at module top — add only what's missing):

```python
    def knn_search_chunks(
        self,
        query_vector: list[float],
        *,
        k: int,
        exclude_ids: list[ChunkId] | None = None,
        candidate_ids: list[ChunkId] | None = None,
    ) -> list[tuple[ChunkId, float]]:
        import numpy as np

        exclude = set(exclude_ids or [])
        if candidate_ids is not None:
            pool_ids = [cid for cid in candidate_ids if cid in self._chunks]
        else:
            pool_ids = list(self._chunks.keys())
        # Filter to chunks with embeddings, minus exclusions.
        pool = [
            (cid, self._chunks[cid].embedding)
            for cid in pool_ids
            if cid not in exclude and self._chunks[cid].embedding is not None
        ]
        if not pool:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return []
        q = q / qn
        ids = [cid for cid, _ in pool]
        M = np.stack([np.asarray(v, dtype=np.float32) for _, v in pool])
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        M = M / norms
        sims = (M @ q).tolist()
        ranked = sorted(zip(ids, sims), key=lambda x: x[1], reverse=True)
        return ranked[:k]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_chunk_store_knn.py -v
```
Expected: 5 PASS (the protocol test from Task 1 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_chunk_store_knn.py src/foodscholar/storage/in_memory.py
git commit -m "M5 (cross-shelf/p1): InMemoryChunkStore.knn_search_chunks"
```

---

## Task 3: Implement `knn_search_chunks` for `ElasticChunkStore`

**Files:**
- Modify: `src/foodscholar/storage/elastic.py`
- Modify: `tests/integration/test_real_models.py` (or whichever integration test file already exercises ES — confirm location first via `grep -l ElasticChunkStore tests/integration/`)

- [ ] **Step 1: Write the failing integration test**

Add to the existing ES integration test file (e.g. `tests/integration/test_elastic_chunk_store.py` — if it doesn't exist, create it). This requires a live ES at `http://localhost:9200`; the test is gated like other integration tests in the repo.

```python
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FOODSCHOLAR_RUN_INTEGRATION") != "1",
    reason="Integration tests require FOODSCHOLAR_RUN_INTEGRATION=1",
)


def test_elastic_knn_search_returns_top_k(elastic_store, embedded_chunks):
    """elastic_store.knn_search_chunks returns top-k cosine neighbors via ES kNN."""
    # `embedded_chunks` is a fixture that upserts ~10 chunks with known
    # embeddings. Pick chunk[0]'s vector as the query; expect chunk[0] back
    # first unless excluded.
    query_id = embedded_chunks[0].chunk_id
    qvec = embedded_chunks[0].embedding
    result = elastic_store.knn_search_chunks(
        query_vector=qvec, k=3, exclude_ids=[query_id], candidate_ids=None,
    )
    assert len(result) == 3
    assert query_id not in [cid for cid, _ in result]
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)
```

If a `embedded_chunks` fixture doesn't exist, write a minimal one inline at the top of the test file that upserts ~5 chunks with random 768-dim vectors via `np.random.RandomState(42)`.

- [ ] **Step 2: Run test to verify it fails**

```bash
FOODSCHOLAR_RUN_INTEGRATION=1 pytest tests/integration/test_elastic_chunk_store.py -v
```
Expected: FAIL with `AttributeError: 'ElasticChunkStore' object has no attribute 'knn_search_chunks'`.

- [ ] **Step 3: Implement on ElasticChunkStore**

Open `src/foodscholar/storage/elastic.py`. Find the `# ------------------------------------------------------------------ reads` section. Add this method after `search` (around line 478):

```python
    def knn_search_chunks(
        self,
        query_vector: list[float],
        *,
        k: int,
        exclude_ids: list[ChunkId] | None = None,
        candidate_ids: list[ChunkId] | None = None,
    ) -> list[tuple[ChunkId, float]]:
        """kNN search using ES `knn` query on the `embedding` field.

        `exclude_ids` becomes a `must_not.ids` filter; `candidate_ids` becomes
        an `ids` filter (the kNN search is restricted to these). ES returns
        cosine-similarity-derived `_score` in [0, 2] (1 + cosine); we map
        back to plain cosine in [-1, 1] so callers see consistent semantics
        across stores.
        """
        # num_candidates governs HNSW recall — 5x k is the ES default.
        num_candidates = max(50, k * 10)

        filters: list[dict[str, Any]] = []
        if candidate_ids:
            filters.append({"ids": {"values": list(candidate_ids)}})
        must_not: list[dict[str, Any]] = []
        if exclude_ids:
            must_not.append({"ids": {"values": list(exclude_ids)}})

        knn_body: dict[str, Any] = {
            "field": "embedding",
            "query_vector": list(query_vector),
            "k": k,
            "num_candidates": num_candidates,
        }
        if filters or must_not:
            knn_body["filter"] = {
                "bool": {
                    "filter": filters,
                    "must_not": must_not,
                }
            }
        body: dict[str, Any] = {
            "size": k,
            "_source": False,
            "knn": knn_body,
        }
        resp = self._es.search(index=self.index, body=body)
        # ES returns _score = (cosine + 1) / 2 for similarity="cosine" docs
        # in 8.6+ (was 1 + cosine in older). Convert to plain cosine [-1, 1].
        out: list[tuple[ChunkId, float]] = []
        for h in resp["hits"]["hits"]:
            es_score = h["_score"]
            cosine = (es_score * 2.0) - 1.0
            out.append((h["_id"], cosine))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
FOODSCHOLAR_RUN_INTEGRATION=1 pytest tests/integration/test_elastic_chunk_store.py -v
```
Expected: PASS.

- [ ] **Step 5: Run the existing unit suite to confirm no regression**

```bash
pytest tests/unit -x -q
```
Expected: all PASS (the new protocol method on `ElasticChunkStore` doesn't affect anything else yet).

- [ ] **Step 6: Commit**

```bash
git add src/foodscholar/storage/elastic.py tests/integration/test_elastic_chunk_store.py
git commit -m "M5 (cross-shelf/p1): ElasticChunkStore.knn_search_chunks via ES kNN"
```

---

## Task 4: Extend the `discovery_pass` literal with `"global_similarity"`

**Files:**
- Modify: `src/foodscholar/io/graph.py:48`
- Modify: `src/foodscholar/layer_b/models.py:20`
- Modify: `src/foodscholar/layer_b/models.py:34` (`ThemeCandidate.pass_name`)
- Modify: `tests/unit/test_layer_b_models.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_layer_b_models.py`:

```python
def test_theme_accepts_global_similarity_pass():
    """Theme.discovery_pass accepts the new 'global_similarity' value."""
    from foodscholar.io.graph import Theme

    t = Theme(
        theme_id="foods/global/saturated_fat_g1",
        label="saturated fat",
        shelf_ids=["foodon:FOODON:00001234", "foodon:FOODON:00005678"],
        chunk_count=12,
        discovered_by="leiden",
        discovery_version="v0.2",
        facet="foods",
        discovery_pass="global_similarity",
    )
    assert t.discovery_pass == "global_similarity"
    assert len(t.shelf_ids) == 2


def test_theme_candidate_accepts_global_similarity_pass():
    from foodscholar.layer_b.models import ThemeCandidate

    c = ThemeCandidate(
        pass_name="global_similarity",
        chunk_ids={"c1", "c2"},
        foodon_ids=set(),
        centroid_embedding=[0.1] * 768,
    )
    assert c.pass_name == "global_similarity"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_layer_b_models.py::test_theme_accepts_global_similarity_pass tests/unit/test_layer_b_models.py::test_theme_candidate_accepts_global_similarity_pass -v
```
Expected: 2 FAILS with pydantic ValidationError on `discovery_pass`/`pass_name`.

- [ ] **Step 3: Add literal in `io/graph.py`**

In `src/foodscholar/io/graph.py:48`, change:

```python
    discovery_pass: Literal["similarity", "relatedness", "merged"]
```

to:

```python
    discovery_pass: Literal[
        "similarity",
        "relatedness",
        "merged",
        "global_similarity",
    ]
```

- [ ] **Step 4: Add literal in `layer_b/models.py`**

In `src/foodscholar/layer_b/models.py:20`, change:

```python
DiscoveryPass = Literal["similarity", "relatedness", "merged"]
```

to:

```python
DiscoveryPass = Literal["similarity", "relatedness", "merged", "global_similarity"]
```

Then in the same file at line 34, change `ThemeCandidate.pass_name`:

```python
    pass_name: Literal["similarity", "relatedness"]
```

to:

```python
    pass_name: Literal["similarity", "relatedness", "global_similarity"]
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/unit/test_layer_b_models.py -v
```
Expected: all PASS (new + existing).

- [ ] **Step 6: Commit**

```bash
git add src/foodscholar/io/graph.py src/foodscholar/layer_b/models.py tests/unit/test_layer_b_models.py
git commit -m "M5 (cross-shelf/p1): add 'global_similarity' to discovery_pass literal"
```

---

## Task 5: Add `global_similarity_max_chunks` config knob

**Files:**
- Modify: `src/foodscholar/config.py`
- Modify: `tests/unit/test_layer_b_config.py`

This is a safety cap. If the attached corpus grows from 6,290 to 200k, you don't want the global pass to silently take an hour. Above the cap, the orchestrator falls back to per-shelf only with a warning. The cap also gives an emergency off-switch.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_layer_b_config.py`:

```python
def test_layer_b_config_has_global_similarity_max_chunks_default():
    from foodscholar.config import LayerBConfig

    cfg = LayerBConfig()
    assert cfg.global_similarity_max_chunks == 50_000


def test_layer_b_config_global_similarity_max_chunks_is_int():
    from foodscholar.config import LayerBConfig

    cfg = LayerBConfig(global_similarity_max_chunks=10)
    assert cfg.global_similarity_max_chunks == 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_layer_b_config.py::test_layer_b_config_has_global_similarity_max_chunks_default -v
```
Expected: FAIL with `AttributeError: 'LayerBConfig' object has no attribute 'global_similarity_max_chunks'`.

- [ ] **Step 3: Add the field**

Open `src/foodscholar/config.py`. Find the `LayerBConfig` class (search for `class LayerBConfig`). Add this field alongside `min_chunks_per_shelf`:

```python
    global_similarity_max_chunks: int = 50_000
    """Safety cap: if the global similarity pass would see more chunks than this,
    fall back to per-shelf Pass 1 and emit a warning. Default 50k is well above
    the current corpus (~6,290 attached chunks) but well below where the kNN
    fan-out would become operationally painful."""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_layer_b_config.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/config.py tests/unit/test_layer_b_config.py
git commit -m "M5 (cross-shelf/p1): add LayerBConfig.global_similarity_max_chunks"
```

---

## Task 6: Add `build_global_similarity_graph` to `semantic_graph.py`

**Files:**
- Modify: `src/foodscholar/layer_b/semantic_graph.py`
- Modify: `tests/unit/test_layer_b_semantic_graph.py`

This is the heart of the change. Instead of computing all-pairs cosine in memory, we fan out k ES kNN searches and build an igraph from the union of returned edges. Same output type (igraph, vertices carry `chunk_id`, edges carry `weight`) as `build_similarity_graph`, so downstream Leiden code is unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_layer_b_semantic_graph.py`:

```python
def test_build_global_similarity_graph_uses_chunk_store_knn():
    """Global similarity graph fans out kNN per chunk via ChunkStore and
    builds an undirected weighted igraph."""
    from foodscholar.config import LayerBConfig
    from foodscholar.io.chunk import Chunk
    from foodscholar.layer_b.semantic_graph import build_global_similarity_graph
    from foodscholar.storage.in_memory import InMemoryChunkStore

    def _chunk(cid: str, vec: list[float]) -> Chunk:
        return Chunk(
            chunk_id=cid, text=cid,
            source_doc_id="d", source_type="textbook", section_type="body",
            embedding=vec, embedding_model="m",
        )

    store = InMemoryChunkStore()
    store.upsert([
        _chunk("A", [1.0, 0.0, 0.0]),
        _chunk("B", [0.99, 0.14, 0.0]),
        _chunk("C", [0.0, 1.0, 0.0]),
        _chunk("D", [0.0, 0.0, 1.0]),
    ])
    cfg = LayerBConfig()
    cfg.similarity.knn_k = 2
    cfg.similarity.edge_threshold = 0.5
    cfg.similarity.require_mutual = False

    g = build_global_similarity_graph(
        chunk_ids=["A", "B", "C", "D"],
        chunk_store=store,
        cfg=cfg.similarity,
    )
    # 4 vertices, in chunk_id order
    assert list(g.vs["chunk_id"]) == ["A", "B", "C", "D"]
    # A and B are nearly parallel (cosine ~0.99) — must be connected.
    edge_pairs = {
        tuple(sorted([g.vs[e.source]["chunk_id"], g.vs[e.target]["chunk_id"]]))
        for e in g.es
    }
    assert ("A", "B") in edge_pairs
    # A and D are orthogonal (cosine 0); should not be connected.
    assert ("A", "D") not in edge_pairs


def test_build_global_similarity_graph_empty_input():
    from foodscholar.config import LayerBConfig
    from foodscholar.layer_b.semantic_graph import build_global_similarity_graph
    from foodscholar.storage.in_memory import InMemoryChunkStore

    g = build_global_similarity_graph(
        chunk_ids=[], chunk_store=InMemoryChunkStore(), cfg=LayerBConfig().similarity,
    )
    assert g.vcount() == 0
    assert g.ecount() == 0


def test_build_global_similarity_graph_respects_require_mutual():
    """When require_mutual=True, an edge A→B only survives if B also lists A
    in its top-k."""
    from foodscholar.config import LayerBConfig
    from foodscholar.io.chunk import Chunk
    from foodscholar.layer_b.semantic_graph import build_global_similarity_graph
    from foodscholar.storage.in_memory import InMemoryChunkStore

    def _chunk(cid, vec):
        return Chunk(
            chunk_id=cid, text=cid, source_doc_id="d", source_type="textbook",
            section_type="body", embedding=vec, embedding_model="m",
        )

    # A's nearest is B; B's nearest is C; C's nearest is B. With require_mutual,
    # only B-C survives.
    store = InMemoryChunkStore()
    store.upsert([
        _chunk("A", [1.0, 0.0, 0.0]),
        _chunk("B", [0.9, 0.4, 0.0]),
        _chunk("C", [0.85, 0.5, 0.0]),
    ])
    cfg = LayerBConfig()
    cfg.similarity.knn_k = 1
    cfg.similarity.edge_threshold = 0.0
    cfg.similarity.require_mutual = True

    g = build_global_similarity_graph(
        chunk_ids=["A", "B", "C"], chunk_store=store, cfg=cfg.similarity,
    )
    edges = {
        tuple(sorted([g.vs[e.source]["chunk_id"], g.vs[e.target]["chunk_id"]]))
        for e in g.es
    }
    # A's top-1 is B but B's top-1 is C, so A-B is asymmetric -> dropped.
    # B's top-1 is C and C's top-1 is B -> kept.
    assert ("B", "C") in edges
    assert ("A", "B") not in edges
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_layer_b_semantic_graph.py::test_build_global_similarity_graph_uses_chunk_store_knn -v
```
Expected: FAIL with `ImportError` on `build_global_similarity_graph`.

- [ ] **Step 3: Implement the function**

Add to `src/foodscholar/layer_b/semantic_graph.py` (below the existing `build_similarity_graph`):

```python
def build_global_similarity_graph(
    chunk_ids: list[ChunkId],
    chunk_store: Any,  # ChunkStore protocol
    cfg: SimilarityConfig,
) -> Any:
    """Build a similarity graph across `chunk_ids` using ChunkStore.knn_search_chunks.

    Unlike `build_similarity_graph` (which computes all-pairs cosine on a
    small per-shelf set), this fans out one kNN call per chunk against the
    chunk store's HNSW index — meant for ~thousands of chunks where the
    O(n^2) in-memory approach would blow up.

    Output shape matches `build_similarity_graph`:
      - vertices carry `chunk_id` attribute, indexed in the order of input
      - edges carry `weight = cosine`
      - empty input or n < 2 returns a graph with vertices but no edges

    The kNN search is restricted to `candidate_ids=chunk_ids` so the global
    graph stays inside the attached corpus even when the underlying store
    contains additional chunks.
    """
    import igraph as ig

    g = ig.Graph()
    if not chunk_ids:
        return g

    g.add_vertices(len(chunk_ids))
    g.vs["chunk_id"] = chunk_ids

    if len(chunk_ids) < 2:
        return g

    # Fetch all the query vectors in one call. The fields-API merge in
    # ElasticChunkStore.get_many handles ES 9.x source-stripping.
    chunks = chunk_store.get_many(chunk_ids)
    qvecs: dict[ChunkId, list[float]] = {
        c.chunk_id: c.embedding for c in chunks if c.embedding is not None
    }

    chunk_id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
    candidate_set = list(chunk_ids)  # restrict kNN to the attached corpus

    # For each chunk: ask the store for its top-k neighbors. Build an
    # asymmetric edge set first, then enforce the mutual/threshold gates.
    neighbors: dict[ChunkId, set[ChunkId]] = {}
    edge_weights: dict[tuple[int, int], float] = {}
    for cid in chunk_ids:
        qvec = qvecs.get(cid)
        if qvec is None:
            neighbors[cid] = set()
            continue
        hits = chunk_store.knn_search_chunks(
            query_vector=qvec,
            k=cfg.knn_k,
            exclude_ids=[cid],
            candidate_ids=candidate_set,
        )
        neighbors[cid] = {nid for nid, _ in hits}
        for nid, score in hits:
            if score < cfg.edge_threshold:
                continue
            if nid not in chunk_id_to_idx:
                continue  # safety: kNN backend returned an unknown id
            i, j = chunk_id_to_idx[cid], chunk_id_to_idx[nid]
            key = (i, j) if i < j else (j, i)
            # If we see the same edge from both directions, take the max
            # (numerical noise can produce slightly different scores).
            prev = edge_weights.get(key)
            if prev is None or score > prev:
                edge_weights[key] = score

    if cfg.require_mutual:
        edge_weights = {
            (i, j): w
            for (i, j), w in edge_weights.items()
            if chunk_ids[j] in neighbors.get(chunk_ids[i], set())
            and chunk_ids[i] in neighbors.get(chunk_ids[j], set())
        }

    if edge_weights:
        edge_list = sorted(edge_weights)
        g.add_edges(edge_list)
        g.es["weight"] = [edge_weights[e] for e in edge_list]
    return g
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_layer_b_semantic_graph.py -v
```
Expected: all PASS (existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_b/semantic_graph.py tests/unit/test_layer_b_semantic_graph.py
git commit -m "M5 (cross-shelf/p2): build_global_similarity_graph via ChunkStore kNN"
```

---

## Task 7: Add `build_global_similarity_candidates` orchestrator helper

**Files:**
- Modify: `src/foodscholar/layer_b/builder.py`
- Modify: `tests/unit/test_layer_b_builder.py`

This wraps the graph + Leiden + centroid computation, returning `ThemeCandidate(pass_name="global_similarity")` records — analogous to the existing `build_shelf_similarity_candidates`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_layer_b_builder.py`:

```python
def test_build_global_similarity_candidates_returns_themecandidate_records():
    """Pass 1 global helper returns ThemeCandidate(pass_name='global_similarity')
    with chunk_ids and centroid set; foodon_ids stay empty (Pass 1 doesn't use
    entities)."""
    from foodscholar.config import LayerBConfig
    from foodscholar.io.chunk import Chunk
    from foodscholar.layer_b.builder import build_global_similarity_candidates
    from foodscholar.storage.in_memory import InMemoryChunkStore

    def _chunk(cid, vec):
        return Chunk(
            chunk_id=cid, text=cid, source_doc_id="d", source_type="textbook",
            section_type="body", embedding=vec, embedding_model="m",
        )

    # Two tight clusters, well separated.
    store = InMemoryChunkStore()
    store.upsert([
        _chunk("A1", [1.0, 0.0, 0.0]),
        _chunk("A2", [0.99, 0.14, 0.0]),
        _chunk("A3", [0.98, 0.20, 0.0]),
        _chunk("B1", [0.0, 0.0, 1.0]),
        _chunk("B2", [0.0, 0.14, 0.99]),
        _chunk("B3", [0.0, 0.20, 0.98]),
    ])
    cfg = LayerBConfig()
    cfg.similarity.knn_k = 3
    cfg.similarity.edge_threshold = 0.5

    cands = build_global_similarity_candidates(
        chunk_ids=["A1", "A2", "A3", "B1", "B2", "B3"],
        chunk_store=store,
        cfg=cfg,
    )
    assert len(cands) >= 1
    for c in cands:
        assert c.pass_name == "global_similarity"
        assert c.foodon_ids == set()
        assert c.centroid_embedding is not None
        assert len(c.centroid_embedding) == 3  # input dim


def test_build_global_similarity_candidates_returns_empty_when_no_embeddings():
    from foodscholar.config import LayerBConfig
    from foodscholar.io.chunk import Chunk
    from foodscholar.layer_b.builder import build_global_similarity_candidates
    from foodscholar.storage.in_memory import InMemoryChunkStore

    store = InMemoryChunkStore()
    store.upsert([
        Chunk(
            chunk_id="A", text="A", source_doc_id="d", source_type="textbook",
            section_type="body", embedding=None, embedding_model=None,
        ),
    ])
    cfg = LayerBConfig()
    cands = build_global_similarity_candidates(
        chunk_ids=["A"], chunk_store=store, cfg=cfg,
    )
    assert cands == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_layer_b_builder.py::test_build_global_similarity_candidates_returns_themecandidate_records -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the function**

In `src/foodscholar/layer_b/builder.py`, add this function near the top (after the existing `build_shelf_similarity_candidates`, around line 113):

```python
def build_global_similarity_candidates(
    chunk_ids: list[str],
    chunk_store: Any,  # ChunkStore
    cfg: LayerBConfig,
) -> list[ThemeCandidate]:
    """Run Pass 1 (similarity) across the WHOLE attached corpus.

    Unlike `build_shelf_similarity_candidates` (per-shelf), this builds one
    big similarity graph fed by `ChunkStore.knn_search_chunks` and emits
    cross-shelf candidates. Output `ThemeCandidate.pass_name` is
    `"global_similarity"` so the merge step can tell global candidates
    apart from per-shelf relatedness candidates.
    """
    import numpy as np

    from foodscholar.layer_b.community import run_leiden
    from foodscholar.layer_b.semantic_graph import build_global_similarity_graph

    if not chunk_ids:
        return []

    g = build_global_similarity_graph(chunk_ids, chunk_store, cfg.similarity)
    communities = run_leiden(g, cfg.leiden)
    if not communities:
        return []

    # Fetch embeddings once for centroid math (kNN already had them, but
    # we want them in numpy form here).
    chunks = chunk_store.get_many(chunk_ids)
    embeddings: dict[str, np.ndarray] = {
        c.chunk_id: np.asarray(c.embedding, dtype=np.float32)
        for c in chunks
        if c.embedding is not None
    }

    index_to_id: list[str] = list(g.vs["chunk_id"])
    out: list[ThemeCandidate] = []
    for members in communities:
        member_ids = {index_to_id[i] for i in members if index_to_id[i] in embeddings}
        if not member_ids:
            continue
        member_vecs = np.stack([embeddings[cid] for cid in member_ids])
        norms = np.linalg.norm(member_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = member_vecs / norms
        centroid = normed.mean(axis=0)
        cn = np.linalg.norm(centroid)
        if cn > 0:
            centroid = centroid / cn
        out.append(
            ThemeCandidate(
                pass_name="global_similarity",
                chunk_ids=member_ids,
                foodon_ids=set(),
                centroid_embedding=centroid.tolist(),
                discovered_by="leiden",
            )
        )
    return out
```

You'll also need to make sure `Any` is imported at the top of the file (`from typing import TYPE_CHECKING, Any`).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_layer_b_builder.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_b/builder.py tests/unit/test_layer_b_builder.py
git commit -m "M5 (cross-shelf/p2): build_global_similarity_candidates orchestrator helper"
```

---

## Task 8: Generalize `merge_candidates` for global × per-shelf

**Files:**
- Modify: `src/foodscholar/layer_b/merge.py`
- Modify: `tests/unit/test_layer_b_merge.py`

Today `merge_candidates(sim_cands, rel_cands, cfg)` pairs candidates assuming both came from the same shelf. We want it to also work with one global sim-candidate set crossed against a *flattened* list of per-shelf rel-candidates. The Jaccard logic is unchanged; only the input shape changes (and the resulting merged theme's `shelf_ids` will be a union).

First, read `merge.py` end-to-end before deciding what to change. The change might be: add a new helper `merge_global_and_local_candidates(global_sim_cands, rel_cands_by_shelf, cfg)` that calls `merge_candidates` internally and tracks the originating shelf per relatedness candidate so it can compute the union for the final theme.

- [ ] **Step 1: Read `merge.py`** (no test yet)

```bash
cat src/foodscholar/layer_b/merge.py
```

Confirm: does `merge_candidates` already accept arbitrary `pass_name`? If yes (it should, since it only looks at `chunk_ids` + `foodon_ids`), we don't modify it and instead add a wrapper.

- [ ] **Step 2: Write the failing test**

Add to `tests/unit/test_layer_b_merge.py`:

```python
def test_merge_global_and_local_returns_themes_with_union_shelf_ids():
    """A global similarity candidate that overlaps a per-shelf relatedness
    candidate produces a merged theme whose source shelves = union of both."""
    from foodscholar.config import LayerBConfig
    from foodscholar.layer_b.merge import merge_global_and_local_candidates
    from foodscholar.layer_b.models import ThemeCandidate

    global_cands = [
        ThemeCandidate(
            pass_name="global_similarity",
            chunk_ids={"c1", "c2", "c3", "c4"},
            foodon_ids=set(),
            centroid_embedding=[0.1] * 3,
        ),
    ]
    # Two per-shelf relatedness candidates, one per shelf. Both overlap the
    # global candidate significantly.
    rel_cands_by_shelf = {
        "shelf:fat": [
            ThemeCandidate(
                pass_name="relatedness",
                chunk_ids={"c1", "c2"},
                foodon_ids={"FOODON:1"},
            ),
        ],
        "shelf:meat": [
            ThemeCandidate(
                pass_name="relatedness",
                chunk_ids={"c3", "c4"},
                foodon_ids={"FOODON:2"},
            ),
        ],
    }
    cfg = LayerBConfig()
    themes, decisions = merge_global_and_local_candidates(
        global_cands, rel_cands_by_shelf, cfg.merge,
    )
    # We expect at least one merged theme that spans both shelves.
    merged = [t for t in themes if t["discovery_pass"] == "merged"]
    assert any(set(t["shelf_ids"]) == {"shelf:fat", "shelf:meat"} for t in merged)


def test_merge_global_and_local_unmerged_global_keeps_union_shelf_ids():
    """A global similarity theme that didn't merge with any relatedness
    candidate still picks up shelf_ids from the chunk-store side; this
    function returns shelf_ids=[] for unmerged globals so the orchestrator
    can attach shelf_ids from chunk.shelf_ids (Task 9)."""
    from foodscholar.config import LayerBConfig
    from foodscholar.layer_b.merge import merge_global_and_local_candidates
    from foodscholar.layer_b.models import ThemeCandidate

    global_cands = [
        ThemeCandidate(
            pass_name="global_similarity",
            chunk_ids={"c100", "c101"},
            foodon_ids=set(),
            centroid_embedding=[0.1] * 3,
        ),
    ]
    cfg = LayerBConfig()
    themes, _ = merge_global_and_local_candidates(global_cands, {}, cfg.merge)
    glob = [t for t in themes if t["discovery_pass"] == "global_similarity"]
    assert len(glob) == 1
    # The merge layer doesn't know the chunk→shelf map; orchestrator fills it.
    assert glob[0]["shelf_ids"] == []
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/unit/test_layer_b_merge.py::test_merge_global_and_local_returns_themes_with_union_shelf_ids -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 4: Implement `merge_global_and_local_candidates`**

In `src/foodscholar/layer_b/merge.py`, add at the end:

```python
def merge_global_and_local_candidates(
    global_sim_cands: list[ThemeCandidate],
    rel_cands_by_shelf: dict[str, list[ThemeCandidate]],
    cfg: MergeConfig,
) -> tuple[list[dict], list[MergeDecision]]:
    """Merge one global similarity-candidate set against per-shelf relatedness
    candidates, producing theme dicts with `shelf_ids: list[str]`.

    Algorithm:
      1. Flatten `rel_cands_by_shelf` into a single list, remembering the
         originating shelf per candidate.
      2. Call the existing `merge_candidates(global_sim_cands, flat_rel, cfg)`.
      3. For each emitted theme dict, set `shelf_ids` to:
         - merged: union of source shelves from the contributing rel-cands
         - global_similarity (unmerged): [] (orchestrator backfills from chunk.shelf_ids)
         - relatedness (unmerged): [origin_shelf]

    Returns `(themes, decisions)` where themes is a list of dicts ready for
    the orchestrator to construct Pydantic Theme records, and decisions is
    a list of MergeDecision audit records.
    """
    flat_rel: list[ThemeCandidate] = []
    rel_origin_shelf: list[str] = []  # parallel to flat_rel
    for shelf_id, cands in rel_cands_by_shelf.items():
        for c in cands:
            flat_rel.append(c)
            rel_origin_shelf.append(shelf_id)

    themes, decisions = merge_candidates(global_sim_cands, flat_rel, cfg)

    # Decisions index into (global_sim_cands, flat_rel). Build a per-merged-
    # theme map of contributing rel-candidate indices so we can union shelves.
    merged_to_rel_idxs: dict[int, list[int]] = {}
    next_merged_theme = 0
    for d in decisions:
        if not d.merged:
            continue
        merged_to_rel_idxs.setdefault(next_merged_theme, []).append(
            d.relatedness_candidate_idx
        )
        next_merged_theme += 1
    # NOTE: The above assumes merged decisions are emitted in the same order
    # as merged themes in `themes`. If `merge_candidates` deviates from that,
    # change this to use a stable join key on chunk_ids.

    out: list[dict] = []
    merged_seq = 0
    for t in themes:
        pass_kind = t["discovery_pass"]
        if pass_kind == "merged":
            rel_idxs = merged_to_rel_idxs.get(merged_seq, [])
            shelves = sorted({rel_origin_shelf[i] for i in rel_idxs})
            t = {**t, "shelf_ids": shelves}
            merged_seq += 1
        elif pass_kind in ("similarity", "global_similarity"):
            # Unmerged global similarity has no source shelf at merge time —
            # orchestrator will backfill from chunk.shelf_ids.
            t = {**t, "shelf_ids": []}
        elif pass_kind == "relatedness":
            # Find the originating shelf by matching chunk_ids.
            origin = None
            for shelf_id, cands in rel_cands_by_shelf.items():
                if any(c.chunk_ids == t["chunk_ids"] for c in cands):
                    origin = shelf_id
                    break
            t = {**t, "shelf_ids": [origin] if origin else []}
        out.append(t)
    return out, decisions
```

You may need a small change inside `merge_candidates` so emitted theme dicts carry `"discovery_pass": "global_similarity"` when the source was a global candidate (today it may hard-code `"similarity"`). Read it carefully and fix if so.

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_layer_b_merge.py -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/foodscholar/layer_b/merge.py tests/unit/test_layer_b_merge.py
git commit -m "M5 (cross-shelf/p2): merge_global_and_local_candidates with union shelf_ids"
```

---

## Task 9: Rewrite `build_layer_b` orchestrator

**Files:**
- Modify: `src/foodscholar/layer_b/builder.py` (the `build_layer_b` function — currently L322-438)
- Modify: `tests/unit/test_layer_b_builder.py`

The new orchestrator flow:

1. Collect attached chunks (`attachments = graph_store.list_chunk_shelf_attachments()`).
2. Skip globally if `len(attached_chunk_ids) > cfg.global_similarity_max_chunks`.
3. Run `build_global_similarity_candidates(attached_chunk_ids, chunk_store, cfg)` — one global Leiden.
4. For each shelf with `chunk_count >= min_chunks_per_shelf`, run `build_shelf_relatedness_candidates(shelf_chunks, cfg)`. Collect into `rel_cands_by_shelf`.
5. Run `merge_global_and_local_candidates(global_cands, rel_cands_by_shelf, cfg.merge)`.
6. For unmerged global themes, backfill `shelf_ids` = union of `chunk.shelf_ids` across members.
7. For each emitted theme: keywords (c-TF-IDF), label (LLM or keyword), primary pick (existing `pick_primary`), build `Theme` Pydantic record.
8. Single `clear_themes()` + `persist_themes(...)` (since persist takes a flat list).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_layer_b_builder.py` (this is a higher-level test that exercises the whole orchestrator with in-memory stores):

```python
def test_build_layer_b_emits_cross_shelf_themes_when_global_finds_them(
    in_memory_fs_with_attached_chunks,
):
    """When the global similarity pass discovers a community spanning chunks
    attached to two different shelves, the resulting Theme has both shelves
    in shelf_ids."""
    fs = in_memory_fs_with_attached_chunks
    # The fixture sets up 6 chunks: 3 on shelf X, 3 on shelf Y. The 6 share a
    # common embedding cluster so the global Pass 1 should find them as one
    # community (the per-shelf v1 code wouldn't see this).

    artifact = fs.build_layer_b(facet="foods", dry_run=False)

    themes = fs.graph_store.list_themes()
    cross_shelf = [t for t in themes if len(t.shelf_ids) >= 2]
    assert cross_shelf, (
        f"expected ≥1 cross-shelf theme; got themes={[(t.label, t.shelf_ids) for t in themes]}"
    )
    # And every theme has at least one shelf.
    assert all(len(t.shelf_ids) >= 1 for t in themes)
```

Add the supporting fixture to the same file or a `conftest.py`:

```python
@pytest.fixture
def in_memory_fs_with_attached_chunks(tmp_path):
    from foodscholar.config import FoodScholarConfig, LayerBConfig
    from foodscholar.facade import FoodScholar
    from foodscholar.io.chunk import Chunk
    from foodscholar.io.graph import Shelf
    from foodscholar.storage.in_memory import InMemoryChunkStore, InMemoryGraphStore

    chunk_store = InMemoryChunkStore()
    graph_store = InMemoryGraphStore()

    # Two shelves under 'foods'
    graph_store.upsert_shelves([
        Shelf(
            shelf_id="shelf:fat", label="fat", facet="foods",
            foodon_id="FOODON:1", chunk_count=3,
        ),
        Shelf(
            shelf_id="shelf:meat", label="meat", facet="foods",
            foodon_id="FOODON:2", chunk_count=3,
        ),
    ])
    # 6 chunks — first 3 attached to 'shelf:fat', last 3 to 'shelf:meat'.
    # All 6 sit in a nearby region of embedding space so the global similarity
    # pass finds them as one cluster.
    base = [1.0, 0.0, 0.0]
    def jitter(i):
        return [base[0] - 0.01 * i, 0.01 * i, 0.0]
    chunks = [
        Chunk(
            chunk_id=f"c{i}", text=f"chunk {i}",
            source_doc_id="d", source_type="textbook", section_type="body",
            embedding=jitter(i), embedding_model="m",
            shelf_ids=["shelf:fat" if i < 3 else "shelf:meat"],
        )
        for i in range(6)
    ]
    chunk_store.upsert(chunks)
    graph_store.attach_chunks_to_shelves_bulk(
        [(c.chunk_id, c.shelf_ids[0]) for c in chunks]
    )

    cfg = FoodScholarConfig()
    cfg.layer_b = LayerBConfig()
    cfg.layer_b.min_chunks_per_shelf = 2
    cfg.layer_b.similarity.knn_k = 5
    cfg.layer_b.similarity.edge_threshold = 0.5

    fs = FoodScholar(cfg, chunk_store=chunk_store, graph_store=graph_store)
    return fs
```

Note: the fixture relies on `InMemoryGraphStore.attach_chunks_to_shelves_bulk` and other API surface; adapt to whatever the actual InMemoryGraphStore method is (run `grep -n "def " src/foodscholar/storage/in_memory.py | head -40` to confirm).

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_layer_b_builder.py::test_build_layer_b_emits_cross_shelf_themes_when_global_finds_them -v
```
Expected: FAIL (the existing per-shelf builder won't produce cross-shelf themes).

- [ ] **Step 3: Replace `build_layer_b`**

In `src/foodscholar/layer_b/builder.py`, replace the entire `build_layer_b` function (currently L322-438) with this new implementation:

```python
def build_layer_b(
    fs,
    *,
    facet: str = "foods",
    dry_run: bool = False,
):
    """Top-level Layer B orchestrator (cross-shelf design).

    Flow:
      1. List shelves of the target facet; collect their attached chunks.
      2. Run Pass 1 (similarity) globally over ALL attached chunks (one Leiden
         on a big graph) — produces cross-shelf candidates.
      3. Run Pass 2 (relatedness) per-shelf — produces per-shelf candidates.
         Entity coherence is sharper inside a shelf's chunk set.
      4. Merge global × per-shelf via `merge_global_and_local_candidates`,
         taking shelf_ids = union across merged sources.
      5. For unmerged global similarity themes, backfill shelf_ids from
         chunk.shelf_ids (union across the theme's member chunks).
      6. Label, pick primary, build Theme records.
      7. Single clear_themes + persist_themes.

    Safety hatch: if the attached corpus exceeds
    `cfg.global_similarity_max_chunks`, log a warning and fall back to the
    old per-shelf Pass 1 (so a misconfigured run can't silently produce a
    multi-hour kNN fan-out).
    """
    from collections import Counter

    from foodscholar.layer_b.label import label_by_keywords, label_by_llm
    from foodscholar.layer_b.merge import merge_global_and_local_candidates
    from foodscholar.layer_b.models import LayerBArtifact
    from foodscholar.layer_b.persist import persist_themes
    from foodscholar.layer_b.primary import pick_primary
    from foodscholar.layer_b.relatedness_graph import build_relatedness_graph
    from foodscholar.versioning import make_artifact_meta

    cfg = fs.config.layer_b
    meta = make_artifact_meta(phase="layer_b", config=fs.config, record_count=0)
    started = _utc_iso()

    # 1. Collect attachments
    attachments = fs.graph_store.list_chunk_shelf_attachments()
    shelf_to_chunks: dict[str, list[str]] = {}
    for chunk_id, shelf_ids in attachments.items():
        for sid in shelf_ids:
            shelf_to_chunks.setdefault(sid, []).append(chunk_id)

    # Filter to facet-relevant shelves only.
    facet_shelves = {s.shelf_id: s for s in fs.graph_store.list_shelves() if s.facet == facet}
    # Exclude the synthetic facet root from the attached set.
    synth_root = f"facet:{facet}"
    attached_chunk_ids = sorted({
        cid for cid, sids in attachments.items()
        if any(sid in facet_shelves and sid != synth_root for sid in sids)
    })

    # 2. Global Pass 1 (cross-shelf similarity)
    if len(attached_chunk_ids) > cfg.global_similarity_max_chunks:
        import warnings
        warnings.warn(
            f"Attached corpus ({len(attached_chunk_ids)}) exceeds "
            f"cfg.global_similarity_max_chunks ({cfg.global_similarity_max_chunks}); "
            "skipping global Pass 1.",
            stacklevel=2,
        )
        global_cands = []
    else:
        global_cands = build_global_similarity_candidates(
            chunk_ids=attached_chunk_ids,
            chunk_store=fs.chunk_store,
            cfg=cfg,
        )

    # 3. Per-shelf Pass 2 (relatedness)
    rel_cands_by_shelf: dict[str, list[ThemeCandidate]] = {}
    for shelf_id, chunk_ids in shelf_to_chunks.items():
        if shelf_id not in facet_shelves or shelf_id == synth_root:
            continue
        if len(chunk_ids) < cfg.min_chunks_per_shelf:
            continue
        chunks = fs.chunk_store.get_many(chunk_ids)
        rel_cands_by_shelf[shelf_id] = build_shelf_relatedness_candidates(chunks, cfg)

    # 4. Merge
    theme_dicts, decisions = merge_global_and_local_candidates(
        global_cands, rel_cands_by_shelf, cfg.merge,
    )

    # 5. Backfill shelf_ids for unmerged global themes using chunk.shelf_ids
    #    (the merge layer leaves them empty intentionally).
    chunk_shelf_map: dict[str, list[str]] = {
        cid: [sid for sid in sids if sid in facet_shelves and sid != synth_root]
        for cid, sids in attachments.items()
    }
    for td in theme_dicts:
        if td["discovery_pass"] != "global_similarity" or td["shelf_ids"]:
            continue
        shelf_union: set[str] = set()
        for cid in td["chunk_ids"]:
            shelf_union.update(chunk_shelf_map.get(cid, []))
        td["shelf_ids"] = sorted(shelf_union)

    # 6. Label + primary + build Theme records
    if not theme_dicts:
        artifact = LayerBArtifact(
            artifact_id=meta.artifact_id,
            facet=facet,
            config_hash=meta.config_hash,
            n_shelves_themed=0, n_shelves_skipped=len(facet_shelves),
            n_themes_total=0, n_themes_by_pass={},
            leiden_seed=cfg.leiden.random_state,
            started_at=started, finished_at=_utc_iso(),
        )
        return artifact

    all_chunk_ids = sorted({cid for td in theme_dicts for cid in td["chunk_ids"]})
    chunks_by_id = {c.chunk_id: c for c in fs.chunk_store.get_many(all_chunk_ids)}

    import numpy as np
    embeddings: dict[str, np.ndarray] = {
        c.chunk_id: np.asarray(c.embedding, dtype=np.float32)
        for c in chunks_by_id.values()
        if c.embedding is not None
    }

    # Per-theme chunk lists for c-TF-IDF labeling.
    theme_chunks: dict[int, list] = {
        i: [chunks_by_id[cid] for cid in td["chunk_ids"] if cid in chunks_by_id]
        for i, td in enumerate(theme_dicts)
    }
    keywords = label_by_keywords(theme_chunks, cfg.labeling)
    if cfg.labeling.strategy == "llm" and fs.llm is not None:
        labels = label_by_llm(theme_chunks, keywords, fs.llm, cfg.labeling)
    else:
        labels = {
            i: " ".join(keywords.get(i, ["unlabeled"])[:3]) for i in theme_chunks
        }

    themes: list = []  # list[Theme]
    chunk_assignments: dict[str, list[tuple[str, bool, float]]] = {}
    seq_by_pass: dict[str, int] = {"global_similarity": 0, "relatedness": 0, "merged": 0}

    # Build shared relatedness graph across the attached corpus for primary picking.
    # (Per-shelf was cheaper but with global themes we need a single graph.)
    all_attached_chunks = [
        chunks_by_id[cid] for cid in all_chunk_ids if cid in chunks_by_id
    ]
    rel_graph = build_relatedness_graph(all_attached_chunks, cfg.relatedness)
    import igraph as ig
    sim_graph = ig.Graph()  # not used by picker today

    from foodscholar.io.graph import Theme

    for i, td in enumerate(theme_dicts):
        pass_kind = td["discovery_pass"]
        seq_by_pass[pass_kind] += 1
        seq = seq_by_pass[pass_kind]
        label = labels.get(i, "unlabeled")
        # For naming: if the theme spans many shelves, use the facet name as slug.
        slug_seed = td["shelf_ids"][0] if td["shelf_ids"] else f"facet_{facet}"
        tid = _theme_id(facet, slug_seed, label, pass_kind, seq)

        ent_counter: Counter[str] = Counter()
        for cid in td["chunk_ids"]:
            c = chunks_by_id.get(cid)
            if c is None:
                continue
            for link in c.entity_links:
                if link.confidence >= cfg.relatedness.tau_strict:
                    ent_counter[link.ontology_id] += 1
        signature = [oid for oid, _ in ent_counter.most_common(10)]

        # Pick primary.
        centroid = None
        if pass_kind in ("global_similarity", "merged"):
            for gc in global_cands:
                if gc.chunk_ids & td["chunk_ids"]:
                    centroid = gc.centroid_embedding
                    break
        primary_chunk = pick_primary(
            chunk_ids=set(td["chunk_ids"]),
            discovery_pass=pass_kind,
            embeddings=embeddings,
            centroid=centroid,
            sim_graph=sim_graph,
            rel_graph=rel_graph,
        )

        themes.append(
            Theme(
                theme_id=tid,
                label=label,
                shelf_ids=td["shelf_ids"],
                chunk_count=len(td["chunk_ids"]),
                discovered_by=td.get("discovered_by", "leiden"),
                discovery_version="v0.2",
                facet=facet,
                discovery_pass=pass_kind,
                keyword_terms=list(keywords.get(i, [])),
                foodon_id_signature=signature,
                config_hash=meta.config_hash,
                version="v0.2",
            )
        )
        chunk_assignments[tid] = [
            (cid, cid == primary_chunk, 1.0 if cid == primary_chunk else 0.5)
            for cid in sorted(td["chunk_ids"])
        ]

    # 7. Persist
    if not dry_run:
        fs.graph_store.clear_themes()
        persist_themes(themes, chunk_assignments, fs.graph_store, fs.chunk_store)

    by_pass: dict[str, int] = {}
    for t in themes:
        by_pass[t.discovery_pass] = by_pass.get(t.discovery_pass, 0) + 1

    return LayerBArtifact(
        artifact_id=meta.artifact_id,
        facet=facet,
        config_hash=meta.config_hash,
        n_shelves_themed=len({s for t in themes for s in t.shelf_ids}),
        n_shelves_skipped=len(facet_shelves) - len({s for t in themes for s in t.shelf_ids}),
        n_themes_total=len(themes),
        n_themes_by_pass=by_pass,
        leiden_seed=cfg.leiden.random_state,
        started_at=started,
        finished_at=_utc_iso(),
    )
```

- [ ] **Step 4: Run unit tests to verify**

```bash
pytest tests/unit/test_layer_b_builder.py -v
```
Expected: all PASS (the new fixture-driven cross-shelf test + all existing tests). If the existing tests fail because they assumed per-shelf behavior, update them to expect the new behavior (this is intentional — that's the contract change).

- [ ] **Step 5: Run the full unit suite**

```bash
pytest tests/unit -x -q
```
Expected: all PASS. Fix anything that broke that wasn't in the test plan — most likely candidates are tests in `test_layer_b_audit.py` and `test_layer_b_pipeline.py` that hard-coded single-shelf assumptions.

- [ ] **Step 6: Commit**

```bash
git add src/foodscholar/layer_b/builder.py tests/unit/test_layer_b_builder.py
git commit -m "M5 (cross-shelf/p3): rewrite build_layer_b for hybrid global/per-shelf"
```

---

## Task 10: Update Layer B audit to handle cross-shelf themes

**Files:**
- Modify: `src/foodscholar/layer_b/audit.py`
- Modify: `tests/unit/test_layer_b_audit.py`

The existing audit assumes one shelf per theme (via `t.shelf_ids[0]`). Cross-shelf themes break that assumption. Key invariants that must still pass:

1. Parity: every `chunk.theme_ids` in ES has a matching `THEME_OF` edge in Neo4j (and vice versa).
2. No dangling theme_ids on chunks (theme_id refers to a nonexistent Theme node).
3. No empty themes (chunk_count > 0 but zero THEME_OF edges).
4. **(new)** Every theme has `len(shelf_ids) >= 1` (since orphan themes with empty shelf_ids would be unreachable from any shelf in the UI).

- [ ] **Step 1: Read the existing audit code**

```bash
cat src/foodscholar/layer_b/audit.py
```

Identify each place that indexes `shelf_ids[0]` or assumes single-shelf.

- [ ] **Step 2: Write the failing test**

Add to `tests/unit/test_layer_b_audit.py`:

```python
def test_audit_passes_when_theme_has_multiple_shelves(in_memory_fs_with_themes):
    """Audit must not flag a theme whose shelf_ids has 2+ entries — that's
    the whole point of cross-shelf themes."""
    fs = in_memory_fs_with_themes  # fixture: setup with one cross-shelf theme
    report = fs.audit_layer_b(facet="foods")
    assert report.passed
    assert report.dangling_edges == 0


def test_audit_flags_theme_with_empty_shelf_ids(in_memory_fs_with_orphan_theme):
    """A theme with shelf_ids=[] is unreachable — must fail audit."""
    fs = in_memory_fs_with_orphan_theme
    report = fs.audit_layer_b(facet="foods")
    assert not report.passed
```

(Add the fixtures alongside — copy the structure of the Task 9 fixture, varying which theme(s) get created.)

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/unit/test_layer_b_audit.py::test_audit_passes_when_theme_has_multiple_shelves -v
```
Expected: FAIL or error.

- [ ] **Step 4: Update audit code**

In `src/foodscholar/layer_b/audit.py`, find every place that does `theme.shelf_ids[0]` and replace with iteration over `theme.shelf_ids`. Add a new gate that counts themes with `len(shelf_ids) == 0`.

Add a field to `LayerBAuditReport` in `src/foodscholar/layer_b/models.py`:

```python
    orphan_themes: int = 0
    """Themes with shelf_ids=[] — unreachable from any shelf in the UI."""
```

And in the `passed` property:

```python
    @property
    def passed(self) -> bool:
        return (
            self.parity == 1.0
            and self.dangling_edges == 0
            and self.empty_themes == 0
            and self.orphan_themes == 0
        )
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/test_layer_b_audit.py tests/unit/test_layer_b_models.py -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/foodscholar/layer_b/audit.py src/foodscholar/layer_b/models.py tests/unit/test_layer_b_audit.py
git commit -m "M5 (cross-shelf/p3): audit handles cross-shelf themes + orphan_themes gate"
```

---

## Task 11: Update integration test for end-to-end cross-shelf

**Files:**
- Modify: `tests/integration/test_layer_b_pipeline.py`

- [ ] **Step 1: Read the existing pipeline test**

```bash
cat tests/integration/test_layer_b_pipeline.py
```

- [ ] **Step 2: Add a cross-shelf assertion**

In the existing end-to-end test, after `fs.build_layer_b(...)`, add:

```python
    themes = fs.graph_store.list_themes()
    cross_shelf_themes = [t for t in themes if len(t.shelf_ids) >= 2]
    print(f"cross-shelf themes: {len(cross_shelf_themes)} of {len(themes)}")
    # We don't assert a minimum count here (it depends on the test fixture's
    # corpus) but we do assert that the data model supports it.
    assert all(len(t.shelf_ids) >= 1 for t in themes), \
        "every theme must be reachable from at least one shelf"
```

- [ ] **Step 3: Run the integration test**

```bash
FOODSCHOLAR_RUN_INTEGRATION=1 pytest tests/integration/test_layer_b_pipeline.py -v
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_layer_b_pipeline.py
git commit -m "M5 (cross-shelf/p4): integration test asserts shelf reachability"
```

---

## Task 12: Update the brief + PROGRESS to reflect v2

**Files:**
- Modify: `layer_b_construction_brief.md`
- Modify: `PROGRESS.md`

- [ ] **Step 1: Update the brief**

In `layer_b_construction_brief.md`:
- §5 ("Out of scope: cross-shelf themes") → update to say "Cross-shelf themes are now first-class via hybrid global/per-shelf design (added 2026-05-27)."
- §6 (orchestrator) → describe the new flow.
- §10 (audit) → add the orphan_themes gate.

(Specific edits to be made when running this task — they should follow the prose style of the surrounding sections.)

- [ ] **Step 2: Add a PROGRESS entry**

Use the `progress-log` skill (`/progress-log` or by mention) to add an iteration entry summarizing the cross-shelf rollout, with bullet points covering:
- Pass 1 went global; Pass 2 stayed per-shelf
- New ChunkStore.knn_search_chunks (ES kNN + InMemory)
- Audit gained orphan_themes
- v0.2 discovery_version (since theme contracts changed)

- [ ] **Step 3: Commit**

```bash
git add layer_b_construction_brief.md PROGRESS.md
git commit -m "M5 (cross-shelf/p4): document v2 brief + PROGRESS entry"
```

---

## Task 13: Wipe + rebuild themes in the live notebook

**Files:** None modified — this is a runbook step.

After all code commits land, re-run Layer B against the live ES+Neo4j:

- [ ] **Step 1: In the notebook, restart kernel**
- [ ] **Step 2: Run the `fs = FoodScholar(...)` cell with `llm=build_llm(LLMConfig(primary=ProviderConfig(provider="groq", ...)))` so the new themes get real Groq labels (not the mock).
- [ ] **Step 3: Run the audit cell to confirm the existing Layer A + chunks are fine.
- [ ] **Step 4: Run `artifact = fs.build_layer_b(facet="foods")`. Expected: takes 2-4 minutes; logs one global pass and ~110 per-shelf relatedness passes.
- [ ] **Step 5: Print summary:

```python
themes = fs.graph_store.list_themes()
single = sum(1 for t in themes if len(t.shelf_ids) == 1)
multi = sum(1 for t in themes if len(t.shelf_ids) >= 2)
print(f"total themes: {len(themes)}  single-shelf: {single}  cross-shelf: {multi}")
print("biggest cross-shelf themes:")
for t in sorted(themes, key=lambda x: -len(x.shelf_ids))[:10]:
    print(f"  {t.label!r:40s} ({len(t.shelf_ids)} shelves, {t.chunk_count} chunks)")
```

Expected: a non-trivial cross-shelf count (target: ≥20% of themes span 2+ shelves on this corpus).

- [ ] **Step 6: Run the audit cell — must PASS including the new orphan_themes gate.

---

## Self-review notes (pre-execution)

- **Spec coverage:** All three brainstorming decisions are addressed — hybrid pass design (Tasks 6-9), attached-chunks-only scope (Task 9 builds `attached_chunk_ids` from the attachments map), and ES kNN backend (Task 3).
- **No placeholders:** Every step has executable code or commands. The two prose-update tasks (Task 12) are inherently narrative but reference specific brief sections.
- **Type consistency:** `discovery_pass` is `"global_similarity"` everywhere (Theme model, ThemeCandidate, builder, merge, audit). `shelf_ids: list[str]` consistently. The `LayerBConfig.global_similarity_max_chunks` knob is referenced consistently by name.
- **Open risks for the implementing engineer to watch:**
  - `merge_candidates` may need a small internal change in Task 8 to carry `pass_name="global_similarity"` into the output theme dict; read it first before assuming.
  - The Task 8 `merged_to_rel_idxs` correlation between MergeDecisions and emitted themes is fragile if merge_candidates ever reorders; if Task 8 tests reveal a bug here, switch to a stable chunk_ids-based join.
  - `pick_primary` may have a hard-coded check for `discovery_pass in ("similarity", "merged")` — extend to include `"global_similarity"` if so.
  - The integration test at Task 11 doesn't assert a minimum cross-shelf theme count because that depends on the corpus; manual verification happens in Task 13 step 5.
