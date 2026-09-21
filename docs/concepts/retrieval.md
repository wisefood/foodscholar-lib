# Retrieval

FoodScholar **retrieves and stops there**. `fs.retrieve()` returns ranked
passages with the scores that ranked them; it does not write an answer.

That boundary is deliberate. Formulating an answer means owning a model, a
prompt registry, a citation format, a safety policy and an editorial gate —
all of which belong to the service asking the question, and none of which the
library can see. A second answering stack in here would be a fork of the one
its consumer already runs, drifting from it release by release.

```python
hits, trace = fs.retrieve("Is olive oil heart-healthy?", k=5)

for h in hits:
    h.chunk.text          # the passage
    h.score               # weighted total
    h.text_sim            # why: it reads like the question
    h.triplet_sim         # why: it asserts something about the question
    h.ppr_score           # why: it sits near the question in the graph
```

## The three branches

The scoring is the **Extended KG-Gen** hybrid, ported from the WiseFood
pipeline (see [Extended KG-Gen](extended-kg-gen.md)):

| branch | weight | question it answers | reads |
|---|---|---|---|
| text similarity | 0.3 | does this passage *read like* the query? | `ChunkStore.knn_search_chunks` |
| mean triplet similarity | 0.3 | does it *assert something* about the query? | `RelationStore.for_chunks` |
| Personalized PageRank | 0.4 | does it sit in the right *neighbourhood*? | `RelationStore.for_entity` |

The three answer genuinely different questions, which is the point of running
all of them. A passage about "sodium and hypertension" scores poorly on text
similarity for the query "salt and blood pressure" and highly on the other
two. Each branch is min-max normalized over the **whole candidate pool**
before weighting, so a branch that separates nothing contributes nothing
rather than handing its full weight to everyone equally.

Only branch 1 needs the corpus. Branches 2 and 3 need
[Layer 0 relations](layer-0-relations.md): if `build_relations()` has not run,
they stay silent and retrieval degrades to plain kNN — which is still a usable
ranking, so a corpus mid-build answers rather than fails.

## Why it reads the stores

The reference implementation loaded a GraphML file plus a `passages.json` and
cached ~520MB of embeddings beside them, so every deployment had to mount a
matching pair of artifacts and keep them in step with the graph. Here each
branch reads the store that already owns the data, so a graph rebuild is
visible immediately with nothing to re-sync.

The trade is a different cost model, worth stating plainly. The reference
scored *every* passage in the corpus because it held them all in memory. This
scores a **candidate pool** — `cfg.retrieval.candidate_k` chunks from the text
branch, which the other two re-rank. A passage outside that pool cannot be
retrieved however well it would have scored on the graph. `candidate_k` is
therefore the recall knob, and the one to raise before `subgraph_depth`.

## Cost, and the bounds on it

A naive version of this is slow and, worse, unboundedly slow. Three limits:

- **`max_relations`** (500) caps the triples embedded per query,
  best-supported first. A pool of 100 chunks can carry a few thousand triples,
  and the tail is single-mention noise that moves no ranking.
- **`max_expansion_calls`** (50) caps `for_entity` round-trips. Hop 2's
  frontier is up to `seed_entities × expand_k × 2` entities, so without this a
  `subgraph_depth` of 2 would issue a thousand store calls for one question.
  Hitting the ceiling stops the walk and scores the subgraph gathered so far.
- **`embed_cache_size`** (50 000) is a process-local, thread-safe LRU over
  triple and entity-label embeddings. Those strings are corpus artifacts that
  change only when `build_relations()` re-runs, so a warm replica re-encodes
  almost nothing. The query itself is deliberately *not* cached — every query
  is new, so it would only evict useful entries.

`subgraph_depth` defaults to **1**, not the reference's 5, for the same
reason: the reference walked an in-memory graph where a hop was free.

## Reading a ranking

`retrieve()` returns a `RetrievalTrace` alongside the hits, so a result can be
reviewed rather than taken on trust: candidate count, relations found and
scored, subgraph size, expansion calls spent, the seed entities PPR started
from, and which branches actually contributed.

`trace.degraded` is the one to watch in production. It is non-empty when the
ranking is less trustworthy than a healthy one — most importantly when the
facade fell back to its **hash embedder** because the real one could not
build. That failure is otherwise invisible: retrieval still returns a
confidently-ordered list, of nonsense.

## Configuration

Everything above is `cfg.retrieval`; see
[configuration](../getting-started/configuration.md) and the annotated block
in `config.example.yaml`. The three weights **must sum to 1.0** — the loader
refuses a config that does not, rather than silently skewing every ranking.

## Not in scope

Answer synthesis, citation rendering and grounding checks. An `Answer` model
and an `fs.query()` phase were planned here and have been withdrawn: the
consumer that would have used them, the FoodScholar API, brings its own.
