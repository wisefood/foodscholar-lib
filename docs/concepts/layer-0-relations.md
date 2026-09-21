# Layer 0 — Relations

Layer 0 is a **typed, corpus-grounded, ontology-anchored edge set** over the
entities FoodScholar already discovers. It answers a question the rest of the
graph cannot: not *what does this chunk mention*, but *what does this chunk
say about the things it mentions*.

```
Layer C   cards          cited write-ups
Layer B   themes         per-shelf topic communities
Layer A   shelves        the FoodOn backbone
──────────────────────────────────────────────────────────────
Layer 0   relations      (:Entity)-[:RELATED {predicate}]->(:Entity)
          entities       (:Entity)
          chunks         (:Chunk)
```

It sits **under** the entity graph, not beside Layer C. Nothing in Layer A/B/C
depends on it, and it depends on nothing above `build_entities()`.

## Why it exists

Before Layer 0, the closest thing to a relationship in FoodScholar was Layer
B's relatedness graph, which weights chunk pairs by *shared FoodOn ids*. That
knows olive oil and LDL cholesterol co-occur. It cannot know that one
**lowers** the other.

```python
# what the entity graph records
(:Chunk)-[:MENTIONS]->(:Entity {ontology_id: "FOODON:03301710"})
(:Chunk)-[:MENTIONS]->(:Entity {ontology_id: "CHEBI:39026"})

# what Layer 0 adds
(:Entity {FOODON:03301710})-[:RELATED {predicate: "reduces"}]->(:Entity {CHEBI:39026})
```

Every edge keeps the chunks it came from, so a relation is as traceable as any
other FoodScholar claim.

## Grounding — the part that matters

The extractor is an LLM, and LLMs emit **strings**:

```
("olive oil", "reduces", "LDL cholesterol")
```

FoodScholar's entities are **ontology ids**. Left alone you would have two
disjoint entity universes in one graph. So every endpoint is run through the
*same* linker that produced every `EntityLink` in the corpus —
`HNSWLinker` over the FoodOn term index:

```
"olive oil"       --linker-->  FOODON:03301710
"LDL cholesterol" --linker-->  CHEBI:39026
```

A relation therefore connects the same `Entity` records Layer A projects onto
shelves and Layer B clusters into themes. That is what makes Layer 0 part of
FoodScholar rather than a second graph living beside it.

### NIL endpoints are kept

Surfaces that do not clear `relations.grounding.min_sim` become
`NIL:<slug>` rather than being dropped:

```python
relation.object_id      # "NIL:hba1c"
relation.object_linked  # False
```

Dropping them would silently delete every relation touching a concept FoodOn
lacks — for a nutrition corpus that is most biomarkers, hormones and
physiological processes. Consumers filter on the `*_linked` flags; the data
layer stays faithful. This mirrors how Layer A marks unsupported nodes
`status="absent"` instead of deleting them.

The **top NIL surfaces are the most valuable diagnostic this phase produces**:
they name what your ontology is missing. `build_relations` logs them.

## The pipeline

```
chunk_store.iter_chunks
   │
   ├─▶ extract    2 LLM calls per chunk: entities, then relations
   ├─▶ dedupe     collapse surface + predicate variants, provenance unions
   ├─▶ ground     surfaces -> ontology ids, one batched linker call
   ├─▶ aggregate  group by (subject_id, predicate, object_id)
   └─▶ persist    RelationStore + (:Entity)-[:RELATED]->(:Entity)
```

Two ordering decisions are deliberate:

- **Dedup before grounding.** Dedup shrinks the distinct surface set the linker
  must encode; grounding then collapses further, since different surfaces
  often share one id. The reverse order does strictly more encoder work.
- **Extraction is the only nondeterministic step.** Everything after a fixed
  extraction is reproducible, which is what the determinism test pins.

### Relation extraction constrains its own endpoints

The relation call's JSON schema restricts `subject` and `object` to an **enum
of the entities from step 1**, so the model picks endpoints from the list
rather than inventing them. A second filter drops any that slip through
anyway. Without this, invented endpoints become NIL nodes and quietly pollute
the graph.

## Provenance

The field the whole layer exists for:

```python
relation.chunk_ids    # ('c1', 'c2') — capped sample
relation.chunk_count  # 2 — the true total
```

When dedup collapses two triples, their chunk sets **union**. A triple seen in
three chunks, merged with a variant seen in two more, carries five.

**Invariant:** every id in `chunk_ids` resolves in the chunk store. Relations
and chunks share one id space because the extractor is handed the corpus the
library already holds.

## Using it

```python
fs = FoodScholar.from_config("config.yaml")
fs.build_entities()
fs.build_relations()                       # or: foodscholar build-relations

fs.relations.for_entity("FOODON:03301710") # edges touching olive oil
fs.relations.for_chunks(["chunk-1"])       # what this passage asserts
fs.relations.by_predicate("reduces")
fs.relations.grounded()                    # both endpoints real ontology ids
fs.relations.predicates()                  # [(predicate, count), ...]
fs.relations.summary()
```

`fs.build_relations(dry_run=True)` extracts, grounds and reports without writing —
the notebook path, and the only way the mock LLM is allowed near this phase.

### Prerequisites

`build_relations` needs three things, and the error you hit tells you which is
missing:

| needs | provided by | otherwise |
|---|---|---|
| chunks in the chunk store | `fs.ingest(...)` | nothing to extract |
| an **LLM** | `llm:` in config | refuses with *"needs a real LLM"* (the mock's output is meaningless); `dry_run=True` overrides |
| the **linker** (for grounding) | `ontology.foodon_path` + the `[annotate]` extra — or `fs.attach_linker(...)` | *"no ontology section in config"* |

So on a bare `FoodScholar.in_memory()` the first thing you hit is the **ontology**,
not the LLM. For a no-services smoke test, attach stubs — this is exactly what the
end-to-end unit test does:

```python
fs = FoodScholar.in_memory()
fs.attach_linker(my_stub_linker)   # anything implementing `Linker`
fs.llm = my_stub_llm               # anything implementing `LLMClient`
fs.build_relations(dry_run=True)
```

The first real run also pays for building the HNSW index over FoodOn (minutes);
later runs load it from the cached `nel_index_path`.

Re-runs skip chunks already covered: **the store is the resume log**, so an
interrupted corpus run continues where it stopped. `force=True` re-extracts
everything. Relation ids are content-addressed
(`sha1(subject, predicate, object)`), so a re-run over an unchanged corpus
rewrites identical records rather than duplicating them.

## Cost

Two LLM calls per chunk. On a corpus of ~14k chunks that is ~28k calls — hours
of local GPU time, or a real invoice on a hosted API. `relations.enabled`
defaults to **false** and `fs.build()` skips the phase, precisely so it is
never triggered by surprise. Measure on a few hundred chunks before committing
to the full corpus:

```python
fs.build_relations(chunk_ids=[c.chunk_id for c in some_sample], dry_run=True)
```

Point `relations.llm` at a cheap local model; it is a separate provider from
`llm.primary`, which Layer C wants to be strong.

## Watch the predicate cardinality

The extractor uses an **open** predicate vocabulary — faithful to the source
text, but it sprawls. After a real run:

```python
fs.relations.predicates(k=50)
```

A corpus yielding thousands of distinct predicates is telling you to close the
set. This also matters for retrieval: distinct predicates between the same
pair count as parallel edges, so under-merging them biases ranking.

## Provenance

Layer 0 is adapted from the **kggen** pipeline developed by WiseFood colleagues
(not distributed with FoodScholar): the extraction prompts are theirs
verbatim, the extraction steps and the provenance-preserving dedup are ports,
and the passage-per-triple idea is the reason the layer exists. The grounding
step is FoodScholar's addition — it is what binds kggen's free-text entities to
this graph's ontology ids. See [Extended KG-Gen](extended-kg-gen.md)
for exactly what was taken and what changed, and the
[relations reference](../reference/relations.md) for the ported modules.

## See also

- [Annotation](annotation.md) — the NER and linker Layer 0 reuses
- [Architecture](architecture.md) — where Layer 0 sits
- [Glossary](glossary.md) — relation, predicate, grounding, NIL endpoint
