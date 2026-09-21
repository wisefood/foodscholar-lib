# Glossary

The vocabulary used across these docs, in one place. Terms link to where they're
explained in depth.

```{glossary}
chunk
  A passage-sized piece of a source document — the atom of the corpus. See
  [Corpus input](corpus-input.md).

grounding
  Mapping an extracted entity *surface form* onto an ontology id with the
  linker, so relation endpoints join the same entity universe as the rest of
  the graph. Without it, Layer 0 would hold free-text strings that nothing
  else can resolve. See [Layer 0](layer-0-relations.md).

NIL endpoint
  A relation endpoint whose surface form did not clear the grounding
  threshold, recorded as `NIL:<slug>` with `subject_linked`/`object_linked`
  set to `False`. Kept rather than dropped — discarding them would silently
  delete every relation touching a concept the ontology lacks. The frequent
  ones name real ontology gaps.

predicate
  The relation type in a triple — the verb phrase linking subject to object
  (`reduces`, `contains`, `associated with`). The vocabulary is open, so watch
  its cardinality via `fs.relations.predicates()`.

relation
  A typed, corpus-grounded edge between two entities:
  `(subject_id, predicate, object_id)` plus the chunks it was extracted from.
  The unit of [Layer 0](layer-0-relations.md).

facet
  One of six independent slices the graph is projected into: `foods`, `health`,
  `nutrients`, `dietary_patterns`, `allergies`, `sustainability`. A chunk's entity
  links route to the relevant facet(s). In the current corpus only **`foods`** is
  meaningfully populated (FoodOn is a food ontology); the others are scaffolded. See
  [Layer A](layer-a-backbone.md).

shelf
  A node in the Layer A backbone — one FoodOn class kept because the corpus has
  evidence for it. Shelves form the coarse, browsable menu. See
  [Layer A](layer-a-backbone.md).

backbone
  Used in two senses. (1) *The ontology backbone*: FoodOn itself, which the graph is
  projected onto. (2) *The backbone projection* (a.k.a. **1a+**): the production Layer A
  method that picks a facet root's supported children and expands down the real FoodOn
  tiers. Context disambiguates; this glossary entry exists because they collide.

1a+
  The name of the production Layer A construction method (`projection="backbone"`). It
  comes from the method bake-off (method "1a", plus refinements). Synonymous with
  *backbone projection*.

support (direct vs lifted)
  The chunk evidence behind a shelf. **Direct support** = chunks that mention this
  exact FoodOn class. **Lifted support** = evidence inherited from descendants by
  walking *up* the is-a tree (a chunk mentioning `cow milk` lifts to `mammalian milk
  product`, `dairy food product`, …). A shelf with high lifted but ~0 direct support is
  an organizational **umbrella**; high direct support marks a genuine topic.

lifted attachment
  Distinct from lifted support. A *chunk* attaches to every shelf on its lift path, so
  one chunk can sit on **multiple shelves**. (Lifted support is about a *shelf's*
  evidence count; lifted attachment is about a *chunk's* multi-shelf membership.) This
  multi-attachment is why Layer B themes are tied to an {term}`origin shelf`.

umbrella class / umbrella rule
  An organizational FoodOn class (`food product`, `vertebrate food product`) that
  absorbs generic mentions — high lifted, low direct support. The *umbrella rule* in the
  prune cascade drops such inflated classes. See [Layer A](layer-a-backbone.md).

filing tier
  An intermediate is-a node that exists only to organize, with a single child and no
  navigational value. The backbone projection **collapses** these.

theme
  A Layer B node — a fine-grained topic community discovered inside a shelf. See
  [Layer B](layer-b-themes.md).

ThemeCandidate
  A community emitted by one Layer B pass *before* merging. The merge turns candidates
  into final `Theme`s. Each candidate carries its `origin_shelf_id`.

discovery_pass
  How a theme was found: `relatedness` (entity pass only), `global_similarity`
  (embedding pass only), or `merged` (both agreed). **Naming landmine:** the value
  `global_similarity` is a historical name — in the production per-shelf mode it means
  *per-shelf embedding similarity*, not a global pass. See [Layer B](layer-b-themes.md).

origin shelf
  The shelf whose chunks a per-shelf theme was built from. A theme attaches to its
  origin shelf, **not** the union of its members' shelves — avoiding cross-shelf
  smearing. See [Layer B](layer-b-themes.md).

card
  A Layer C node — a short, fully-cited LLM write-up for a shelf or theme. See
  [Layer C](layer-c-cards.md).

NER
  Named-entity recognition — finding food/health mention spans in text. FoodScholar uses
  GLiNER. See [Annotation](annotation.md).

NEL
  Named-entity linking — resolving a mention to a FoodOn id. FoodScholar uses a
  single-tier dense (HNSW) linker. Pre-computed NEL can be supplied as CSVs.

c-TF-IDF
  Class-based TF-IDF: TF-IDF where each *theme* (not each document) is one "document".
  Used to pick a theme's discriminative keyword label.

RRF
  Reciprocal-rank fusion — the method that blends BM25 and kNN result lists into one
  hybrid ranking at retrieval time.

denormalization
  Copying a chunk's `shelf_ids` / `theme_ids` onto its Elasticsearch document so
  retrieval can filter without round-tripping to Neo4j. Kept consistent by audit
  invariants. See [Architecture](architecture.md).
```
