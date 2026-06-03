# The FoodOn ontology

[FoodOn](https://foodon.org) is the backbone FoodScholar projects onto — a ~39k-term
ontology of food products with a real `is-a` hierarchy. `fs.ontology` is the lookup API
used by the linker, the Layer A projection, and Layer C prompts.

## Lazy loading & caching

The ontology loads on first access to `fs.ontology`: it parses
`config.ontology.foodon_path` (OWL, via [pronto](https://pronto.readthedocs.io)) and
caches the result to a Parquet file beside it, so subsequent runs skip the parse.

```yaml
ontology:
  foodon_path: data/foodon.owl
  cache_path: data/foodon_cache.parquet
  prefix_filter: ["FOODON:"]    # restrict to a prefix, or null to load all
```

## Lookup API

```python
fs.ontology.name_to_id("olive oil")                  # "FOODON:..." | None
fs.ontology.id_to_label("FOODON:03309927")
fs.ontology.id_to_synonyms("FOODON:03309927", include_related=True)
fs.ontology.id_to_ancestors("FOODON:03309927")       # closed transitive set
fs.ontology.id_to_descendants("FOODON:00001002")
fs.ontology.search("olive", limit=25)
```

`id_to_ancestors` is what Layer A walks to build [*lifted* support](glossary.md) — a
chunk mentioning `cow milk` counts as evidence for every class above it:

```{mermaid}
flowchart TD
    A[cow milk] --> B[mammalian milk product]
    B --> C[milk or milk based food product]
    C --> D[dairy food product]
    D --> E[vertebrate food product]
    E --> F[animal food product]
    F --> G[Foods]
```

```python
fs.ontology.id_to_ancestors("FOODON:03310029")
# {"FOODON:03315150",   # mammalian milk product
#  "FOODON:00001257",   # milk or milk based food product
#  "FOODON:00004242",   # animal food product
#  "FOODON:00001002", ...}   # ... up to Foods (the closed transitive set)
```

`name_to_id` and the synonym lookups are exact-match conveniences; the production **NEL
linker does not use them** — it resolves mentions by dense kNN over term embeddings (see
[Annotation](annotation.md)).

## One ontology, mostly one facet

FoodOn is a **food** ontology. That's why the `foods` facet is rich (a few hundred
shelves) while `health`, `nutrients`, `dietary_patterns`, `allergies`, and
`sustainability` are **scaffolded but essentially unpopulated** in the current corpus —
there's no food-class backbone to project them onto. `prefix_filter: ["FOODON:"]` keeps
the loaded ontology to FOODON ids (real `.owl` files embed CHEBI/NCBITaxon/BFO terms
inline), which the food linker wants — but it also means non-food facets have no ontology
to draw from here. Treat `foods` as the production facet today; the multi-facet design is
in place for ontologies that would back the others.

```{tip}
For tests and notebooks you can skip the loader entirely and attach an in-memory API
built from a handful of terms:

    from foodscholar.ontology import FoodOnAPI
    fs.attach_ontology(FoodOnAPI(terms))

That's how the unit tests exercise Layer A against a tiny fixture ontology with no OWL
file at all.
```
