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

`id_to_ancestors` is what Layer A walks to build *lifted* support — a chunk mentioning
`olive oil` counts as evidence for `vegetable oil` and every class above it. `name_to_id`
and the synonym lookups are the linker's tiers 1–2.

```{tip}
For tests and notebooks you can skip the loader entirely and attach an in-memory API
built from a handful of terms:

    from foodscholar.ontology import FoodOnAPI
    fs.attach_ontology(FoodOnAPI(terms))

That's how the unit tests exercise Layer A against a tiny fixture ontology with no OWL
file at all.
```
