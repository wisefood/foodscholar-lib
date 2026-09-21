# Command line

The `foodscholar` CLI wraps the same facade methods as the Python API — each command is
a thin wrapper around one phase. Every command takes `--config` (the YAML from
[](../getting-started/configuration.md)).

```bash
foodscholar info          --config config.yaml   # versions + resolved backends
foodscholar init          --config config.yaml   # provision the stores
foodscholar chunk-corpus  --config config.yaml   # PDFs -> corpus CSVs (see below)
foodscholar annotate      --config config.yaml   # NER + linking + embeddings
foodscholar build-relations --config config.yaml # Layer 0 typed relations
foodscholar build-layer-a --config config.yaml   # FoodOn-projected shelves
foodscholar attach        --config config.yaml   # attach chunks to shelves
foodscholar build-layer-b --config config.yaml   # per-shelf themes
foodscholar build-layer-c --config config.yaml   # cited cards
foodscholar build-all     --config config.yaml   # the full pipeline, in order
foodscholar sweep-layer-b  --config config.yaml  # grid-search the Layer B knobs
foodscholar report-layer-b --config config.yaml  # Layer B coverage/quality report
foodscholar bench-layer-c  --config config.yaml  # score Layer C cards against their chunks
foodscholar query "Is olive oil heart-healthy?" --config config.yaml  # retrieve (see below)
foodscholar version
```

`query` retrieves; it does not answer. It prints the ranked passages as JSON, each with
its three branch scores and the retrieval trace, and takes `--top-k/-k`. Formulating an
answer is the calling application's job — see [](../concepts/retrieval.md).

Three commands are **not** part of `build-all`:

- `chunk-corpus` produces the corpus `build-all` consumes, so it runs *before*
  the pipeline. It takes `--pdf-dir`, `--out-dir`, `--source-type`,
  `--metadata-csv` and `--excluded-pages`. Re-chunking an ingested corpus
  assigns new chunk ids and orphans existing relations and attachments, so it
  refuses to overwrite without `--force`. See
  [](chunking-a-corpus.md).
- `build-relations` costs an LLM pass over the whole corpus, so it is opt-in
  (`relations.enabled: true`). Use `--dry-run` to extract and report without
  writing, and `--force` to re-extract chunks already covered. See
  [](../concepts/layer-0-relations.md). Skipping it leaves `query` with only its
  text branch — see [](building-the-graph.md).
- `query` reads the built graph rather than writing to it, so it runs after the
  pipeline rather than as part of it.

`sweep-layer-b`, `report-layer-b` and `bench-layer-c` are tuning and evaluation tools
rather than build phases — see [](tuning-layer-b.md) for the first two.

`build-all` runs the phases end to end; the individual `build-*` commands let you re-run
a single stage after changing its config (e.g. re-run `build-layer-b` after tuning the
Layer B knobs). Because every command loads the same config, the resolved backends shown
by `info` are exactly what each phase will use.

```{tip}
The CLI is the natural entry point for scheduled / CI builds. For interactive
exploration and visualization, the Python API and
[`notebooks/graph_build.ipynb`](building-the-graph.md) are more ergonomic.
```
