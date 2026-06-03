# Command line

The `foodscholar` CLI wraps the same facade methods as the Python API — each command is
a thin wrapper around one phase. Every command takes `--config` (the YAML from
[](../getting-started/configuration.md)).

```bash
foodscholar info          --config config.yaml   # versions + resolved backends
foodscholar init          --config config.yaml   # provision the stores
foodscholar annotate      --config config.yaml   # NER + linking + embeddings
foodscholar build-layer-a --config config.yaml   # FoodOn-projected shelves
foodscholar attach        --config config.yaml   # attach chunks to shelves
foodscholar build-layer-b --config config.yaml   # per-shelf themes
foodscholar build-layer-c --config config.yaml   # cited cards
foodscholar build-all     --config config.yaml   # the full pipeline, in order
foodscholar query "Is olive oil heart-healthy?" --config config.yaml
foodscholar version
```

`build-all` runs the phases end to end; the individual `build-*` commands let you re-run
a single stage after changing its config (e.g. re-run `build-layer-b` after tuning the
Layer B knobs). Because every command loads the same config, the resolved backends shown
by `info` are exactly what each phase will use.

```{tip}
The CLI is the natural entry point for scheduled / CI builds. For interactive
exploration and visualization, the Python API and
[`notebooks/graph_build.ipynb`](building-the-graph.md) are more ergonomic.
```
