"""One shard of the annotate phase, in its own process.

A module rather than a notebook closure on purpose: ``ProcessPoolExecutor``
pickles the callable by qualified name, and under the ``spawn`` start method a
function defined in a notebook cell cannot be re-imported by the child. Spawn
is not optional here — CUDA and ``fork`` do not mix, and a forked child that
touches an already-initialised CUDA context deadlocks or corrupts it.

Two entry points, because the GPU and the databases are not always the same
machine:

``annotate_corpus_shard``
    **File in, file out.** Reads a corpus shard, annotates it, writes an
    annotated parquet. Needs no Elasticsearch, no Neo4j and no network — the
    stores are in memory. This is the one to use on a GPU box that cannot
    reach the cluster.

``annotate_store_shard``
    **Store in, store out.** Reads a shard of chunk ids from the configured
    chunk store, annotates it, writes the results back. For the case where one
    machine can see both the GPU and the database.

Each worker is self-contained: it builds its own FoodScholar from the same
config dict, so it has its own GLiNER, its own linker and its own embedder.
That is the cost of the parallelism (roughly 3-4 GB of RAM per worker) and the
reason the NEL index is built once beforehand — workers load it from disk
instead of re-encoding FoodOn each.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _set_thread_limits() -> None:
    """Cap threads per worker, before torch is imported.

    Left to its own devices torch grabs every core in every process at once,
    and N workers each spawning N threads is slower than one worker. This has
    to run before the first torch import, which is why every entry point calls
    it first and the heavy imports are function-local.
    """
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("FS_WORKER_THREADS", "2"))
    os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("FS_WORKER_THREADS", "2"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _annotate(fs: Any, chunks: list[Any]) -> list[Any]:
    """Run the library's annotate pass over `chunks` and return the enriched copies.

    The runner annotates everything in the store it is handed, so the shard
    goes into a scratch store. This reuses the library's exact annotate path —
    same batching, same GLiNER, same HNSW linker, same embedder — rather than
    reimplementing any of it.
    """
    from foodscholar.annotate import runner
    from foodscholar.storage.memory import InMemoryChunkStore

    scratch = InMemoryChunkStore()
    scratch.upsert(chunks)
    runner.run(
        scratch,
        ner=fs.ner,
        linker=fs.linker,
        embedder=fs.embedder,
        config=fs.config,
    )
    return scratch.scan()


def _summarise(requested: int, annotated: list[Any]) -> dict[str, Any]:
    """Counts callers can act on.

    Mentions and links are counted from the chunks rather than read off the
    phase's ArtifactMeta, which carries `record_count` and nothing else. They
    are the only quality signal this phase produces: a shard that annotated
    every chunk and linked none is a broken NEL index, not a successful run.
    """
    return {
        "requested": requested,
        "annotated": len(annotated),
        "mentions": sum(len(c.mentions or []) for c in annotated),
        "links": sum(len(c.entity_links or []) for c in annotated),
    }


def annotate_corpus_shard(payload: tuple[dict[str, Any], str, str]) -> dict[str, Any]:
    """Annotate one corpus shard file and write an annotated parquet.

    Skips the work entirely when the output already exists and is non-empty,
    which is what makes an interrupted fan-out resumable: re-running the driver
    only picks up the shards that never finished.
    """
    config_dict, in_path, out_path = payload
    _set_thread_limits()

    out = Path(out_path)
    if out.exists() and out.stat().st_size > 0:
        return {"shard": out.name, "skipped": True, "requested": 0,
                "annotated": 0, "mentions": 0, "links": 0}

    from foodscholar import FoodScholar
    from foodscholar.corpus import load_chunks, write_chunks_parquet

    fs = FoodScholar.from_config(config_dict)
    chunks = load_chunks(in_path)
    if not chunks:
        return {"shard": out.name, "skipped": False, "requested": 0,
                "annotated": 0, "mentions": 0, "links": 0}

    annotated = _annotate(fs, chunks)

    # Written to a temporary name and moved into place, so an interrupted write
    # cannot leave a truncated parquet that the skip-if-exists check above
    # would then treat as a finished shard.
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".partial")
    write_chunks_parquet(annotated, tmp)
    tmp.replace(out)

    return {"shard": out.name, "skipped": False, **_summarise(len(chunks), annotated)}


def annotate_store_shard(payload: tuple[dict[str, Any], list[str]]) -> dict[str, Any]:
    """Annotate one shard of chunk ids, reading and writing the configured store.

    Returns a summary rather than the chunks themselves: shipping annotated
    chunks back through the pool would pickle every 768-float embedding across
    a process boundary for no reason — the worker has already persisted them.
    """
    config_dict, chunk_ids = payload
    _set_thread_limits()

    from foodscholar import FoodScholar

    fs = FoodScholar.from_config(config_dict)
    store = fs.chunk_store

    chunks = store.get_many(list(chunk_ids))
    if not chunks:
        return {"requested": len(chunk_ids), "annotated": 0, "mentions": 0, "links": 0}

    annotated = _annotate(fs, chunks)
    store.upsert(annotated)
    return _summarise(len(chunk_ids), annotated)
