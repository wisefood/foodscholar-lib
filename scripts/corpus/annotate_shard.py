"""One shard of the annotate phase, in its own process.

A module rather than a notebook closure on purpose: ``ProcessPoolExecutor``
pickles the callable by qualified name, and under the ``spawn`` start method a
function defined in a notebook cell cannot be re-imported by the child. Spawn
is not optional here — CUDA and ``fork`` do not mix, and a forked child that
touches an already-initialised CUDA context deadlocks or corrupts it.

Each worker is self-contained: it builds its own FoodScholar from the same
config dict, so it has its own GLiNER, its own linker and its own embedder.
That is the cost of the parallelism (roughly 3-4 GB of RAM per worker) and the
reason the NEL index is built once beforehand — workers load it from disk
instead of re-encoding FoodOn each.
"""
from __future__ import annotations

import os
from typing import Any


def annotate_shard(payload: tuple[dict[str, Any], list[str]]) -> dict[str, Any]:
    """Annotate one shard of chunk ids and write the results back to the store.

    Returns a summary rather than the chunks themselves: shipping annotated
    chunks back through the pool would pickle every 768-float embedding across
    a process boundary for no reason — the worker has already persisted them.
    """
    config_dict, chunk_ids = payload

    # Threads per worker. Left to its own devices torch grabs every core in
    # every process at once, and N workers each spawning N threads is slower
    # than one. Set before torch is imported, which is why this is at the top.
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("FS_WORKER_THREADS", "2"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from foodscholar import FoodScholar
    from foodscholar.annotate import runner
    from foodscholar.storage.memory import InMemoryChunkStore

    fs = FoodScholar.from_config(config_dict)
    store = fs.chunk_store

    chunks = store.get_many(list(chunk_ids))
    if not chunks:
        return {"requested": len(chunk_ids), "annotated": 0, "mentions": 0, "links": 0}

    # The runner annotates everything in the store it is handed, so the shard
    # goes into a scratch store. This reuses the library's exact annotate path
    # — same batching, same NER, same linker — without reaching into it.
    scratch = InMemoryChunkStore()
    scratch.upsert(chunks)
    meta = runner.run(
        scratch,
        ner=fs.ner,
        linker=fs.linker,
        embedder=fs.embedder,
        config=fs.config,
    )

    annotated = scratch.scan()
    store.upsert(annotated)

    # Counted here rather than read off `meta`: ArtifactMeta carries
    # `record_count` and nothing else, so mention and link totals have to come
    # from the chunks. They are the phase's only quality signal — a shard that
    # annotated every chunk and linked none is a broken NEL index, not a
    # successful run.
    return {
        "requested": len(chunk_ids),
        "annotated": len(annotated),
        "mentions": sum(len(c.mentions or []) for c in annotated),
        "links": sum(len(c.entity_links or []) for c in annotated),
        "record_count": meta.record_count,
    }
