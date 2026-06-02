"""Content-addressed annotation cache — the reproducible artifact.

The agentic annotation pipeline is non-deterministic, but BRIEF §13 demands
idempotent reruns. The resolution (docs/DESIGN_agentic_annotate.md §4): the
cache *is* the artifact.

  - **Key** — `sha256(chunk_text + agent_model_id + prompt_version +
    ontology_hash)`. Any change to the corpus text, the model, the prompt, or
    the ontology changes the key, so a stale entry is never silently reused.
  - **Value** — the full `list[EntityLink]` the agent produced for that chunk.
  - **Behaviour** — `fs.annotate()` looks up each chunk; a hit replays the
    stored links at zero LLM cost, a miss runs the agent and stores the result.

Re-running annotate over an unchanged corpus with an unchanged model / prompt
/ ontology is therefore a pure cache replay: deterministic, free, idempotent.

Storage is **SQLite** (decision log, docs/DESIGN_agentic_annotate.md): indexed
point lookups + incremental upserts match the access pattern; Parquet
(immutable, bulk-scan) does not. The DB is a single file under `data/`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from foodscholar.io.chunk import EntityLink
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

_log = get_logger("foodscholar.annotate.cache")

# Bump when the on-disk schema changes so an old DB is detected, not misread.
SCHEMA_VERSION = 1


def cache_key(
    chunk_text: str,
    *,
    agent_model_id: str,
    prompt_version: str,
    ontology_hash: str,
) -> str:
    """Content-addressed key for one chunk's annotation.

    Combines everything an annotation depends on: the chunk text, the agent
    model, the prompt version, and a hash identifying the ontology. The four
    parts are length-prefixed before hashing so no concatenation collision is
    possible (e.g. `"ab" + "c"` vs `"a" + "bc"`).
    """
    h = hashlib.sha256()
    for part in (chunk_text, agent_model_id, prompt_version, ontology_hash):
        b = part.encode("utf-8")
        h.update(f"{len(b)}:".encode())
        h.update(b)
    return h.hexdigest()


class AnnotationCache:
    """SQLite-backed content-addressed store of per-chunk `list[EntityLink]`.

    Open with a path for a persistent cache, or `AnnotationCache(":memory:")`
    (the default) for an ephemeral one — handy in tests. The connection is
    held open for the object's lifetime; call `close()` or use it as a context
    manager.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                key            TEXT PRIMARY KEY,
                chunk_id       TEXT NOT NULL,
                links_json     TEXT NOT NULL,
                schema_version INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()

    # -- core operations ---------------------------------------------------

    def get(self, key: str) -> list[EntityLink] | None:
        """Return the cached links for `key`, or None on a miss."""
        row = self._conn.execute(
            "SELECT links_json, schema_version FROM annotations WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        links_json, schema_version = row
        if schema_version != SCHEMA_VERSION:
            # An entry from an older schema — treat as a miss so it is
            # recomputed and overwritten rather than mis-deserialized.
            _log.info("cache.schema_mismatch", key=key, found=schema_version)
            return None
        raw = json.loads(links_json)
        return [EntityLink.model_validate(item) for item in raw]

    def put(self, key: str, chunk_id: str, links: Iterable[EntityLink]) -> None:
        """Store (or replace) the links for `key`. Idempotent on the key."""
        links_json = json.dumps([link.model_dump(mode="json") for link in links])
        self._conn.execute(
            """
            INSERT INTO annotations (key, chunk_id, links_json, schema_version)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                chunk_id   = excluded.chunk_id,
                links_json = excluded.links_json,
                schema_version = excluded.schema_version
            """,
            (key, chunk_id, links_json, SCHEMA_VERSION),
        )
        self._conn.commit()

    def __contains__(self, key: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM annotations WHERE key = ?", (key,)
        ).fetchone()
        return row is not None

    def __len__(self) -> int:
        (n,) = self._conn.execute("SELECT COUNT(*) FROM annotations").fetchone()
        return int(n)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> AnnotationCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
