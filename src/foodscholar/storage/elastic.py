"""ElasticChunkStore — concrete ChunkStore backed by Elasticsearch 8.x.

Implementation deferred to the storage-backend milestone. The unit-test
foundation relies on InMemoryChunkStore.
"""

from __future__ import annotations


class ElasticChunkStore:
    def __init__(self, url: str, index: str) -> None:
        self.url = url
        self.index = index
        raise NotImplementedError(
            "ElasticChunkStore is not implemented yet — install the [elastic] extra "
            "and implement in the storage-backend milestone."
        )
