"""Neo4jGraphStore — concrete GraphStore backed by Neo4j 5.x.

Implementation deferred to the storage-backend milestone. The unit-test
foundation relies on InMemoryGraphStore.
"""

from __future__ import annotations


class Neo4jGraphStore:
    def __init__(self, url: str, user: str, password: str) -> None:
        self.url = url
        self.user = user
        self.password = password
        raise NotImplementedError(
            "Neo4jGraphStore is not implemented yet — install the [neo4j] extra "
            "and implement in the storage-backend milestone."
        )
