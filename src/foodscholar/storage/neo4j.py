"""`Neo4jGraphStore` — `GraphStore` backed by Neo4j 5.x.

Graph model (Cypher labels):

  - (:Shelf {shelf_id, label, facet, depth, foodon_id})
  - (:Theme {theme_id, label, discovered_by, discovery_version})
  - (:Card  {card_id, target_id, target_type, title, summary, evidence_quality,
             llm_model, prompt_version, generated_at})
  - (:Chunk {chunk_id})  — stub nodes, full body lives in Elastic

Relationships:

  (:Shelf)-[:PARENT_OF]->(:Shelf)
  (:Theme)-[:IN_SHELF]->(:Shelf)
  (:Chunk)-[:ATTACHED_TO]->(:Shelf)
  (:Chunk)-[:ATTACHED_TO]->(:Theme)
  (:Card)-[:DESCRIBES_SHELF]->(:Shelf)
  (:Card)-[:DESCRIBES_THEME]->(:Theme)

`init()` creates the necessary unique constraints (idempotent — `IF NOT EXISTS`).
Credentials come from `cfg.storage.graph_store.{user, password}`; if the
password is unset, it falls back to `$NEO4J_PASSWORD`. URL defaults to
`bolt://localhost:7687` for the typical local-dev case.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

from foodscholar.io.chunk import ChunkId
from foodscholar.io.entity import Entity
from foodscholar.io.graph import Card, Shelf, ShelfId, Theme, ThemeId
from foodscholar.io.ontology import OntologyId
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    pass

_log = get_logger("foodscholar.storage.neo4j")

_DEFAULT_URL = "bolt://localhost:7687"
_DEFAULT_USER = "neo4j"


def _resolve_password(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("NEO4J_PASSWORD")
    if env:
        return env
    raise RuntimeError(
        "Neo4j password is not configured. Set cfg.storage.graph_store.password "
        "or the NEO4J_PASSWORD environment variable."
    )


class Neo4jGraphStore:
    """Neo4j 5.x backed implementation of the `GraphStore` protocol."""

    def __init__(
        self,
        url: str = "",
        user: str = "",
        password: str | None = None,
    ) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "the 'neo4j>=5' package is required for Neo4jGraphStore. "
                "Install with: pip install 'foodscholar[neo4j]'"
            ) from e
        self.url = url or _DEFAULT_URL
        self.user = user or _DEFAULT_USER
        self._driver = GraphDatabase.driver(
            self.url, auth=(self.user, _resolve_password(password))
        )

    def close(self) -> None:
        self._driver.close()

    # ------------------------------------------------------------------ admin

    def init(self) -> None:
        """Create unique constraints. Idempotent — safe to re-run."""
        statements = [
            "CREATE CONSTRAINT shelf_id IF NOT EXISTS FOR (s:Shelf) REQUIRE s.shelf_id IS UNIQUE",
            "CREATE CONSTRAINT theme_id IF NOT EXISTS FOR (t:Theme) REQUIRE t.theme_id IS UNIQUE",
            "CREATE CONSTRAINT card_id IF NOT EXISTS FOR (c:Card) REQUIRE c.card_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.ontology_id IS UNIQUE",
        ]
        with self._driver.session() as session:
            for stmt in statements:
                session.run(stmt)
        _log.info("neo4j.constraints_ready", url=self.url)

    # ------------------------------------------------------------------ shelves

    def clear_layer_a(self) -> None:
        """`DETACH DELETE` every (:Shelf) node — kills PARENT_OF / HAS_THEME /
        HAS_CHUNK / DESCRIBES edges attached to shelves in one shot. Themes and
        chunks (kept as separate node types) survive."""
        with self._driver.session() as session:
            session.run("MATCH (s:Shelf) DETACH DELETE s")
        _log.info("neo4j.layer_a_cleared", url=self.url)

    def clear_attachments(self) -> None:
        """Drop every `(:Chunk)-[r:ATTACHED_TO]->(:Shelf)` edge.

        Chunk and Shelf nodes survive; theme attachments (whose target is
        `(:Theme)`) are untouched. `fs.attach()` calls this so a re-run
        produces a clean projection without ghost edges from a previous
        config.
        """
        with self._driver.session() as session:
            session.run(
                "MATCH (:Chunk)-[r:ATTACHED_TO]->(:Shelf) DELETE r"
            )
        _log.info("neo4j.attachments_cleared", url=self.url)

    def upsert_shelves(self, shelves: list[Shelf]) -> None:
        if not shelves:
            return
        rows = [
            {
                "shelf_id": s.shelf_id,
                "label": s.label,
                "facet": s.facet,
                "depth": s.depth,
                "foodon_id": s.foodon_id,
                "parent_shelf_id": s.parent_shelf_id,
                "chunk_count": s.chunk_count,
                "support_direct": s.support_direct,
                "support_lifted": s.support_lifted,
                "see_also": list(s.see_also),
            }
            for s in shelves
        ]
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (s:Shelf {shelf_id: row.shelf_id})
                SET s.label = row.label,
                    s.facet = row.facet,
                    s.depth = row.depth,
                    s.foodon_id = row.foodon_id,
                    s.chunk_count = row.chunk_count,
                    s.support_direct = row.support_direct,
                    s.support_lifted = row.support_lifted,
                    s.see_also = row.see_also
                WITH s, row
                WHERE row.parent_shelf_id IS NOT NULL
                MERGE (p:Shelf {shelf_id: row.parent_shelf_id})
                MERGE (p)-[:PARENT_OF]->(s)
                """,
                rows=rows,
            )

    def get_shelf(self, shelf_id: ShelfId) -> Shelf | None:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s:Shelf {shelf_id: $shelf_id})
                OPTIONAL MATCH (p:Shelf)-[:PARENT_OF]->(s)
                RETURN s, p.shelf_id AS parent_shelf_id
                """,
                shelf_id=shelf_id,
            )
            record = result.single()
            if record is None or record["s"] is None:
                return None
            return _shelf_from_record(record["s"], record["parent_shelf_id"])

    def list_shelves(self) -> list[Shelf]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s:Shelf)
                OPTIONAL MATCH (p:Shelf)-[:PARENT_OF]->(s)
                RETURN s, p.shelf_id AS parent_shelf_id
                """,
            )
            return [_shelf_from_record(r["s"], r["parent_shelf_id"]) for r in result]

    def get_neighbors(self, shelf_id: ShelfId, hops: int = 1) -> list[ShelfId]:
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (s:Shelf {{shelf_id: $shelf_id}})
                MATCH (s)-[:PARENT_OF*1..{int(hops)}]-(n:Shelf)
                WHERE n.shelf_id <> $shelf_id
                RETURN DISTINCT n.shelf_id AS shelf_id
                """,
                shelf_id=shelf_id,
            )
            return [r["shelf_id"] for r in result]

    # ------------------------------------------------------------------ themes

    def upsert_themes(self, themes: list[Theme]) -> None:
        if not themes:
            return
        rows = [
            {
                "theme_id": t.theme_id,
                "label": t.label,
                "parent_theme_id": t.parent_theme_id,
                "shelf_ids": list(t.shelf_ids),
                "chunk_count": t.chunk_count,
                "discovered_by": t.discovered_by,
                "discovery_version": t.discovery_version,
                # Layer B extensions
                "facet": t.facet,
                "discovery_pass": t.discovery_pass,
                "keyword_terms": list(t.keyword_terms),
                "foodon_id_signature": list(t.foodon_id_signature),
                "config_hash": t.config_hash,
                "version": t.version,
            }
            for t in themes
        ]
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (t:Theme {theme_id: row.theme_id})
                SET t.label = row.label,
                    t.parent_theme_id = row.parent_theme_id,
                    t.chunk_count = row.chunk_count,
                    t.discovered_by = row.discovered_by,
                    t.discovery_version = row.discovery_version,
                    t.facet = row.facet,
                    t.discovery_pass = row.discovery_pass,
                    t.keyword_terms = row.keyword_terms,
                    t.foodon_id_signature = row.foodon_id_signature,
                    t.config_hash = row.config_hash,
                    t.version = row.version
                WITH t, row
                UNWIND row.shelf_ids AS sid
                MERGE (s:Shelf {shelf_id: sid})
                MERGE (t)-[:IN_SHELF]->(s)
                """,
                rows=rows,
            )

    def get_themes_for_shelf(self, shelf_id: ShelfId) -> list[Theme]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (t:Theme)-[:IN_SHELF]->(s:Shelf {shelf_id: $shelf_id})
                OPTIONAL MATCH (t)-[:IN_SHELF]->(s2:Shelf)
                WITH t, collect(DISTINCT s2.shelf_id) AS shelf_ids
                RETURN t, shelf_ids
                """,
                shelf_id=shelf_id,
            )
            return [_theme_from_record(r["t"], r["shelf_ids"]) for r in result]

    def list_themes(self) -> list[Theme]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (t:Theme)
                OPTIONAL MATCH (t)-[:IN_SHELF]->(s:Shelf)
                WITH t, collect(DISTINCT s.shelf_id) AS shelf_ids
                RETURN t, shelf_ids
                """,
            )
            return [_theme_from_record(r["t"], r["shelf_ids"]) for r in result]

    def list_chunk_shelf_attachments(self) -> dict[ChunkId, set[str]]:
        out: dict[ChunkId, set[str]] = defaultdict(set)
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)-[:ATTACHED_TO]->(s:Shelf)
                RETURN c.chunk_id AS chunk_id, s.shelf_id AS shelf_id
                """,
            )
            for r in result:
                out[r["chunk_id"]].add(r["shelf_id"])
        return dict(out)

    def list_chunk_foodon_mentions(self) -> dict[ChunkId, set[str]]:
        out: dict[ChunkId, set[str]] = defaultdict(set)
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE e.ontology_id STARTS WITH 'FOODON:'
                RETURN c.chunk_id AS chunk_id, e.ontology_id AS ontology_id
                """,
            )
            for r in result:
                out[r["chunk_id"]].add(r["ontology_id"])
        return dict(out)

    def get_chunks_for_theme(self, theme_id: ThemeId) -> list[ChunkId]:
        # Reads either edge label so legacy `attach_chunks_to_theme` writes
        # (`ATTACHED_TO`) and new Layer B `attach_chunks_to_themes_bulk`
        # writes (`THEME_OF` with `primary` + `weight` properties) both
        # surface here. New code should use the bulk method.
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)-[r:ATTACHED_TO|THEME_OF]->(t:Theme {theme_id: $theme_id})
                RETURN DISTINCT c.chunk_id AS chunk_id
                """,
                theme_id=theme_id,
            )
            return [r["chunk_id"] for r in result]

    # ------------------------------------------------------------------ cards

    def upsert_cards(self, cards: list[Card]) -> None:
        if not cards:
            return
        rows = [
            {
                "card_id": c.card_id,
                "target_id": c.target_id,
                "target_type": c.target_type,
                "title": c.title,
                "summary": c.summary,
                "tip": c.tip,
                "evidence_quality": c.evidence_quality,
                "controversy_note": c.controversy_note,
                "confidence_note": c.confidence_note,
                "cited_chunk_ids": list(c.cited_chunk_ids),
                "llm_model": c.llm_model,
                "prompt_version": c.prompt_version,
                "safety_flagged": c.safety_flagged,
                "generated_at": c.generated_at.isoformat(),
            }
            for c in cards
        ]
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (c:Card {card_id: row.card_id})
                SET c.target_id = row.target_id,
                    c.target_type = row.target_type,
                    c.title = row.title,
                    c.summary = row.summary,
                    c.tip = row.tip,
                    c.evidence_quality = row.evidence_quality,
                    c.controversy_note = row.controversy_note,
                    c.confidence_note = row.confidence_note,
                    c.cited_chunk_ids = row.cited_chunk_ids,
                    c.llm_model = row.llm_model,
                    c.prompt_version = row.prompt_version,
                    c.safety_flagged = row.safety_flagged,
                    c.generated_at = row.generated_at
                WITH c, row
                CALL {
                    WITH c, row
                    WITH c, row WHERE row.target_type = 'shelf'
                    MERGE (s:Shelf {shelf_id: row.target_id})
                    MERGE (c)-[:DESCRIBES_SHELF]->(s)
                    RETURN c AS skip
                }
                WITH c, row
                CALL {
                    WITH c, row
                    WITH c, row WHERE row.target_type = 'theme'
                    MERGE (t:Theme {theme_id: row.target_id})
                    MERGE (c)-[:DESCRIBES_THEME]->(t)
                    RETURN c AS skip
                }
                RETURN count(*) AS _
                """,
                rows=rows,
            )

    def get_card(
        self, target_id: str, target_type: Literal["shelf", "theme"]
    ) -> Card | None:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Card {target_id: $target_id, target_type: $target_type})
                RETURN c
                """,
                target_id=target_id,
                target_type=target_type,
            )
            record = result.single()
            if record is None:
                return None
            return _card_from_record(record["c"])

    # ------------------------------------------------------------------ attachments

    def attach_chunks_to_shelf(
        self,
        shelf_id: ShelfId,
        attachments: list[tuple[ChunkId, list[str]]],
    ) -> None:
        """Wire `(:Chunk)-[r:ATTACHED_TO]->(:Shelf)` edges with `lifted_from`.

        Dedupes attachments by chunk_id before sending — Neo4j's MERGE on
        relationships would collapse duplicates anyway, but having the same
        chunk_id twice in `$rows` makes our row count diverge from the
        actual edge count and breaks any "rows sent == edges created"
        invariant the caller depends on. Last-write-wins on `lifted_from`
        if a chunk_id appears twice (shouldn't happen in correct usage but
        is harmless when it does).
        """
        if not attachments:
            return
        # Dedupe by chunk_id, preserving the last seen lifted_from. dict
        # ordering preserves insertion order so the iteration is deterministic.
        deduped: dict[ChunkId, list[str]] = {}
        for cid, lifted_from in attachments:
            deduped[cid] = list(lifted_from)
        rows = [
            {"chunk_id": cid, "lifted_from": lf}
            for cid, lf in deduped.items()
        ]
        if not rows:
            return
        # `execute_write` uses a managed transaction with automatic retry on
        # transient errors (deadlocks, leader elections). `session.run` in
        # auto-commit mode silently swallows TransientError retries on some
        # driver/server combinations — leading to lost writes that surface as
        # missing edges after a parallel attach. The managed-tx path is the
        # documented correct pattern for concurrent MERGE on unique-constraint
        # nodes (chunk_id), and it's the same cost as the implicit form.
        cypher = (
            "MATCH (s:Shelf {shelf_id: $shelf_id}) "
            "UNWIND $rows AS row "
            "MERGE (c:Chunk {chunk_id: row.chunk_id}) "
            "MERGE (c)-[r:ATTACHED_TO]->(s) "
            "SET r.lifted_from = row.lifted_from"
        )
        with self._driver.session() as session:
            session.execute_write(
                lambda tx: tx.run(cypher, shelf_id=shelf_id, rows=rows).consume()
            )

    def attach_chunks_to_theme(self, theme_id: ThemeId, chunk_ids: list[ChunkId]) -> None:
        if not chunk_ids:
            return
        with self._driver.session() as session:
            session.run(
                """
                MATCH (t:Theme {theme_id: $theme_id})
                UNWIND $chunk_ids AS cid
                MERGE (c:Chunk {chunk_id: cid})
                MERGE (c)-[:ATTACHED_TO]->(t)
                """,
                theme_id=theme_id,
                chunk_ids=chunk_ids,
            )

    def attach_chunks_to_themes_bulk(
        self,
        items: list[tuple[ChunkId, ThemeId, bool, float]],
    ) -> None:
        """Bulk write `(:Chunk)-[:THEME_OF {primary, weight}]->(:Theme)` edges.

        Layer B's persist path. One UNWIND-driven session.run per call covers
        the whole batch — single network round-trip on Neo4j.
        """
        if not items:
            return
        rows = [
            {
                "chunk_id": chunk_id,
                "theme_id": theme_id,
                "primary": bool(primary),
                "weight": float(weight),
            }
            for chunk_id, theme_id, primary, weight in items
        ]
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (c:Chunk {chunk_id: row.chunk_id})
                MERGE (t:Theme {theme_id: row.theme_id})
                MERGE (c)-[r:THEME_OF]->(t)
                SET r.primary = row.primary,
                    r.weight = row.weight
                """,
                rows=rows,
            )

    def clear_themes(self) -> None:
        """`DETACH DELETE` every `(:Theme)` node — kills HAS_THEME (from
        shelves), THEME_OF / ATTACHED_TO (from chunks), and DESCRIBES (from
        theme-target cards) in one shot.

        `build_layer_b()` calls this at the start so re-runs with a different
        config don't leave ghost themes. Shelves and chunks survive. Cards
        with `target_type='shelf'` survive (they live on shelves, not themes).
        """
        with self._driver.session() as session:
            session.run("MATCH (t:Theme) DETACH DELETE t")
        _log.info("neo4j.themes_cleared", url=self.url)

    # ------------------------------------------------------------------ entities

    def upsert_entities(self, entities: list[Entity]) -> None:
        if not entities:
            return
        rows = [
            {
                "ontology_id": e.ontology_id,
                "prefix": e.prefix,
                "label": e.label,
                "synonyms": list(e.synonyms),
                "ancestor_ids": list(e.ancestor_ids),
                "facet_hint": e.facet_hint,
                "mention_count": e.mention_count,
                "chunk_count": e.chunk_count,
                "last_seen": e.last_seen.isoformat(),
            }
            for e in entities
        ]
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {ontology_id: row.ontology_id})
                SET e.prefix = row.prefix,
                    e.label = row.label,
                    e.synonyms = row.synonyms,
                    e.ancestor_ids = row.ancestor_ids,
                    e.facet_hint = row.facet_hint,
                    e.mention_count = row.mention_count,
                    e.chunk_count = row.chunk_count,
                    e.last_seen = row.last_seen
                """,
                rows=rows,
            )

    def attach_chunks_to_entity(
        self,
        ontology_id: OntologyId,
        chunk_links: list[tuple[ChunkId, float, str]],
    ) -> None:
        if not chunk_links:
            return
        rows = [
            {"chunk_id": cid, "confidence": float(conf), "method": str(method)}
            for cid, conf, method in chunk_links
        ]
        with self._driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {ontology_id: $ontology_id})
                WITH e
                UNWIND $rows AS row
                MERGE (c:Chunk {chunk_id: row.chunk_id})
                MERGE (c)-[r:MENTIONS]->(e)
                SET r.confidence = row.confidence,
                    r.method = row.method
                """,
                ontology_id=ontology_id,
                rows=rows,
            )


# ---------------------------------------------------------------------- helpers


def _shelf_from_record(node: Any, parent_shelf_id: str | None) -> Shelf:
    return Shelf(
        shelf_id=node["shelf_id"],
        label=node["label"],
        facet=node["facet"],
        depth=int(node["depth"]),
        foodon_id=node.get("foodon_id"),
        parent_shelf_id=parent_shelf_id,
        chunk_count=int(node.get("chunk_count") or 0),
        support_direct=int(node.get("support_direct") or 0),
        support_lifted=int(node.get("support_lifted") or 0),
        see_also=list(node.get("see_also") or []),
    )


def _theme_from_record(node: Any, shelf_ids: list[str]) -> Theme:
    return Theme(
        theme_id=node["theme_id"],
        label=node["label"],
        parent_theme_id=node.get("parent_theme_id"),
        shelf_ids=shelf_ids,
        chunk_count=int(node.get("chunk_count") or 0),
        discovered_by=node["discovered_by"],
        discovery_version=node["discovery_version"],
        # Layer B extensions — old theme nodes may not carry these yet.
        facet=node.get("facet") or "foods",
        discovery_pass=node.get("discovery_pass") or "similarity",
        keyword_terms=list(node.get("keyword_terms") or []),
        foodon_id_signature=list(node.get("foodon_id_signature") or []),
        config_hash=node.get("config_hash") or "",
        version=node.get("version") or "",
    )


def _card_from_record(node: Any) -> Card:
    from datetime import datetime

    return Card(
        card_id=node["card_id"],
        target_id=node["target_id"],
        target_type=node["target_type"],
        title=node["title"],
        summary=node["summary"],
        tip=node.get("tip"),
        evidence_quality=node["evidence_quality"],
        controversy_note=node.get("controversy_note"),
        confidence_note=node.get("confidence_note"),
        cited_chunk_ids=list(node.get("cited_chunk_ids") or []),
        llm_model=node["llm_model"],
        prompt_version=node["prompt_version"],
        safety_flagged=bool(node.get("safety_flagged")),
        generated_at=datetime.fromisoformat(node["generated_at"]),
    )
