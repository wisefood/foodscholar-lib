"""Build `VizGraph`s from the foodscholar stores.

One function per visualization level. Each takes the facade (or the relevant
sub-stores) and returns a `VizGraph` the renderers consume.

  - L0  `entity_histogram`     — corpus-wide entity counts (no graph).
  - L1  `entity_neighborhood`  — one entity + its chunks + co-mentioned entities.
  - L2  `shelf_view`           — one shelf + its themes + its chunks (Layer A/B).
  - L3  `backbone`             — shelves + themes + cards (Layer A/B/C).
  - L4  `ontology_subtree`     — FoodOn ancestors / descendants of one term.

Builders are pure: they only read. They never call `init`, `upsert`, or
mutate the stores.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from foodscholar.viz.model import VizEdge, VizGraph, VizNode

if TYPE_CHECKING:
    from foodscholar.facade import FoodScholar
    from foodscholar.io.chunk import Chunk
    from foodscholar.io.entity import Entity
    from foodscholar.ontology import FoodOnAPI

# Per-shelf detail caps surfaced in the tree's tabbed panel (Terms / Entities / Sources).
TOP_TERMS = 100
TOP_ENTITIES = 40
TOP_SOURCES = 40
TOP_THEME_CHUNKS = 25

_WORD_RE = re.compile(r"[a-z][a-z\-]{2,}")
# Small English + nutrition-corpus stopword set so "Top terms" surface signal, not filler.
_STOPWORDS = frozenset(["the", "of", "and", "to", "in", "for", "is", "on", "with", "as", "by", "an", "at", "or", "be", "are", "from", "this", "that", "these", "those", "it", "its", "can", "may", "also", "such", "other", "more", "most", "than", "then", "they", "them", "their", "there", "which", "who", "whom", "whose", "into", "over", "under", "between", "within", "without", "about", "above", "below", "not", "no", "nor", "but", "if", "so", "we", "you", "your", "our", "his", "her", "she", "him", "he", "was", "were", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "should", "could", "one", "two", "each", "per", "any", "all", "some", "many", "few", "both", "either", "neither", "same", "very", "much", "like", "used", "use", "using", "uses", "based", "food", "foods", "diet", "dietary", "nutrition", "nutritional", "eat", "eating", "intake", "amount", "amounts", "source", "sources", "high", "low", "rich", "content", "contain", "contains", "containing", "include", "includes", "including", "example", "examples", "figure", "table", "chapter", "section", "page", "see", "e.g", "i.e", "etc"])


def _tokenize(text: str) -> Counter[str]:
    return Counter(w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOPWORDS)

# How many chunks / co-entities / descendants the default views surface
# inline. Renderers can't usefully show thousands of nodes, so the builder
# trims with `max_*` knobs and the truncation count lands in `attrs`.
DEFAULT_MAX_CHUNKS = 12
DEFAULT_MAX_CO_ENTITIES = 25
DEFAULT_MAX_DESCENDANTS = 30


# ---------------------------------------------------------------------- L0


def entity_histogram(
    fs: FoodScholar,
    *,
    prefix: str | None = None,
    k: int = 30,
) -> VizGraph:
    """Top-`k` entities by `chunk_count`. Disconnected — pure node stats.

    The renderers (esp. matplotlib) treat this as a bar chart over
    `node.weight = chunk_count`. The graph renderers degenerate to a
    scatter of unconnected dots — useful for at-a-glance prefix density.
    """
    entities = fs.entity_store.list_by_prefix(prefix, k=k) if prefix else \
        sorted(fs.entity_store.scan(), key=lambda e: e.chunk_count, reverse=True)[:k]

    nodes = [_entity_node(e) for e in entities]
    return VizGraph(
        title=f"Top {len(nodes)} entities" + (f" ({prefix})" if prefix else ""),
        nodes=nodes,
        edges=[],
        level="L0",
        attrs={"prefix": prefix, "k": k, "n_total": len(entities)},
    )


# ---------------------------------------------------------------------- L1


def entity_neighborhood(
    fs: FoodScholar,
    ontology_id: str,
    *,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    max_co_entities: int = DEFAULT_MAX_CO_ENTITIES,
) -> VizGraph:
    """Anchor entity + its mentioning chunks + co-mentioned entities.

    Edges:
      (chunk)-[mentions]->(entity)        for every chunk mention
      (chunk)-[mentions]->(co-entity)     for entities co-occurring in those chunks

    The chunk store carries the chunks; the entity store carries the entity
    metadata. Falls back to a single-node graph if the entity isn't found.
    """
    anchor = fs.entity_store.get(ontology_id)
    if anchor is None:
        return VizGraph(
            title=f"{ontology_id}: not found",
            nodes=[VizNode(id=ontology_id, label=ontology_id, kind="entity")],
            edges=[],
            level="L1",
            attrs={"missing": True, "ontology_id": ontology_id},
        )

    chunks = fs.entities.chunks_for(ontology_id, k=max_chunks)
    nodes: list[VizNode] = [_entity_node(anchor, anchor=True)]
    edges: list[VizEdge] = []
    seen_entity_ids = {anchor.ontology_id}
    co_counts: Counter[str] = Counter()

    for chunk in chunks:
        chunk_node = VizNode(
            id=chunk.chunk_id,
            label=_chunk_label(chunk.text),
            kind="chunk",
            attrs={
                "source_type": chunk.source_type,
                "section_type": chunk.section_type,
                "text_preview": chunk.text[:200],
                "foodon_ids": list(chunk.foodon_ids[:10]),
            },
        )
        nodes.append(chunk_node)
        edges.append(VizEdge(
            source=chunk.chunk_id, target=anchor.ontology_id,
            kind="mentions", weight=1.0,
        ))
        # Track co-mentioned entities (excluding the anchor).
        for link in chunk.entity_links:
            if link.ontology_id == anchor.ontology_id:
                continue
            co_counts[link.ontology_id] += 1

    # Top-N co-mentioned entities — pull their Entity records for label/prefix.
    top_co = [oid for oid, _ in co_counts.most_common(max_co_entities)]
    co_ents = fs.entity_store.get_many(top_co) if top_co else []
    co_by_id = {e.ontology_id: e for e in co_ents}
    for oid in top_co:
        ent = co_by_id.get(oid)
        if ent is None:
            # Co-entity exists in chunk.entity_links but not in entity store
            # — surfaced as a placeholder so the edge isn't dangling.
            ent_node = VizNode(id=oid, label=oid, kind="entity", attrs={"placeholder": True})
        else:
            ent_node = _entity_node(ent)
        if ent_node.id not in seen_entity_ids:
            nodes.append(ent_node)
            seen_entity_ids.add(ent_node.id)
        # Wire the chunk → co-entity edges (one per chunk that mentions both).
        for chunk in chunks:
            if any(ln.ontology_id == oid for ln in chunk.entity_links):
                edges.append(VizEdge(
                    source=chunk.chunk_id, target=oid,
                    kind="mentions", weight=float(co_counts[oid]),
                ))

    return VizGraph(
        title=f"Neighborhood of {anchor.label} ({anchor.ontology_id})",
        nodes=nodes,
        edges=edges,
        level="L1",
        attrs={
            "anchor": anchor.ontology_id,
            "n_chunks": len(chunks),
            "n_chunks_total": anchor.chunk_count,
            "n_co_entities": len(top_co),
        },
    )


# ---------------------------------------------------------------------- L2


def shelf_view(
    fs: FoodScholar,
    shelf_id: str,
    *,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> VizGraph:
    """One shelf + its themes + a sample of attached chunks.

    Stays empty if Layer A hasn't been built (no shelves yet) — the
    builder reports it via `attrs["empty_state"]` so renderers can show a
    helpful message.
    """
    shelf_handle = fs.graph.shelf(shelf_id)
    if shelf_handle is None:
        return _empty_layer_graph(
            "L2",
            title=f"Shelf {shelf_id}: not found",
            reason="Shelf does not exist. Run fs.build_layer_a() first.",
        )

    shelf = shelf_handle.model
    nodes: list[VizNode] = [_shelf_node(shelf)]
    edges: list[VizEdge] = []

    for theme_handle in shelf_handle.themes():
        nodes.append(_theme_node(theme_handle.model))
        edges.append(VizEdge(
            source=shelf.shelf_id, target=theme_handle.model.theme_id,
            kind="has_theme",
        ))

    chunks = shelf_handle.chunks()[:max_chunks]
    for chunk in chunks:
        nodes.append(VizNode(
            id=chunk.chunk_id,
            label=_chunk_label(chunk.text),
            kind="chunk",
            attrs={"source_type": chunk.source_type, "text_preview": chunk.text[:200]},
        ))
        edges.append(VizEdge(
            source=chunk.chunk_id, target=shelf.shelf_id,
            kind="attached_to",
        ))

    return VizGraph(
        title=f"Shelf: {shelf.label}",
        nodes=nodes,
        edges=edges,
        level="L2",
        attrs={"shelf_id": shelf.shelf_id, "facet": shelf.facet, "n_chunks": len(chunks)},
    )


# ---------------------------------------------------------------------- L3


def backbone(
    fs: FoodScholar,
    *,
    facet: str | None = None,
    include_cards: bool = True,
) -> VizGraph:
    """Whole shelf/theme/card backbone (Layer A + B + C).

    Empty graph with a helpful message if no shelves exist.
    """
    shelves = fs.graph.shelves(facet=facet) if facet else fs.graph.shelves()
    if not shelves:
        return _empty_layer_graph(
            "L3",
            title="Backbone: empty",
            reason="No shelves yet. Run fs.build_layer_a() to populate the backbone.",
        )

    nodes: list[VizNode] = []
    edges: list[VizEdge] = []

    for sh in shelves:
        nodes.append(_shelf_node(sh.model))
        if sh.model.parent_shelf_id:
            edges.append(VizEdge(
                source=sh.model.parent_shelf_id, target=sh.model.shelf_id,
                kind="parent_of",
            ))
        for theme_h in sh.themes():
            theme = theme_h.model
            # Theme may appear in multiple shelves — dedupe by id.
            if not any(n.id == theme.theme_id for n in nodes):
                nodes.append(_theme_node(theme))
            edges.append(VizEdge(
                source=sh.model.shelf_id, target=theme.theme_id, kind="has_theme",
            ))
            # Layer C makes per-theme cards — attach them (deduped by card id).
            if include_cards:
                tcard_h = theme_h.card()
                if tcard_h is not None and not _has_card(nodes, tcard_h.model.card_id):
                    nodes.append(_card_node(tcard_h.model))
                    edges.append(VizEdge(
                        source=tcard_h.model.card_id, target=theme.theme_id,
                        kind="describes",
                    ))
        if include_cards:
            card_h = sh.card()
            if card_h is not None and not _has_card(nodes, card_h.model.card_id):
                nodes.append(_card_node(card_h.model))
                edges.append(VizEdge(
                    source=card_h.model.card_id, target=sh.model.shelf_id,
                    kind="describes",
                ))

    return VizGraph(
        title="Backbone" + (f" ({facet})" if facet else ""),
        nodes=nodes,
        edges=edges,
        level="L3",
        attrs={"facet": facet, "n_shelves": len(shelves)},
    )


def layer_a_tree(fs: FoodScholar, facet: str | None = "foods") -> VizGraph:
    """Full Layer A shelf tree, with each shelf's Layer B themes grouped by
    `discovery_pass` (and each theme's Layer C card) in node attrs. Sub-threshold
    shelves (below `min_chunks_per_shelf`) are kept but flagged `eligible=False`
    and carry no themes. One `parent_of` edge per `parent_shelf_id`.

    `facet=None` walks EVERY facet's shelves into one tree (each facet's roots
    become top-level branches) — the full A→B→C graph across all facets.
    """
    min_chunks = fs.config.layer_b.min_chunks_per_shelf
    shelves = fs.graph.shelves(facet=facet) if facet else fs.graph.shelves()

    # One pass over the corpus: tokenize each chunk once, accumulate document
    # frequency (for tf-idf-style term scoring), and bucket chunks by shelf.
    all_chunks = fs.chunk_store.scan()
    n_docs = max(1, len(all_chunks))
    tokens_by_chunk: dict[str, Counter[str]] = {}
    doc_freq: Counter[str] = Counter()
    chunks_by_shelf: dict[str, list[Chunk]] = defaultdict(list)
    chunk_by_id: dict[str, Chunk] = {}
    for c in all_chunks:
        toks = _tokenize(c.text)
        tokens_by_chunk[c.chunk_id] = toks
        doc_freq.update(toks.keys())
        chunk_by_id[c.chunk_id] = c
        for sid in c.shelf_ids:
            chunks_by_shelf[sid].append(c)

    pass1_thr = fs.config.layer_b.similarity.edge_threshold

    try:
        _onto = fs.ontology
    except Exception:  # ontology not loaded (e.g. in_memory without FoodOn) — fall back to ids
        _onto = None

    def _label(ontology_id: str) -> str:
        if _onto is not None:
            try:
                return _onto.id_to_label(ontology_id) or ontology_id
            except Exception:
                return ontology_id
        return ontology_id

    def _shelf_terms(chunks: list[Chunk]) -> list[dict[str, Any]]:
        tf: Counter[str] = Counter()
        for c in chunks:
            tf.update(tokens_by_chunk.get(c.chunk_id, {}))
        # tf · smoothed-idf (sklearn-style, always positive) so ubiquitous words rank
        # below shelf-distinctive ones without ever zeroing out on small corpora.
        scored = sorted(
            tf.items(),
            key=lambda kv: kv[1] * (math.log((1 + n_docs) / (1 + doc_freq[kv[0]])) + 1),
            reverse=True,
        )
        return [{"term": w, "count": k} for w, k in scored[:TOP_TERMS]]

    def _shelf_entities(chunks: list[Chunk]) -> list[dict[str, Any]]:
        c: Counter[str] = Counter()
        for ch in chunks:
            c.update(ch.foodon_ids or [])
        return [
            {"id": i, "label": _label(i), "count": k}
            for i, k in c.most_common(TOP_ENTITIES)
        ]

    def _shelf_sources(chunks: list[Chunk]) -> list[dict[str, Any]]:
        agg: dict[str, dict[str, Any]] = {}
        for ch in chunks:
            row = agg.get(ch.source_doc_id)
            if row is None:
                agg[ch.source_doc_id] = row = {
                    "doc_id": ch.source_doc_id,
                    "source_type": str(ch.source_type),
                    "year": ch.year,
                    "count": 0,
                }
            row["count"] += 1
        return sorted(agg.values(), key=lambda r: -r["count"])[:TOP_SOURCES]

    def _theme_chunks(theme_id: str, discovery_pass: str) -> list[dict[str, Any]]:
        """Member chunks of a theme, each tagged direct/indirect by *which pass linked it*:
        similarity themes → all direct (text core); relatedness themes → all indirect (entity
        link); merged themes → per-chunk (direct iff it has a text neighbour ≥ pass-1 threshold)."""
        ids = list(fs.graph_store.get_chunks_for_theme(theme_id))
        vecs = {i: chunk_by_id[i].embedding for i in ids
                if i in chunk_by_id and chunk_by_id[i].embedding}

        def _badge(cid: str) -> str:
            if discovery_pass == "global_similarity":
                return "direct"
            if discovery_pass == "relatedness":
                return "indirect"
            v = vecs.get(cid)  # merged: reconstruct from text neighbours
            if v is None:
                return "indirect"
            a = np.asarray(v, dtype=float)
            na = float(np.linalg.norm(a)) or 1.0
            best = 0.0
            for other, w in vecs.items():
                if other == cid:
                    continue
                b = np.asarray(w, dtype=float)
                nb = float(np.linalg.norm(b)) or 1.0
                best = max(best, float(a @ b) / (na * nb))
            return "direct" if best >= pass1_thr else "indirect"

        rows = []
        for cid in ids[:TOP_THEME_CHUNKS]:
            ch = chunk_by_id.get(cid)
            text = (ch.text if ch else "")[:200]
            rows.append({
                "chunk_id": cid,
                "snippet": " ".join(text.split()),
                "source_doc_id": ch.source_doc_id if ch else None,
                "link": _badge(cid),
            })
        rows.sort(key=lambda r: r["link"] != "direct")  # direct first
        return rows

    nodes: list[VizNode] = []
    edges: list[VizEdge] = []
    n_eligible = 0
    n_themes = 0

    for sh in shelves:
        s = sh.model
        buckets: dict[str, list[dict[str, Any]]] = {
            "merged": [], "global_similarity": [], "relatedness": [],
        }
        for th in sh.themes():
            t = th.model
            bucket = buckets.get(t.discovery_pass)
            if bucket is None:  # unknown pass — skip defensively
                continue
            tcard_h = th.card()
            card = None
            if tcard_h is not None:
                cm = tcard_h.model
                card = {
                    "title": cm.title,
                    "summary": cm.summary,
                    "evidence_quality": cm.evidence_quality,
                    "tip": cm.tip,
                }
            bucket.append({
                "theme_id": t.theme_id,
                "label": t.label,
                "chunk_count": t.chunk_count,
                "keyword_terms": list(t.keyword_terms),
                "discovery_pass": t.discovery_pass,
                "chunks": _theme_chunks(t.theme_id, t.discovery_pass),
                "card": card,
            })
            n_themes += 1

        eligible = s.chunk_count >= min_chunks
        if eligible:
            n_eligible += 1

        shelf_chunks = chunks_by_shelf.get(s.shelf_id, [])
        nodes.append(VizNode(
            id=s.shelf_id,
            label=s.display_label or s.label,
            kind="shelf",
            weight=float(s.chunk_count),
            facet=s.facet,
            attrs={
                "chunk_count": s.chunk_count,
                "support_direct": s.support_direct,
                "support_lifted": s.support_lifted,
                "depth": s.depth,
                "foodon_id": s.foodon_id,
                "eligible": eligible,
                "themes": buckets,
                "terms": _shelf_terms(shelf_chunks),
                "entities": _shelf_entities(shelf_chunks),
                "sources": _shelf_sources(shelf_chunks),
            },
        ))
        if s.parent_shelf_id is not None:
            edges.append(VizEdge(
                source=s.parent_shelf_id, target=s.shelf_id, kind="parent_of",
            ))

    return VizGraph(
        title=f"Layer A tree — {facet or 'all facets'}",
        nodes=nodes,
        edges=edges,
        level="L3",
        attrs={
            "facet": facet or "all",
            "min_chunks_per_shelf": min_chunks,
            "n_shelves": len(nodes),
            "n_eligible": n_eligible,
            "n_themes": n_themes,
        },
    )


# ---------------------------------------------------------------------- L4


def ontology_subtree(
    ontology: FoodOnAPI,
    ontology_id: str,
    *,
    max_descendants: int = DEFAULT_MAX_DESCENDANTS,
    include_ancestors: bool = True,
) -> VizGraph:
    """Subtree of the loaded ontology centered on `ontology_id`.

    Up the tree: every ancestor (closed transitive set).
    Down the tree: up to `max_descendants` immediate / transitive descendants.
    """
    term = ontology.get(ontology_id) if hasattr(ontology, "get") else None
    if term is None:
        return VizGraph(
            title=f"{ontology_id}: not in ontology",
            nodes=[VizNode(id=ontology_id, label=ontology_id, kind="ontology_term")],
            edges=[],
            level="L4",
            attrs={"missing": True, "ontology_id": ontology_id},
        )

    nodes: list[VizNode] = [_ontology_node(ontology, term.id, anchor=True)]
    edges: list[VizEdge] = []
    seen: set[str] = {term.id}

    if include_ancestors:
        # Walk parents until roots; ancestor_ids is closed transitive but we
        # only need direct-parent edges for the tree to make sense visually.
        frontier = {term.id}
        while frontier:
            next_frontier: set[str] = set()
            for cid in frontier:
                for pid in ontology.id_to_parents(cid):
                    edges.append(VizEdge(source=cid, target=pid, kind="is_a"))
                    if pid not in seen:
                        nodes.append(_ontology_node(ontology, pid))
                        seen.add(pid)
                        next_frontier.add(pid)
            frontier = next_frontier

    descendants = ontology.id_to_descendants(term.id)[:max_descendants]
    for did in descendants:
        if did in seen:
            continue
        nodes.append(_ontology_node(ontology, did))
        seen.add(did)
        # Connect descendant to its closest known ancestor in the subtree
        # (parent if available; otherwise the anchor).
        parents = ontology.id_to_parents(did)
        target = next((p for p in parents if p in seen), term.id)
        edges.append(VizEdge(source=did, target=target, kind="is_a"))

    return VizGraph(
        title=f"Ontology subtree: {term.label} ({term.id})",
        nodes=nodes,
        edges=edges,
        level="L4",
        attrs={
            "anchor": term.id,
            "n_ancestors": len(nodes) - len(descendants) - 1,
            "n_descendants_shown": len(descendants),
            "n_descendants_total": len(ontology.id_to_descendants(term.id)),
        },
    )


# ---------------------------------------------------------------- helpers


def _entity_node(e: Entity, *, anchor: bool = False) -> VizNode:
    return VizNode(
        id=e.ontology_id,
        label=e.label,
        kind="entity",
        weight=float(e.chunk_count),
        facet=e.facet_hint,
        attrs={
            "prefix": e.prefix,
            "mention_count": e.mention_count,
            "chunk_count": e.chunk_count,
            "is_anchor": anchor,
            "synonyms": list(e.synonyms[:5]),
        },
    )


def _shelf_node(s: Any) -> VizNode:
    # Prefer the human-facing display_label (set on grouped shelves, e.g.
    # "Fruits" for the FoodOn node "plant fruit food product"); fall back to the
    # raw FoodOn label for top-down shelves that don't set one.
    display = getattr(s, "display_label", None) or s.label
    return VizNode(
        id=s.shelf_id,
        label=display,
        kind="shelf",
        weight=float(s.chunk_count),
        facet=s.facet,
        attrs={"depth": s.depth, "foodon_id": s.foodon_id, "ontology_label": s.label},
    )


def _theme_node(t: Any) -> VizNode:
    return VizNode(
        id=t.theme_id,
        label=t.label,
        kind="theme",
        weight=float(t.chunk_count),
        attrs={"discovered_by": t.discovered_by, "discovery_version": t.discovery_version},
    )


def _has_card(nodes: list[VizNode], card_id: str) -> bool:
    return any(n.id == card_id and n.kind == "card" for n in nodes)


def _card_node(c: Any) -> VizNode:
    return VizNode(
        id=c.card_id,
        label=c.title,
        kind="card",
        attrs={
            "target_id": c.target_id,
            "target_type": c.target_type,
            "evidence_quality": c.evidence_quality,
            "summary_preview": c.summary[:200],
        },
    )


def _ontology_node(ontology: FoodOnAPI, term_id: str, *, anchor: bool = False) -> VizNode:
    label = ontology.id_to_label(term_id) or term_id
    return VizNode(
        id=term_id,
        label=label,
        kind="ontology_term",
        attrs={"is_anchor": anchor, "ontology_id": term_id},
    )


def _chunk_label(text: str, *, max_len: int = 60) -> str:
    """Single-line chunk label for graph rendering."""
    snippet = " ".join(text.split())
    return snippet[:max_len] + ("…" if len(snippet) > max_len else "")


def _empty_layer_graph(level: str, *, title: str, reason: str) -> VizGraph:
    """Placeholder graph for Layer A/B views when those phases haven't run."""
    return VizGraph(
        title=title,
        nodes=[VizNode(id="empty", label=reason, kind="anchor")],
        edges=[],
        level=level,  # type: ignore[arg-type]
        attrs={"empty_state": True, "reason": reason},
    )
