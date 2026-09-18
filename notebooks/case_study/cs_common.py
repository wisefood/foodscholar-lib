"""Shared helpers for the FoodScholar qualitative case study (NB1–NB4).

Single source of truth for: pre-registered config (anchors, queries, rubric
knobs), facade construction, corpus loading, anchor resolution, graph
persistence across kernels, environment capture, and figure styling.

Every notebook starts with::

    import cs_common as cs
    fs = cs.get_fs(with_llm=False)            # or with_llm=True for NB4
    env = cs.capture_env(fs)
    cs.save_json(cs.artifacts_dir("nb1_backbone") / "env.json", env)

ADAPTATION NOTES (verified against foodscholar-lib @ this commit — see
RUN.md for the deviation log):

* The in-memory facade ships a deterministic *mock* embedder (dim-8 hash).
  The case-study corpus (`data/annotated.parquet`) already carries **real
  BGE-base** chunk embeddings, so we DO NOT call ``fs.embed()`` — re-embedding
  would overwrite the real vectors with mock ones. Layer A/B/C read the
  existing vectors as-is.
* There is no graphviz ``dot`` binary in this environment, so PNG figures are
  produced with matplotlib (which we control); the interactive shelf/theme
  trees stay HTML-only via the pyvis/"tree" backends.
* The in-memory graph store is RAM-only. NB1 persists the built graph to JSON
  (shelves/themes/cards) + the chunk store to parquet so NB2–NB4 reload it in
  fresh kernels without rebuilding (`save_graph` / `load_graph`).
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import platform
import random
from pathlib import Path
from typing import Any

SEED = 42
random.seed(SEED)

# notebooks/case_study/ -> repo root is two levels up.
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
ART = ROOT / "artifacts"

# Abstract-expanded, already-annotated corpus: 34,359 chunks via the NEL/ingest
# path (textbook 12,194 + guide 1,150 + abstract 21,015) + the cached FoodOn
# term table. NOTE: the annotated parquet carries NO chunk vectors (the ingest
# path doesn't embed; the `embedding` column is 100% null); real BGE-base
# vectors live in a separate npy cache, loaded by `load_real_embeddings` so kNN
# retrieval (NB3) is genuinely semantic. Built by `build_corpus.py`.
ANNOTATED_PARQUET = REPO_ROOT / "data" / "annotated_with_abstracts.parquet"
FOODON_OWL = REPO_ROOT / "data" / "foodon.owl"
FOODON_CACHE = REPO_ROOT / "data" / "foodon_cache.parquet"

# Cached real BGE-base-en-v1.5 chunk vectors (14487 × 768), keyed positionally
# to BGE_IDS_JSON. Composition: 13,252 textbook/guide base vectors + anchor-
# subtree vectors embedded on demand (218 olive/legume + 297 fish + 720 dietary
# fibre = 1,235). Chunks outside the anchor subtrees stay BM25-only at search
# time. `load_real_embeddings` guards the npy/ids alignment (length + dedup).
BGE_EMB_NPY = REPO_ROOT / "data" / "cache" / "bge_base_emb_14487.npy"
BGE_IDS_JSON = REPO_ROOT / "data" / "cache" / "bge_base_ids_14487.json"
BGE_MODEL_ID = "BAAI/bge-base-en-v1.5"

# Shared graph snapshot written by NB1, reloaded by NB2–NB4.
GRAPH_SNAPSHOT = ART / "shared" / "layer_a_graph.json"
CHUNK_SNAPSHOT = ART / "shared" / "layer_a_chunks.parquet"

# Brand palette (matches the deliverable). The first five are the deck accent
# colours; the rest are the neutral ink / surface tones the slide toolkit uses.
COLORS = {
    "hierarchy": "#4E7D33",  # hierarchical-graph green  (Layer A / backbone)
    "teal": "#2E6E8E",       # Leiden / flat
    "amber": "#AD7A23",      # raw FoodOn / delta highlight
    "purple": "#5E47A6",     # BERTopic / graph-expanded
    "grey": "#8A8F98",
    # neutrals for the slide frame
    "ink": "#1F2933",        # primary text
    "muted": "#5A6472",      # secondary text
    "surface": "#FFFFFF",    # figure background
    "panel": "#F4F6F8",      # light panel fill
    "rule": "#D9DEE4",       # hairline rules
}

# Light tints of each accent (for bar fills / panel backgrounds).
TINTS = {
    "hierarchy": "#DCE7D2",
    "teal": "#D5E3EB",
    "amber": "#EEE0C7",
    "purple": "#E0DAF0",
}

# ---- Pre-registered configuration (the ONLY place to edit anchors/queries) ----
#
# DEVIATION from the build spec (recorded in prereg.md + every summary.md):
# the spec's 2nd anchor was "dietary fibre" in the `nutrients` facet. In the
# real build Layer A is FoodOn-only (prefix_filter=["FOODON:"]), so `nutrients`
# is a degenerate flat stub (one root shelf, no hierarchy, no themes). We
# therefore re-anchor P2 onto the canonical high-fibre FOOD subtree —
# `legume food product` (FOODON:00001264, 760 chunks, depth 3, real bean/
# lentil/soybean descendants) — keeping both anchors in the one facet Layer A
# actually populates. The fibre / blood-sugar / cholesterol query is unchanged
# in intent; legumes are its prototypical food carrier.
#
CONFIG: dict[str, Any] = {
    "anchors": {
        "olive_oil": {
            "paradigm": "P1-food",
            "facet": "foods",
            "labels": ["olive oil", "olive oil (raw)", "olive oil food product"],
            "fallback_labels": ["vegetable oil", "plant fat or oil refined food product"],
        },
        "legume": {
            "paradigm": "P2-food-as-fibre-source",
            "facet": "foods",
            "labels": ["legume food product", "legume", "bean food product"],
            "fallback_labels": ["cereal grain food product", "plant seed food product"],
        },
        # P3 — a primarily abstract-driven anchor (added for the §6.1.3
        # deliverable: shows journal abstracts driving a Layer C card
        # end-to-end). `fish food product` is a canonical nutrition-science
        # topic (omega-3 / cardiovascular) with rich abstract coverage
        # (789-chunk subtree, ~40% abstracts).
        "fish": {
            "paradigm": "P3-abstract-driven",
            "facet": "foods",
            "labels": ["fish food product", "fish (whole)", "fish"],
            "fallback_labels": ["vertebrate food product", "animal food product"],
        },
        # P2-nutrient — a NON-FOOD anchor in the `nutrients` facet, so the case
        # study isn't foods-only. Verified read-only: dietary fibre =
        # CDNO:0000005, depth 2, 1,512 chunks (1,214 direct / 298 lifted),
        # status=active, clears min_support=20 by ~75x, with a real subtree
        # (soluble/insoluble/prebiotic/fermentable/viscous). Only available
        # because the multi-facet build (prefix_filter=None) populates nutrients
        # from CDNO/CHEBI. NOTE: CDNO is among the better-covered non-FOODON
        # ontologies, so this anchor is on the STRONG end of the (noisier/sparser)
        # nutrients facet — watch theme labels for facet-routing mislink noise.
        "dietary_fibre": {
            "paradigm": "P2-nutrient",
            "facet": "nutrients",
            "labels": ["dietary fibre", "dietary fiber", "fibre", "fiber"],
            "fallback_labels": ["carbohydrate", "polysaccharide"],
        },
    },
    "queries": {
        "olive_oil": "Is olive oil good for cardiovascular health?",
        "legume": "How do legumes and dietary fibre affect blood sugar and cholesterol?",
        "fish": "What are the cardiovascular and omega-3 benefits of eating fish?",
        "dietary_fibre": "How does dietary fibre affect blood sugar and cholesterol?",
    },
    "retrieval_k": 10,
    "samples_per_theme": 3,
    "seed": SEED,
}


def artifacts_dir(nb: str) -> Path:
    d = ART / nb
    d.mkdir(parents=True, exist_ok=True)
    return d


def shared_dir() -> Path:
    d = ART / "shared"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def capture_env(fs: Any = None, *, extra: dict | None = None) -> dict:
    import importlib.metadata as md

    def ver(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except Exception:
            return None

    env = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python": platform.python_version(),
        "seed": SEED,
        "packages": {
            p: ver(p)
            for p in [
                "foodscholar",
                "bertopic",
                "umap-learn",
                "hdbscan",
                "scikit-learn",
                "sumy",
                "neo4j",
                "elasticsearch",
                "groq",
            ]
        },
        "config_hash": hashlib.sha256(
            json.dumps(CONFIG, sort_keys=True).encode()
        ).hexdigest()[:12],
    }
    if fs is not None:
        env["foodscholar_config_hash"] = getattr(fs, "config_hash", None)
        env["chunk_store"] = fs.config.storage.chunk_store.backend
        env["graph_store"] = fs.config.storage.graph_store.backend
        env["llm_model"] = fs.llm.model_id
        env["llm_is_mock"] = is_mock_llm(fs)
    if extra:
        env.update(extra)
    return env


# ------------------------------------------------------------------ facade build

def _case_study_config(*, with_llm: bool) -> dict:
    """In-memory config pointing at the real annotated corpus + cached FoodOn.

    The Groq ``llm`` section is included ONLY when ``with_llm=True``:
    ``FoodScholar.from_config`` eagerly builds the LLM client whenever
    ``cfg.llm`` is set, and the Groq client raises at construction if
    GROQ_API_KEY is absent. NB1/NB3 don't need an LLM, so we leave the section
    out and the facade falls back to the harmless offline mock for them.
    """
    cfg = {
        "corpus": {"chunks_path": str(ANNOTATED_PARQUET)},
        "ontology": {
            "foodon_path": str(FOODON_OWL),
            "cache_path": str(FOODON_CACHE),
            "include_imports": False,
            # prefix_filter=None admits ALL OBO terms in the cache (not just
            # FOODON:), so the non-food facets populate from their ontologies via
            # PREFIX_TO_FACET (CHEBI/CDNO→nutrients, UBERON/PATO→health,
            # ENVO→sustainability). foods/anchors are unchanged. dietary_patterns
            # and allergies stay stubs (no entity_type/prefix routes to them, and
            # the prototype NER leaves entity_type='other'). KNOWN LIMITATION: the
            # non-FOODON cache coverage is thin (e.g. 203 UBERON terms vs ~12.6k
            # UBERON links) and prefix routing inherits NEL mislink noise
            # (heart-disease→UBERON etc.) — see layer_a/facet.py route_link_to_facet.
            "prefix_filter": None,
        },
        "layer_a": {
            "min_support": 20,
            "max_depth": 5,
            "collapse_single_child_chains": True,
            "blacklist_terms": [
                "material entity",
                "physical object",
                "manufactured product",
            ],
            "facets": [
                "foods",
                "health",
                "sustainability",
                "dietary_patterns",
                "allergies",
                "nutrients",
            ],
        },
        "layer_c": {
            "llm_model": "llama-3.3-70b-versatile",
            "prompt_version": "v1",
            "sample_size": 12,
        },
        "storage": {
            "chunk_store": {"backend": "memory"},
            "graph_store": {"backend": "memory"},
            "card_store": {"backend": "memory"},
        },
    }
    if with_llm:
        cfg["llm"] = {
            "primary": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
            "timeout_s": 30,
            "max_retries": 2,
        }
    return cfg


def is_mock_llm(fs: Any) -> bool:
    """True iff the facade's LLM is the built-in offline mock."""
    return getattr(fs.llm, "model_id", "") == "mock-llm-v0"


def get_fs(*, with_llm: bool):
    """Construct the FoodScholar facade against the case-study corpus.

    - ``with_llm=False`` (NB1/NB3): the facade still carries a config LLM
      section, but no LLM call is made, so a missing GROQ_API_KEY is fine.
    - ``with_llm=True`` (NB2 LLM labels / NB4 cards): a REAL Groq client is
      required. We build it eagerly and raise if it would fall back to the
      mock, so the notebook fails loudly rather than fabricating output.
    """
    from foodscholar import FoodScholar

    cfg = _case_study_config(with_llm=with_llm)
    if with_llm:
        fs = FoodScholar.from_config(cfg)  # auto-builds Groq; raises if no key
        if is_mock_llm(fs):
            raise RuntimeError(
                "with_llm=True but the facade resolved to the mock LLM. "
                "Set GROQ_API_KEY and retry — NB4 must never use 'Mock answer' output."
            )
    else:
        fs = FoodScholar.from_config(cfg)
    return fs


def load_corpus(fs: Any) -> int:
    """Load the prebuilt annotated chunks. Returns the chunk count.

    The annotated parquet carries NO vectors; call `load_real_embeddings(fs)`
    afterwards to attach the cached BGE-base vectors before any kNN retrieval.
    """
    return fs.load_chunks(str(ANNOTATED_PARQUET))


def load_real_embeddings(fs: Any) -> dict[str, int]:
    """Attach cached real BGE-base vectors to chunks already in the store.

    Reads the (13252 × 768) npy cache + its chunk-id index and writes each
    vector back via `chunk_store.update_embeddings_bulk`, tagging
    `embedding_model = BAAI/bge-base-en-v1.5` so the facade treats them as
    real. Chunks with no cached vector are left unembedded (BM25-only at
    search time). Returns coverage counts.

    This is the honest substitute for `fs.embed()`: it avoids loading the
    ~440 MB model while still giving kNN genuine semantic vectors, instead of
    the dim-8 mock hash the in-memory facade would otherwise use.
    """
    import json as _json

    import numpy as np

    if not BGE_EMB_NPY.exists() or not BGE_IDS_JSON.exists():
        raise FileNotFoundError(
            f"BGE cache missing ({BGE_EMB_NPY} / {BGE_IDS_JSON}); "
            "NB3 kNN retrieval needs real vectors."
        )
    emb = np.load(BGE_EMB_NPY)
    ids = _json.loads(BGE_IDS_JSON.read_text())
    # Alignment guards — the cache attaches vectors POSITIONALLY (emb[i] for
    # ids[i]). If the .npy and .json ever desync (truncation, an append that
    # didn't dedup, a wrong-corpus cache) the wrong vector would be attached
    # under the real-model tag with NO error → silently poisoned kNN. Fail loud.
    if len(ids) != emb.shape[0]:
        raise ValueError(
            f"BGE cache misaligned: {len(ids)} ids vs {emb.shape[0]} vectors "
            f"({BGE_IDS_JSON.name} / {BGE_EMB_NPY.name})"
        )
    if len(set(ids)) != len(ids):
        raise ValueError(
            "BGE cache has duplicate chunk ids (a regeneration appended without "
            "dedup) — attachment by position is unsafe; rebuild the cache."
        )
    present = {c.chunk_id for c in fs.chunk_store.scan()}
    missing = set(ids) - present
    if missing:
        raise ValueError(
            f"BGE cache drift: {len(missing)} cached ids are absent from the "
            f"corpus (e.g. {sorted(missing)[:3]}) — cache built for a different "
            "corpus than the one loaded."
        )
    updates = [
        (cid, emb[i].tolist(), BGE_MODEL_ID)
        for i, cid in enumerate(ids)
        if cid in present
    ]
    fs.chunk_store.update_embeddings_bulk(updates)
    return {
        "cache_vectors": len(ids),
        "chunks_in_store": len(present),
        "attached": len(updates),
        "dim": int(emb.shape[1]),
    }


def bge_query_vector(text: str) -> list[float]:
    """Embed a query in the SAME BGE-base space as the cached chunk vectors.

    Loads the real HF embedder once (cached on the module). Falls back with a
    clear error if the `[annotate]` extra / model weights are unavailable —
    kNN against BGE chunk vectors is meaningless with a mismatched embedder.
    """
    global _BGE_EMBEDDER
    if _BGE_EMBEDDER is None:
        from foodscholar.annotate.embedder import HFEmbedder

        _BGE_EMBEDDER = HFEmbedder(BGE_MODEL_ID)
    return _BGE_EMBEDDER.embed([text])[0]


_BGE_EMBEDDER: Any = None


def real_embedded_fraction(fs: Any) -> float:
    """Fraction of ALL chunks carrying a real (non-mock) embedding.

    Counts the whole store (not a head-slice): the corpus is ordered
    textbook→guide→abstract, so a `scan()[:N]` head-sample would only see the
    already-embedded textbook chunks and report ~1.0 even when the abstracts
    (loaded last) have no vectors. This must reflect the true coverage so the
    NB3 embed decision is made on real numbers.
    """
    from foodscholar.facade import _is_real_embedding

    n = real = 0
    for chunk in fs.chunk_store.scan():
        n += 1
        if _is_real_embedding(chunk.embedding_model):
            real += 1
    return real / n if n else 0.0


# ------------------------------------------------------------------ anchors

def subtree_chunk_ids(fs: Any, shelf, *, facet: str = "foods") -> list[str]:
    """Builder-faithful subtree chunk set for a shelf.

    Reconstructs exactly what `build_layer_b` themes under ``scope="subtree"``:
    the union of the shelf's own attachment-edge chunks with every descendant
    shelf's, read from ``graph_store.list_chunk_shelf_attachments`` (the edge
    table the builder iterates) — NOT the chunk-side ``shelf_ids`` denorm
    (which over-counts via lifted multi-shelf chunks; that's why a shelf's
    ``chunk_count`` can exceed this set). Same primitive both clusterers see,
    so an A/B comparison over this set is apples-to-apples with the real build.
    """
    att = fs.graph_store.list_chunk_shelf_attachments()
    shelf_to_chunks: dict[str, list[str]] = {}
    for cid, sids in att.items():
        for sid in sids:
            shelf_to_chunks.setdefault(sid, []).append(cid)
    children: dict[str, list[str]] = {}
    for s in fs.graph.shelves(facet=facet):
        if s.parent_shelf_id is not None:
            children.setdefault(s.parent_shelf_id, []).append(s.shelf_id)
    seen: set[str] = set()
    stack = [shelf.shelf_id]
    while stack:
        sid = stack.pop()
        seen.update(shelf_to_chunks.get(sid, ()))
        stack.extend(children.get(sid, ()))
    return sorted(seen)


def resolve_shelf(fs: Any, facet: str, labels: list[str], fallback_labels: list[str]):
    """Find an anchor shelf by case-insensitive label match; else fallback; else raise.

    Returns ``(ShelfHandle, used_fallback: bool, matched_label: str)``. Matching
    prefers an exact (case-insensitive) hit on ``label``/``display_label`` over a
    substring hit, and among substring hits prefers the shelf with the most
    attached chunks (the salient one for the case study).
    """
    shelves = list(fs.graph.shelves(facet=facet))

    def _find(cands: list[str]):
        cl = [c.lower() for c in cands]
        exact = []
        substr = []
        for s in shelves:
            lab = (s.label or "").lower()
            dl = ((s.model.display_label or "") if s.model.display_label else "").lower()
            if lab in cl or dl in cl:
                exact.append(s)
            elif any(c in lab for c in cl):
                substr.append(s)
        if exact:
            return max(exact, key=lambda s: s.chunk_count), True
        if substr:
            return max(substr, key=lambda s: s.chunk_count), False
        return None, False

    s, _exact = _find(labels)
    if s is not None:
        return s, False, s.label
    s, _exact = _find(fallback_labels)
    if s is not None:
        return s, True, s.label
    raise LookupError(
        f"No shelf for {labels} (or fallback {fallback_labels}) in facet '{facet}'"
    )


def disp_label(shelf) -> str:
    """The shelf's LLM alias (`display_label`) if present, else its raw label.

    With keyword/mock builds there is no alias, so this returns the raw FoodOn/
    CDNO label; after an LLM-aliased NB1 run it returns the friendly name.
    """
    dl = getattr(shelf, "display_label", None) or getattr(shelf.model, "display_label", None)
    return dl or shelf.label


def breadcrumb(fs: Any, shelf) -> list[str]:
    """Root→…→shelf path, using each shelf's display_label when aliased."""
    path = [disp_label(shelf)]
    cur = shelf
    seen = {shelf.shelf_id}
    while cur.parent_shelf_id is not None:
        parent = fs.graph.shelf(cur.parent_shelf_id)
        if parent is None or parent.shelf_id in seen:
            break
        path.append(disp_label(parent))
        seen.add(parent.shelf_id)
        cur = parent
    return list(reversed(path))


def shelf_record(fs: Any, shelf, *, fallback_used: bool) -> dict:
    """Serialize an anchor shelf to the `anchor_shelves.json` schema."""
    m = shelf.model
    return {
        "shelf_id": m.shelf_id,
        "label": m.label,
        "display_label": m.display_label,
        "facet": m.facet,
        "depth": m.depth,
        "foodon_id": m.foodon_id,
        "parent_shelf_id": m.parent_shelf_id,
        "chunk_count": m.chunk_count,
        "support_direct": m.support_direct,
        "support_lifted": m.support_lifted,
        "fallback_used": fallback_used,
        "breadcrumb": breadcrumb(fs, shelf),
    }


# ------------------------------------------------------------------ persistence

def save_graph(fs: Any) -> dict[str, str]:
    """Persist the in-memory graph (shelves/themes/cards) + chunks to disk.

    Returns the written paths. Cards have no public `list_*` accessor on the
    graph store, so we read them back per shelf/theme target.
    """
    from foodscholar.corpus import write_chunks_parquet

    shared_dir()
    shelves = fs.graph_store.list_shelves()
    themes = fs.graph_store.list_themes()
    cards = []
    for s in shelves:
        c = fs.graph_store.get_card(s.shelf_id, "shelf")
        if c is not None:
            cards.append(c)
    for t in themes:
        c = fs.graph_store.get_card(t.theme_id, "theme")
        if c is not None:
            cards.append(c)

    data = {
        "shelves": [s.model_dump(mode="json") for s in shelves],
        "themes": [t.model_dump(mode="json") for t in themes],
        "cards": [c.model_dump(mode="json") for c in cards],
        "config_hash": fs.config_hash,
    }
    save_json(GRAPH_SNAPSHOT, data)
    n = write_chunks_parquet(fs.chunk_store.scan(), CHUNK_SNAPSHOT)
    return {
        "graph": str(GRAPH_SNAPSHOT),
        "chunks": str(CHUNK_SNAPSHOT),
        "n_chunks": str(n),
        "n_shelves": str(len(shelves)),
        "n_themes": str(len(themes)),
        "n_cards": str(len(cards)),
    }


def load_graph(fs: Any, *, themes: bool = True, cards: bool = True) -> dict[str, int]:
    """Reload a graph snapshot written by `save_graph` into a fresh facade.

    Loads chunks from the snapshot parquet (preserving real embeddings + any
    shelf_ids/theme_ids denorm) and re-upserts shelves (+ optionally themes /
    cards) into the graph store.
    """
    from foodscholar.io.graph import Card, Shelf, Theme

    if not GRAPH_SNAPSHOT.exists() or not CHUNK_SNAPSHOT.exists():
        raise FileNotFoundError(
            "graph snapshot missing — run NB1 first to build & persist Layer A."
        )
    n_chunks = fs.load_chunks(str(CHUNK_SNAPSHOT))
    data = load_json(GRAPH_SNAPSHOT)
    shelf_models = [Shelf.model_validate(s) for s in data["shelves"]]
    fs.graph_store.upsert_shelves(shelf_models)

    # Rebuild the shelf↔chunk attachment EDGES from the denormalized
    # `chunk.shelf_ids`. The graph store's edge table (read by
    # build_layer_b via list_chunk_shelf_attachments) is RAM-only and is NOT
    # captured in the JSON snapshot — only the chunk-side denorm survives the
    # parquet round-trip. Without this, every shelf looks empty to Layer B.
    shelf_ids_present = {s.shelf_id for s in shelf_models}
    by_shelf: dict[str, list[tuple[str, list]]] = {}
    for c in fs.chunk_store.scan():
        for sid in c.shelf_ids:
            if sid in shelf_ids_present:
                by_shelf.setdefault(sid, []).append((c.chunk_id, []))
    for sid, atts in by_shelf.items():
        fs.graph_store.attach_chunks_to_shelf(sid, atts)

    n_themes = 0
    n_cards = 0
    if themes and data.get("themes"):
        theme_models = [Theme.model_validate(t) for t in data["themes"]]
        fs.graph_store.upsert_themes(theme_models)
        n_themes = len(theme_models)
        # Restore theme↔chunk edges from the chunk-side `theme_ids` denorm.
        theme_ids_present = {t.theme_id for t in theme_models}
        by_theme: dict[str, list[str]] = {}
        for c in fs.chunk_store.scan():
            for tid in c.theme_ids:
                if tid in theme_ids_present:
                    by_theme.setdefault(tid, []).append(c.chunk_id)
        for tid, cids in by_theme.items():
            fs.graph_store.attach_chunks_to_theme(tid, cids)
    if cards and data.get("cards"):
        card_models = [Card.model_validate(c) for c in data["cards"]]
        fs.graph_store.upsert_cards(card_models)
        n_cards = len(card_models)
    return {"chunks": n_chunks, "shelves": len(shelf_models), "themes": n_themes, "cards": n_cards}


def resolve_anchors(fs: Any) -> dict[str, dict]:
    """Resolve every CONFIG anchor against the current build.

    Returns ``{key: {"shelf": ShelfHandle, "fallback_used": bool,
    "matched_label": str}}``.
    """
    out: dict[str, dict] = {}
    for key, a in CONFIG["anchors"].items():
        shelf, fb, matched = resolve_shelf(
            fs, a["facet"], a["labels"], a["fallback_labels"]
        )
        out[key] = {"shelf": shelf, "fallback_used": fb, "matched_label": matched}
    return out


# ============================================================ slide-figure toolkit
#
# Every case-study figure is a 16:9 slide-ready PNG built through these helpers,
# so NB1–NB4 share one look: brand palette, consistent type scale, a title +
# optional eyebrow/caption frame, and generous margins. Pure matplotlib — no
# extra deps. Call `init_mpl()` once per notebook; build with `slide()`.

# 16:9 at a deck-friendly size; 200 DPI → ~2560×1440 px, crisp on projectors.
SLIDE_W, SLIDE_H = 12.8, 7.2
SLIDE_DPI = 200


def init_mpl():
    """Install the slide-deck matplotlib theme. Returns the pyplot module."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (SLIDE_W, SLIDE_H),
            "figure.dpi": SLIDE_DPI,
            "savefig.dpi": SLIDE_DPI,
            "figure.facecolor": COLORS["surface"],
            "savefig.facecolor": COLORS["surface"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "text.color": COLORS["ink"],
            "axes.titlesize": 17,
            "axes.titlecolor": COLORS["ink"],
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.labelcolor": COLORS["muted"],
            "axes.edgecolor": COLORS["rule"],
            "axes.facecolor": COLORS["surface"],
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12.5,
            "legend.frameon": False,
        }
    )
    return plt


def slide(title: str, *, eyebrow: str | None = None, caption: str | None = None):
    """Open a 16:9 slide figure with a title band; return ``(fig, content_ax)``.

    - ``eyebrow``: small uppercase kicker above the title (e.g. "LENS 1 · M1").
    - ``caption``: one-line takeaway pinned to the bottom.
    The returned axis is a clean content area (no spines/ticks, 0–1 coords) you
    draw into; or ignore it and add your own axes via ``fig.add_axes``.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H))
    # Title band
    y = 0.965
    if eyebrow:
        fig.text(0.045, 0.985, eyebrow.upper(), fontsize=12.5, weight="bold",
                 color=COLORS["teal"], va="top", ha="left")
        y = 0.945
    fig.text(0.045, y, title, fontsize=22, weight="bold", color=COLORS["ink"],
             va="top", ha="left")
    # Accent rule under the title
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.86, 0.86], color=COLORS["rule"],
                              lw=1.2, solid_capstyle="round"))
    if caption:
        fig.text(0.045, 0.035, caption, fontsize=13, color=COLORS["muted"],
                 va="bottom", ha="left", style="italic")
    # Content axis occupies the band below the title, above the caption.
    bottom = 0.12 if caption else 0.07
    ax = fig.add_axes([0.045, bottom, 0.91, 0.86 - bottom - 0.02])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def save_slide(fig, path: Path) -> Path:
    """Save a slide figure (brand background) and close it. Returns the path."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=COLORS["surface"])
    plt.close(fig)
    return path


def chip(ax, x, y, text, *, color, fg="white", fontsize=13, weight="bold", pad=0.45):
    """Draw a rounded pill/chip at (x, y) in axis coords. Returns approx width."""
    ax.text(x, y, text, fontsize=fontsize, color=fg, weight=weight,
            ha="left", va="center",
            bbox=dict(boxstyle=f"round,pad={pad}", fc=color, ec="none"))
    return 0.0125 * len(str(text)) + 0.04


def kpi(ax, x, y, value, label, *, color, w=0.2, h=0.42, value_fontsize=28):
    """A big-number KPI callout (value over a small label) at (x, y) top-left."""
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                         fc=COLORS["panel"], ec=COLORS["rule"], lw=1.0,
                         transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)
    ax.text(x + w / 2, y - h * 0.40, str(value), fontsize=value_fontsize, weight="bold",
            color=color, ha="center", va="center", transform=ax.transAxes)
    ax.text(x + w / 2, y - h * 0.80, label, fontsize=11.5, color=COLORS["muted"],
            ha="center", va="center", transform=ax.transAxes)


def labelled_bars(ax, categories, series, *, ylabel="", value_fmt="{:.0f}"):
    """Grouped bar chart on a real axis. ``series`` = list of (name, values, color).

    Annotates each bar with its value, applies the deck style, and returns the
    axis for further tweaks. Use inside a ``slide()`` by adding a sub-axis.
    """
    import numpy as np

    n = len(series)
    x = np.arange(len(categories))
    width = min(0.8 / max(n, 1), 0.38)
    offsets = (np.arange(n) - (n - 1) / 2) * width
    for (name, vals, color), off in zip(series, offsets):
        bars = ax.bar(x + off, vals, width=width, color=color, label=name,
                      edgecolor="white", linewidth=0.8, zorder=3)
        for b, v in zip(bars, vals):
            if v is None:
                continue
            ax.annotate(value_fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=11, color=COLORS["ink"], weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=COLORS["rule"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    if n > 1:
        ax.legend(loc="upper right")
    return ax


def excerpt(text: str, n: int = 240) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"
