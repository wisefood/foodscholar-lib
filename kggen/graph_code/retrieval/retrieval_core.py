"""Core building blocks for KGGEN_extended hybrid passage retrieval.

This module contains the *functions only* (no execution / no query loop):

    1. loading the graph + passages,
    2. building the retrieval indexes (cached to disk),
    3. building the embeddings (cached to disk),
    4. the three scoring branches,
    5. the hybrid retrieval function,
    6. a `Retriever` class that lazily loads indexes + embeddings
       (creating them once and reusing them for later queries).

Pipeline (same as `retrieval_passages_ppr_subgraph.ipynb`):

    Query
     |
     +-- cosine(query, passage text)                     weight 0.3
     +-- mean cosine(query, triplets of a passage)       weight 0.3
     +-- PPR on the unified k-hop subgraph of the
         top-k entities, propagated to passages           weight 0.4

Notes
-----
* Nothing heavy is imported at module level (torch / sentence_transformers are
  imported lazily) so that ``CUDA_VISIBLE_DEVICES`` can be set after import.
* Indexes and embeddings are cached under ``cache_dir`` and reloaded on the next
  run, so querying is fast after the first run.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

# Bump when the on-disk layout of the caches changes (invalidates old caches).
CACHE_VERSION = 2

DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH_PATH = DEFAULT_BASE_DIR / "kg_output_chunks" / "_aggregated_all" / "aggregated_graph.graphml"
DEFAULT_PASSAGES_PATH = DEFAULT_BASE_DIR / "kg_output_chunks" / "_aggregated_all" / "passages.json"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
DEFAULT_HF_CACHE = Path("/mnt/data/huggingface_cache")

DEFAULT_ENCODER_MODEL = "BAAI/bge-large-en-v1.5"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RetrievalConfig:
    """All tunables of the retrieval pipeline."""

    graph_path: Path = DEFAULT_GRAPH_PATH
    passages_path: Path = DEFAULT_PASSAGES_PATH
    cache_dir: Path = DEFAULT_CACHE_DIR
    hf_cache_dir: Path = DEFAULT_HF_CACHE

    encoder_model_name: str = DEFAULT_ENCODER_MODEL

    top_k: int = 5               # number of source passages returned per query
    subgraph_depth: int = 5      # k-hop depth of the subgraph around the top-k entities
    ppr_alpha: float = 0.85      # PageRank damping factor
    ppr_max_iter: int = 100
    batch_size: int = 32         # embedding batch size (auto-halved on CUDA OOM)

    # hybrid weights (must sum to 1.0)
    w_text: float = 0.3
    w_triplet: float = 0.3
    w_ppr: float = 0.4

    # compute device
    gpu_index: Optional[int] = 0     # GPU 0 (RTX 4090) is usually busy
    device: str = "auto"             # "auto" | "cuda" | "cpu"

    verbose: bool = True

    def __post_init__(self) -> None:
        self.graph_path = Path(self.graph_path)
        self.passages_path = Path(self.passages_path)
        self.cache_dir = Path(self.cache_dir)
        self.hf_cache_dir = Path(self.hf_cache_dir)
        total = self.w_text + self.w_triplet + self.w_ppr
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Hybrid weights must sum to 1.0 (got {total}).")

    @property
    def indexes_cache_file(self) -> Path:
        return self.cache_dir / "indexes" / "indexes.pkl"

    @property
    def indexes_meta_file(self) -> Path:
        return self.cache_dir / "indexes" / "indexes_meta.json"

    @property
    def embeddings_dir(self) -> Path:
        return self.cache_dir / "embeddings" / _slug(self.encoder_model_name)

    def describe(self) -> str:
        return (
            f"graph          : {self.graph_path}\n"
            f"passages       : {self.passages_path}\n"
            f"cache          : {self.cache_dir}\n"
            f"encoder        : {self.encoder_model_name}\n"
            f"top_k          : {self.top_k}\n"
            f"subgraph depth : {self.subgraph_depth}\n"
            f"weights        : text={self.w_text}, triplet={self.w_triplet}, ppr={self.w_ppr}"
        )


def _slug(text: str) -> str:
    """Filesystem-safe slug for a model name."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in text)


def _log(config: RetrievalConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Bundles
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class IndexBundle:
    """Everything derived from the aggregated graph (all cached on disk)."""

    meta: Dict[str, Any] = field(default_factory=dict)
    # list of {'source', 'relation', 'target', 'passage_ids'}
    triple_passages: List[Dict[str, Any]] = field(default_factory=list)
    # passage_id -> [triple indices]
    passage_to_triples: Dict[str, List[int]] = field(default_factory=dict)
    # entity -> [passage_id, ...]  (provenance via Source edges)
    entity_to_passages: Dict[str, List[str]] = field(default_factory=dict)
    # relation-only MultiDiGraph (distinct predicates kept as parallel edges)
    relation_graph: Any = None
    # entity node names, in the order used for the embeddings
    entity_nodes: List[str] = field(default_factory=list)
    # passage_id -> node attribute dict as stored ON THE GRAPH
    # (keys: 'type', 'text', 'metadata'; the GraphML 'metadata' string is parsed
    # into a dict, so `passage_node_attrs[pid]['metadata']` is a real dict)
    passage_node_attrs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class EmbeddingBundle:
    """All embeddings used by the three scoring branches (cached on disk)."""

    model_name: str = ""
    passage_ids: List[str] = field(default_factory=list)
    passage_embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    triplet_embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    entity_nodes: List[str] = field(default_factory=list)
    entity_embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    entity_index: Dict[str, int] = field(default_factory=dict)
    loaded_from_cache: bool = False

    @property
    def dim(self) -> int:
        return int(self.passage_embeddings.shape[1]) if self.passage_embeddings.size else 0


@dataclass
class RetrievalResult:
    """Result of a single query."""

    query: str
    passage_ids: List[str] = field(default_factory=list)
    passage_texts: List[str] = field(default_factory=list)
    passage_records: List[Dict[str, Any]] = field(default_factory=list)
    # raw node-attribute dicts of the retrieved passages, as stored ON THE GRAPH
    # (keys: 'type', 'text', 'metadata' — with 'metadata' already parsed to a dict)
    passage_graph_attrs: List[Dict[str, Any]] = field(default_factory=list)
    score_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    top_entities: List[str] = field(default_factory=list)
    top_entity_scores: Dict[str, float] = field(default_factory=dict)
    subgraph_nodes: int = 0
    subgraph_edges: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # -- metadata access ----------------------------------------------------
    def get_metadata(
        self,
        keys: Optional[Sequence[str]] = None,
        include_scores: bool = False,
    ) -> List[Dict[str, Any]]:
        """Metadata of the retrieved passages, one dict per passage in rank order.

        Parameters
        ----------
        keys
            Optional subset of metadata fields to keep, e.g.
            ``("file", "title", "page_number")``. ``None`` keeps everything.
        include_scores
            Also add ``final_score`` / ``text_sim`` / ``triplet_sim`` / ``ppr_score``.

        Example
        -------
        >>> result.get_metadata(keys=("file", "page_number"), include_scores=True)
        [{'rank': 1, 'passage_id': '...', 'file': '...', 'page_number': 102,
          'final_score': 0.9252, ...}, ...]
        """
        entries: List[Dict[str, Any]] = []
        for rank, pid in enumerate(self.passage_ids, 1):
            record = self.passage_records[rank - 1] if rank <= len(self.passage_records) else {}
            meta = dict(record.get("metadata") or {})
            if keys is not None:
                meta = {k: meta.get(k) for k in keys}
            entry: Dict[str, Any] = {"rank": rank, "passage_id": pid, **meta}
            if include_scores:
                bd = self.score_breakdown[rank - 1] if rank <= len(self.score_breakdown) else {}
                entry.update({
                    "final_score": bd.get("final_score"),
                    "text_sim": bd.get("text_sim"),
                    "triplet_sim": bd.get("triplet_sim"),
                    "ppr_score": bd.get("ppr_score"),
                })
            entries.append(entry)
        return entries

    def metadata_json(self, keys: Optional[Sequence[str]] = None,
                      include_scores: bool = False, indent: int = 2) -> str:
        """`get_metadata(...)` serialized as a JSON string."""
        return json.dumps(self.get_metadata(keys=keys, include_scores=include_scores),
                          indent=indent, default=str)

    # -- graph metadata access ----------------------------------------------
    def get_graph_metadata(
        self,
        keys: Optional[Sequence[str]] = None,
        include_scores: bool = False,
        include_graph_attrs: bool = False,
    ) -> List[Dict[str, Any]]:
        """Metadata of the retrieved passages **as stored on the graph nodes**.

        This is the `metadata` attribute of the passage nodes of the aggregated
        graph (not the `passages.json` copy). ``keys``, ``include_scores`` and
        ``include_graph_attrs`` behave as in `get_graph_passage_metadata`.

        Example
        -------
        >>> result.get_graph_metadata(keys=("file", "page_number"), include_scores=True)
        [{'rank': 1, 'passage_id': '...', 'file': '...', 'page_number': 38,
          'final_score': 0.9321, ...}, ...]
        """
        entries: List[Dict[str, Any]] = []
        for rank, pid in enumerate(self.passage_ids, 1):
            attrs = (self.passage_graph_attrs[rank - 1]
                     if rank <= len(self.passage_graph_attrs) else {})
            raw_meta = attrs.get("metadata")
            meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
            if keys is not None:
                meta = {k: meta.get(k) for k in keys}
            entry: Dict[str, Any] = {"rank": rank, "passage_id": pid, **meta}
            if include_graph_attrs:
                entry["graph_type"] = attrs.get("type")
                entry["graph_text"] = attrs.get("text")
            if include_scores:
                bd = self.score_breakdown[rank - 1] if rank <= len(self.score_breakdown) else {}
                entry.update({
                    "final_score": bd.get("final_score"),
                    "text_sim": bd.get("text_sim"),
                    "triplet_sim": bd.get("triplet_sim"),
                    "ppr_score": bd.get("ppr_score"),
                })
            entries.append(entry)
        return entries

    def graph_metadata_json(self, keys: Optional[Sequence[str]] = None,
                            include_scores: bool = False,
                            include_graph_attrs: bool = False,
                            indent: int = 2) -> str:
        """`get_graph_metadata(...)` serialized as a JSON string."""
        return json.dumps(
            self.get_graph_metadata(keys=keys, include_scores=include_scores,
                                    include_graph_attrs=include_graph_attrs),
            indent=indent, default=str,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────


def load_passages(passages_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load `passages.json` → {passage_id: {'text': ..., 'metadata': {...}}}."""
    with open(passages_path) as fh:
        passages = json.load(fh)
    return passages


# Metadata fields carried by `passages.json` (the file also has `text`).
METADATA_FIELDS: Tuple[str, ...] = (
    "file", "urn", "pdf_name", "country", "title", "audience",
    "pages", "heading", "page_number",
)


def get_passage_metadata(
    passage_ids: Sequence[str],
    passages: Dict[str, Dict[str, Any]],
    keys: Optional[Sequence[str]] = None,
    include_rank: bool = True,
) -> List[Dict[str, Any]]:
    """Look up the metadata of passages by id.

    Only needs the `passages.json` mapping — no graph, no embeddings, no GPU.

    Parameters
    ----------
    passage_ids
        Passage ids to look up (e.g. ``result.passage_ids``).
    passages
        The ``{passage_id: {'text': ..., 'metadata': {...}}}`` mapping.
    keys
        Optional subset of metadata fields to keep (``None`` keeps all fields).
    include_rank
        Prefix each entry with its 1-based ``rank``.

    Returns
    -------
    list of dicts, one per passage, with ``passage_id`` plus its metadata fields.
    Unknown ids yield ``{'passage_id': id, 'missing': True}``.
    """
    entries: List[Dict[str, Any]] = []
    for rank, pid in enumerate(passage_ids, 1):
        record = passages.get(pid)
        if record is None:
            entry: Dict[str, Any] = {"passage_id": pid, "missing": True}
        else:
            meta = dict(record.get("metadata") or {})
            if keys is not None:
                meta = {k: meta.get(k) for k in keys}
            entry = {"passage_id": pid, **meta}
        entries.append({"rank": rank, **entry} if include_rank else entry)
    return entries


def load_metadata_index(passages_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load only the metadata (without the passage texts) from `passages.json`."""
    with open(passages_path) as fh:
        raw = json.load(fh)
    return {pid: dict(rec.get("metadata") or {}) for pid, rec in raw.items()}


# Node attributes carried by passage nodes in the aggregated graph.
GRAPH_PASSAGE_ATTRS: Tuple[str, ...] = ("type", "text", "metadata")


def graph_passage_metadata_index(bundle: IndexBundle) -> Dict[str, Dict[str, Any]]:
    """``{passage_id: metadata}`` taken from the passage **nodes of the graph**.

    The graph stores the metadata as the `metadata` attribute of each passage
    node (a JSON string in the GraphML file, parsed into a dict by `build_indexes`).
    """
    index: Dict[str, Dict[str, Any]] = {}
    for pid, attrs in bundle.passage_node_attrs.items():
        meta = attrs.get("metadata")
        index[pid] = dict(meta) if isinstance(meta, dict) else {}
    return index


def get_graph_passage_metadata(
    passage_ids: Sequence[str],
    bundle: IndexBundle,
    keys: Optional[Sequence[str]] = None,
    include_rank: bool = True,
    include_graph_attrs: bool = False,
) -> List[Dict[str, Any]]:
    """Metadata of passages as stored on the passage **nodes of the graph**.

    Unlike `get_passage_metadata` (which reads `passages.json`), this reads the
    node attributes of the aggregated graph, which are cached together with the
    indexes — no extra file access is needed.

    Parameters
    ----------
    passage_ids
        Passage ids to look up (e.g. ``result.passage_ids``).
    bundle
        The `IndexBundle` (`Retriever.indexes`).
    keys
        Optional subset of metadata fields to keep (``None`` keeps all fields).
    include_rank
        Prefix each entry with its 1-based ``rank``.
    include_graph_attrs
        Also expose the raw node attributes (``graph_type``, ``graph_text``).

    Unknown ids yield ``{'passage_id': id, 'missing': True}``.
    """
    entries: List[Dict[str, Any]] = []
    for rank, pid in enumerate(passage_ids, 1):
        attrs = bundle.passage_node_attrs.get(pid)
        if attrs is None:
            entry: Dict[str, Any] = {"passage_id": pid, "missing": True}
        else:
            raw_meta = attrs.get("metadata")
            meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
            if keys is not None:
                meta = {k: meta.get(k) for k in keys}
            entry = {"passage_id": pid, **meta}
            if include_graph_attrs:
                entry["graph_type"] = attrs.get("type")
                entry["graph_text"] = attrs.get("text")
        entries.append({"rank": rank, **entry} if include_rank else entry)
    return entries


def load_graph(graph_path: Path, verbose: bool = False):
    """Read the aggregated graphml as a (Multi)DiGraph."""
    import networkx as nx

    t0 = time.time()
    graph = nx.read_graphml(graph_path)
    if not nx.is_directed(graph):
        graph = graph.to_directed()
    if verbose:
        print(f"Graph loaded: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges in {time.time() - t0:.1f}s", flush=True)
    return graph


def _graph_fingerprint(graph_path: Path) -> Dict[str, Any]:
    stat = graph_path.stat()
    return {
        "graph_path": str(graph_path),
        "graph_size": stat.st_size,
        "graph_mtime": round(stat.st_mtime, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Indexes (triple/passage/entity lookups + relation-only graph)
# ─────────────────────────────────────────────────────────────────────────────


def _build_indexes_from_graph(graph, passages: Dict[str, Dict[str, Any]]) -> IndexBundle:
    """Derive every lookup table from the raw aggregated graph."""
    import networkx as nx

    # 1. triple -> passage ids (Relation edges carry `passage_ids`)
    triple_passages: List[Dict[str, Any]] = []
    for u, v, d in graph.edges(data=True):
        if d.get("type") != "Relation":
            continue
        pids = [p.strip() for p in str(d.get("passage_ids", "")).split(",") if p.strip()]
        triple_passages.append({
            "source": u,
            "relation": d.get("relation", "<none>"),
            "target": v,
            "passage_ids": pids,
        })

    # 2. passage -> triple indices (inverse map, for the mean-triplet branch)
    passage_to_triples: Dict[str, List[int]] = defaultdict(list)
    for i, t in enumerate(triple_passages):
        for p in t["passage_ids"]:
            passage_to_triples[p].append(i)

    # 3. entity -> passages (provenance via Source edges, used to spread PPR mass)
    entity_to_passages: Dict[str, List[str]] = defaultdict(set)  # type: ignore[assignment]
    for u, v, d in graph.edges(data=True):
        if d.get("type") == "Source":
            entity_to_passages[u].add(v)  # type: ignore[union-attr]
    entity_to_passages = {k: sorted(v) for k, v in entity_to_passages.items()}  # type: ignore[union-attr]

    # 4. Relation-only MultiDiGraph for the k-hop subgraphs.
    #    Distinct predicates are kept as parallel edges so that nx.pagerank
    #    counts them as multiple votes (mirroring the triplet-similarity branch).
    relation_graph = nx.MultiDiGraph()
    for u, v, d in graph.edges(data=True):
        if d.get("type") != "Relation":
            continue
        if u == v or not str(u).strip() or not str(v).strip():
            continue
        relation_graph.add_edge(u, v, relation=d.get("relation", "<none>"))

    # 5. entity nodes (those eligible to become PPR seeds)
    entity_nodes = [
        n for n, d in graph.nodes(data=True)
        if d.get("type") == "entity" and str(n).strip()
    ]

    # 6. passage node attributes as stored on the graph.
    #    GraphML only holds scalars, so the `metadata` attribute is a JSON string;
    #    parse it back into a dict here.
    passage_node_attrs: Dict[str, Dict[str, Any]] = {}
    for n, d in graph.nodes(data=True):
        if d.get("type") != "passage":
            continue
        attrs = dict(d)
        raw_meta = attrs.get("metadata")
        if isinstance(raw_meta, str):
            try:
                attrs["metadata"] = json.loads(raw_meta)
            except json.JSONDecodeError:
                pass  # keep the raw string if it is not valid JSON
        passage_node_attrs[str(n)] = attrs

    return IndexBundle(
        triple_passages=triple_passages,
        passage_to_triples=dict(passage_to_triples),
        entity_to_passages=entity_to_passages,
        relation_graph=relation_graph,
        entity_nodes=entity_nodes,
        passage_node_attrs=passage_node_attrs,
    )


def _indexes_meta(bundle: IndexBundle, passages: Dict[str, Any], graph_path: Path) -> Dict[str, Any]:
    meta = {
        "cache_version": CACHE_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **_graph_fingerprint(graph_path),
        "n_passages": len(passages),
        "n_triples": len(bundle.triple_passages),
        "n_passages_with_triples": len(bundle.passage_to_triples),
        "n_entities_with_passages": len(bundle.entity_to_passages),
        "n_relation_nodes": bundle.relation_graph.number_of_nodes(),
        "n_relation_edges": bundle.relation_graph.number_of_edges(),
        "n_entity_nodes": len(bundle.entity_nodes),
        "n_passage_nodes_with_attrs": len(bundle.passage_node_attrs),
    }
    return meta


def build_indexes(
    config: Optional[RetrievalConfig] = None,
    force_rebuild: bool = False,
) -> Tuple[IndexBundle, Dict[str, Dict[str, Any]]]:
    """Build (or load from cache) the lookup tables derived from the graph.

    Returns
    -------
    (IndexBundle, passages)
    """
    config = config or RetrievalConfig()

    _log(config, f"Loading passages from {config.passages_path} ...")
    passages = load_passages(config.passages_path)
    _log(config, f"  → {len(passages)} passages")

    if not force_rebuild and config.indexes_cache_file.exists():
        # cheap check first: reuse cached meta only if it matches the graph file
        try:
            probe = _graph_fingerprint(config.graph_path)
        except OSError:
            probe = {}
        probe["cache_version"] = CACHE_VERSION
        probe["n_passages"] = len(passages)
        cached_meta: Dict[str, Any] = {}
        if config.indexes_meta_file.exists():
            try:
                with open(config.indexes_meta_file) as fh:
                    cached_meta = json.load(fh)
            except (OSError, json.JSONDecodeError):
                cached_meta = {}
        identical_probe = all(cached_meta.get(k) == v for k, v in probe.items())
        if identical_probe:
            _log(config, f"Loading indexes from cache {config.indexes_cache_file} ...")
            t0 = time.time()
            try:
                with open(config.indexes_cache_file, "rb") as fh:
                    bundle: IndexBundle = pickle.load(fh)
                if not hasattr(bundle, "passage_node_attrs"):
                    raise ValueError("cached indexes lack the graph passage attributes")
                _log(config, f"  → indexes loaded in {time.time() - t0:.1f}s "
                            f"({len(bundle.triple_passages)} triples)")
                return bundle, passages
            except Exception as exc:  # corrupted / outdated cache → rebuild
                _log(config, f"  ! cache unreadable ({exc}); rebuilding indexes")

    _log(config, f"Building indexes from {config.graph_path} ...")
    graph = load_graph(config.graph_path, verbose=config.verbose)
    t0 = time.time()
    bundle = _build_indexes_from_graph(graph, passages)
    meta = _indexes_meta(bundle, passages, config.graph_path)
    bundle.meta = meta
    _log(config, f"  → {meta['n_triples']} triples, "
                f"{meta['n_passages_with_triples']} passages with triples, "
                f"{meta['n_entities_with_passages']} entities with passages, "
                f"relation graph {meta['n_relation_nodes']}n/{meta['n_relation_edges']}e "
                f"in {time.time() - t0:.1f}s")

    config.indexes_cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config.indexes_cache_file, "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(config.indexes_meta_file, "w") as fh:
        json.dump(meta, fh, indent=2)
    _log(config, f"  → indexes cached at {config.indexes_cache_file}")

    del graph  # not needed anymore (frees ~a few hundred MB)
    return bundle, passages


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────


def _configure_hf_cache(hf_cache_dir: Path) -> None:
    """Point HuggingFace / sentence-transformers at a shared cache directory."""
    hf_cache_dir = Path(hf_cache_dir)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache_dir / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(hf_cache_dir / "sentence-transformers"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _configure_gpu(config: RetrievalConfig) -> None:
    """Expose a single GPU to this process (must run before torch is imported)."""
    if config.gpu_index is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_index)
    # reduce fragmentation on a partially occupied GPU
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


_EMBEDDER_CACHE: Dict[Tuple[str, str], Any] = {}


def get_embedder(config: Optional[RetrievalConfig] = None):
    """Load (and memoize) the sentence-transformer encoder."""
    config = config or RetrievalConfig()
    _configure_hf_cache(config.hf_cache_dir)
    _configure_gpu(config)

    from sentence_transformers import SentenceTransformer

    if config.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    key = (config.encoder_model_name, device)
    if key not in _EMBEDDER_CACHE:
        _log(config, f"Loading encoder {config.encoder_model_name} on {device} ...")
        _EMBEDDER_CACHE[key] = SentenceTransformer(
            config.encoder_model_name,
            cache_folder=str(Path(config.hf_cache_dir) / "sentence-transformers"),
            device=device,
        )
    return _EMBEDDER_CACHE[key]


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _embed(embedder, texts: Sequence[str], batch_size: int, desc: str,
           verbose: bool = True) -> np.ndarray:
    """Encode `texts`, surviving a partially occupied GPU.

    If the GPU runs out of memory the batch size is halved and the batch is
    retried; as a last resort the model is moved to the CPU.
    """
    import time as _time

    texts = list(texts)
    t0 = _time.time()
    batch_size = max(1, int(batch_size))

    while True:
        try:
            embeddings = embedder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=verbose,
                convert_to_numpy=True,
            )
            if verbose:
                print(f"  → {desc}: {embeddings.shape} in {_time.time() - t0:.1f}s "
                      f"(batch_size={batch_size})", flush=True)
            return np.asarray(embeddings, dtype=np.float32)
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            import torch

            torch.cuda.empty_cache()
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                if verbose:
                    print(f"  ! CUDA OOM while encoding {desc} → retrying with "
                          f"batch_size={batch_size}", flush=True)
                continue
            # single-sample batches still do not fit → fall back to CPU
            if verbose:
                print(f"  ! CUDA OOM with batch_size=1 → moving encoder to CPU "
                      f"for {desc}", flush=True)
            embedder.to("cpu")
            torch.cuda.empty_cache()
            batch_size = 32


def _embeddings_fingerprint(config: RetrievalConfig, bundle: IndexBundle,
                            passages: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "model_name": config.encoder_model_name,
        "n_passages": len(passages),
        "n_triples": len(bundle.triple_passages),
        "n_entities": len(bundle.entity_nodes),
    }


def _embeddings_cache_files(config: RetrievalConfig) -> Dict[str, Path]:
    d = config.embeddings_dir
    return {
        "dir": d,
        "meta": d / "embeddings_meta.json",
        "ids": d / "ids.json",
        "passages": d / "passage_embeddings.npy",
        "triplets": d / "triplet_embeddings.npy",
        "entities": d / "entity_embeddings.npy",
    }


def _try_load_embeddings(config: RetrievalConfig, expected: Dict[str, Any]) -> Optional[EmbeddingBundle]:
    files = _embeddings_cache_files(config)
    if not files["meta"].exists():
        return None
    try:
        with open(files["meta"]) as fh:
            meta = json.load(fh)
        if any(meta.get(k) != v for k, v in expected.items()):
            _log(config, "  ! embeddings cache is stale (config changed) → rebuilding")
            return None
        with open(files["ids"]) as fh:
            ids = json.load(fh)
        bundle = EmbeddingBundle(
            model_name=meta["model_name"],
            passage_ids=ids["passage_ids"],
            passage_embeddings=np.load(files["passages"]),
            triplet_embeddings=np.load(files["triplets"]),
            entity_nodes=ids["entity_nodes"],
            entity_embeddings=np.load(files["entities"]),
            loaded_from_cache=True,
        )
        bundle.entity_index = {n: i for i, n in enumerate(bundle.entity_nodes)}
        return bundle
    except Exception as exc:
        _log(config, f"  ! embeddings cache unreadable ({exc}) → rebuilding")
        return None


def build_embeddings(
    bundle: IndexBundle,
    passages: Dict[str, Dict[str, Any]],
    config: Optional[RetrievalConfig] = None,
    force_rebuild: bool = False,
    embedder=None,
) -> EmbeddingBundle:
    """Build (or load from cache) the passage / triplet / entity embeddings."""
    config = config or RetrievalConfig()
    expected = _embeddings_fingerprint(config, bundle, passages)

    files = _embeddings_cache_files(config)
    if not force_rebuild:
        cached = _try_load_embeddings(config, expected)
        if cached is not None:
            _log(config, f"Embeddings loaded from cache {files['dir']} "
                        f"(passages {cached.passage_embeddings.shape}, "
                        f"triplets {cached.triplet_embeddings.shape}, "
                        f"entities {cached.entity_embeddings.shape})")
            return cached
        if any(files[k].exists() for k in ("passages", "triplets", "entities")):
            _log(config, f"No usable embeddings cache in {files['dir']} → creating embeddings")
    else:
        _log(config, "Forcing rebuild of all embeddings")

    embedder = embedder or get_embedder(config)
    _log(config, f"Creating embeddings (device={embedder.device}) ...")

    passage_ids = list(passages.keys())
    passage_texts = [passages[p]["text"] for p in passage_ids]
    triplet_texts = [
        f"{t['source']} {t['relation']} {t['target']}" for t in bundle.triple_passages
    ]
    entity_nodes = list(bundle.entity_nodes)

    passage_embeddings = _embed(embedder, passage_texts, config.batch_size,
                                "passage embeddings", config.verbose)
    triplet_embeddings = _embed(embedder, triplet_texts, config.batch_size,
                                "triplet embeddings", config.verbose)
    entity_embeddings = _embed(embedder, entity_nodes, config.batch_size,
                                "entity embeddings", config.verbose)

    out = EmbeddingBundle(
        model_name=config.encoder_model_name,
        passage_ids=passage_ids,
        passage_embeddings=passage_embeddings,
        triplet_embeddings=triplet_embeddings,
        entity_nodes=entity_nodes,
        entity_embeddings=entity_embeddings,
        entity_index={n: i for i, n in enumerate(entity_nodes)},
        loaded_from_cache=False,
    )

    files["dir"].mkdir(parents=True, exist_ok=True)
    np.save(files["passages"], out.passage_embeddings)
    np.save(files["triplets"], out.triplet_embeddings)
    np.save(files["entities"], out.entity_embeddings)
    with open(files["ids"], "w") as fh:
        json.dump({"passage_ids": out.passage_ids, "entity_nodes": out.entity_nodes}, fh)
    meta = {**expected, "dim": out.dim, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(files["meta"], "w") as fh:
        json.dump(meta, fh, indent=2)
    _log(config, f"  → embeddings cached at {files['dir']}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Scoring branches
# ─────────────────────────────────────────────────────────────────────────────


def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize a dict of scores to [0, 1]."""
    if not scores:
        return scores
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _cosine_to_all(q_emb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_similarity

    return cosine_similarity(q_emb, matrix)[0]


def score_text_similarity(q_emb: np.ndarray, embeddings: EmbeddingBundle) -> Dict[str, float]:
    """Branch 1: cosine(query, passage text)."""
    sims = _cosine_to_all(q_emb, embeddings.passage_embeddings)
    return {pid: float(s) for pid, s in zip(embeddings.passage_ids, sims)}


def score_triplet_similarity(q_emb: np.ndarray, bundle: IndexBundle,
                             embeddings: EmbeddingBundle,
                             passages: Dict[str, Any]) -> Dict[str, float]:
    """Branch 2: mean cosine(query, triple) over all triples of a passage."""
    sims = _cosine_to_all(q_emb, embeddings.triplet_embeddings)
    per_passage: Dict[str, List[float]] = defaultdict(list)
    for i, s in enumerate(sims):
        for p in bundle.triple_passages[i]["passage_ids"]:
            if p in passages:
                per_passage[p].append(float(s))
    return {p: float(np.mean(v)) for p, v in per_passage.items()}


def score_ppr_subgraph(
    q_emb: np.ndarray,
    bundle: IndexBundle,
    embeddings: EmbeddingBundle,
    config: RetrievalConfig,
    top_k: Optional[int] = None,
    depth: Optional[int] = None,
) -> Tuple[Dict[str, float], List[str], Dict[str, float], Any]:
    """Branch 3: PPR on the unified k-hop subgraph of the top-k entities.

    Steps: top-k entities by query–entity cosine → unified k-hop subgraph
    (union of undirected ego-graphs) → Personalized PageRank seeded on those
    entities → PPR mass propagated to passages via `Source` edges.

    Returns ``(passage_ppr_scores, top_entities, top_entity_scores, subgraph)``.
    """
    import networkx as nx

    top_k = top_k or config.top_k
    depth = depth if depth is not None else config.subgraph_depth
    relation_graph = bundle.relation_graph

    # 1. top-k entities by query–entity cosine similarity
    sims = _cosine_to_all(q_emb, embeddings.entity_embeddings)
    order = np.argsort(-sims)[:top_k]
    top_entity_scores = {embeddings.entity_nodes[i]: float(sims[i]) for i in order}

    # keep only entities that participate in the Relation graph (ego_graph needs them)
    top_entities = [e for e in top_entity_scores if e in relation_graph]
    top_entity_scores = {e: top_entity_scores[e] for e in top_entities}
    if not top_entities:
        return {}, [], {}, nx.MultiDiGraph()

    # 2. unified k-hop subgraph = union of the k-hop neighborhoods (undirected)
    sub = nx.MultiDiGraph()
    for e in top_entities:
        ego = nx.ego_graph(relation_graph, e, radius=depth, undirected=True)
        sub.add_nodes_from(ego.nodes())
        sub.add_edges_from(ego.edges(keys=True, data=True))
    sub.add_nodes_from(top_entities)  # keep isolated seeds present

    # 3. Personalized PageRank seeded on the top-k entities
    #    (parallel edges = distinct predicates count as multiple votes)
    personalization = {e: 1.0 for e in top_entities}
    ppr = nx.pagerank(sub, alpha=config.ppr_alpha, personalization=personalization,
                      max_iter=config.ppr_max_iter)

    # 4. propagate PPR mass to passages via Source edges
    passage_ppr: Dict[str, float] = defaultdict(float)
    for e, s in ppr.items():
        for p in bundle.entity_to_passages.get(e, ()):
            passage_ppr[p] += s
    return dict(passage_ppr), top_entities, top_entity_scores, sub


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid retrieval
# ─────────────────────────────────────────────────────────────────────────────


def retrieve_passages_ppr(
    query: str,
    bundle: IndexBundle,
    embeddings: EmbeddingBundle,
    passages: Dict[str, Dict[str, Any]],
    config: Optional[RetrievalConfig] = None,
    embedder=None,
    top_k: Optional[int] = None,
    depth: Optional[int] = None,
) -> RetrievalResult:
    """Hybrid retrieval: w_text·textSim + w_triplet·meanTripletSim + w_ppr·PPR."""
    config = config or RetrievalConfig()
    top_k = top_k or config.top_k
    embedder = embedder or get_embedder(config)

    t0 = time.time()
    q_emb = np.asarray(embedder.encode([query]), dtype=np.float32).reshape(1, -1)

    # branch 1: text similarity
    text_n = normalize(score_text_similarity(q_emb, embeddings))

    # branch 2: mean triplet similarity per source passage
    triplet_n = normalize(score_triplet_similarity(q_emb, bundle, embeddings, passages))

    # branch 3: PPR on the unified k-hop subgraph
    ppr_scores, top_entities, top_entity_scores, sub = score_ppr_subgraph(
        q_emb, bundle, embeddings, config, top_k=top_k, depth=depth
    )
    ppr_n = normalize(ppr_scores)

    # final weighted score, over graph-connected passages only
    # (passages that appear in at least one triple)
    final: Dict[str, float] = {}
    for p in passages:
        if p not in bundle.passage_to_triples:
            continue
        final[p] = (config.w_text * text_n.get(p, 0.0)
                    + config.w_triplet * triplet_n.get(p, 0.0)
                    + config.w_ppr * ppr_n.get(p, 0.0))

    ranked = sorted(
        final,
        key=lambda p: (final[p], text_n.get(p, 0.0), triplet_n.get(p, 0.0), ppr_n.get(p, 0.0)),
        reverse=True,
    )[:top_k]

    breakdown = [{
        "passage_id": p,
        "text_sim": text_n.get(p, 0.0),
        "triplet_sim": triplet_n.get(p, 0.0),
        "ppr_score": ppr_n.get(p, 0.0),
        "final_score": final[p],
    } for p in ranked]

    records = []
    for p in ranked:
        record = dict(passages[p])
        record["passage_id"] = p
        records.append(record)

    # node attributes of the retrieved passages, straight from the graph
    graph_attrs = [dict(bundle.passage_node_attrs.get(p, {})) for p in ranked]

    return RetrievalResult(
        query=query,
        passage_ids=list(ranked),
        passage_texts=[passages[p]["text"] for p in ranked],
        passage_records=records,
        passage_graph_attrs=graph_attrs,
        score_breakdown=breakdown,
        top_entities=list(top_entities),
        top_entity_scores=top_entity_scores,
        subgraph_nodes=sub.number_of_nodes(),
        subgraph_edges=sub.number_of_edges(),
        elapsed_seconds=time.time() - t0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# High-level Retriever (lazy, reusable across queries)
# ─────────────────────────────────────────────────────────────────────────────


class Retriever:
    """Loads (or creates) indexes + embeddings once, then answers queries.

    On the first instantiation the indexes / embeddings are created from
    scratch and written to ``config.cache_dir``. Subsequent runs (new process
    included) load them from that cache instead of recomputing.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self._indexes: Optional[IndexBundle] = None
        self._passages: Optional[Dict[str, Dict[str, Any]]] = None
        self._embeddings: Optional[EmbeddingBundle] = None
        self._embedder = None

    # -- lazy loading -------------------------------------------------------
    @property
    def indexes(self) -> IndexBundle:
        if self._indexes is None:
            self._indexes, self._passages = build_indexes(self.config)
        return self._indexes

    @property
    def passages(self) -> Dict[str, Dict[str, Any]]:
        if self._passages is None:
            self._indexes, self._passages = build_indexes(self.config)
        return self._passages  # type: ignore[return-value]

    @property
    def embeddings(self) -> EmbeddingBundle:
        if self._embeddings is None:
            self._embeddings = build_embeddings(self.indexes, self.passages, self.config)
        return self._embeddings

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder(self.config)
        return self._embedder

    def warm_up(self) -> "Retriever":
        """Force indexes + embeddings (+ encoder) to be loaded/created."""
        _ = self.embeddings
        _ = self.embedder
        return self

    # -- querying -----------------------------------------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None,
                 depth: Optional[int] = None) -> RetrievalResult:
        return retrieve_passages_ppr(
            query,
            bundle=self.indexes,
            embeddings=self.embeddings,
            passages=self.passages,
            config=self.config,
            embedder=self.embedder,
            top_k=top_k,
            depth=depth,
        )

    def retrieve_many(self, queries: Iterable[str], top_k: Optional[int] = None,
                      depth: Optional[int] = None) -> List[RetrievalResult]:
        return [self.retrieve(q, top_k=top_k, depth=depth) for q in queries]

    # -- metadata -----------------------------------------------------------
    def get_metadata(self, passage_ids: Sequence[str],
                     keys: Optional[Sequence[str]] = None,
                     include_rank: bool = True) -> List[Dict[str, Any]]:
        """Metadata for arbitrary passage ids (loads only `passages.json`).

        No embeddings, encoder or GPU are needed.
        """
        return get_passage_metadata(passage_ids, self.passages, keys=keys,
                                    include_rank=include_rank)

    def get_metadata_index(self) -> Dict[str, Dict[str, Any]]:
        """The full ``{passage_id: metadata}`` mapping of the corpus."""
        return {pid: dict(rec.get("metadata") or {}) for pid, rec in self.passages.items()}

    def retrieve_metadata(self, query: str, top_k: Optional[int] = None,
                          depth: Optional[int] = None,
                          keys: Optional[Sequence[str]] = None,
                          include_scores: bool = True) -> List[Dict[str, Any]]:
        """Run a query and return **only** the metadata of the retrieved passages."""
        result = self.retrieve(query, top_k=top_k, depth=depth)
        return result.get_metadata(keys=keys, include_scores=include_scores)

    # -- graph metadata (node attributes of the aggregated graph) -----------
    def graph_metadata_index(self) -> Dict[str, Dict[str, Any]]:
        """``{passage_id: metadata}`` for every passage **node of the graph**."""
        return graph_passage_metadata_index(self.indexes)

    def get_graph_metadata(self, passage_ids: Sequence[str],
                           keys: Optional[Sequence[str]] = None,
                           include_rank: bool = True,
                           include_graph_attrs: bool = False) -> List[Dict[str, Any]]:
        """Metadata of arbitrary passage ids, read from the graph's passage nodes.

        Needs the indexes only (no embeddings, encoder or GPU).
        """
        return get_graph_passage_metadata(passage_ids, self.indexes, keys=keys,
                                          include_rank=include_rank,
                                          include_graph_attrs=include_graph_attrs)

    def retrieve_graph_metadata(self, query: str, top_k: Optional[int] = None,
                                depth: Optional[int] = None,
                                keys: Optional[Sequence[str]] = None,
                                include_scores: bool = True,
                                include_graph_attrs: bool = False) -> List[Dict[str, Any]]:
        """Run a query and return the graph-node metadata of the retrieved passages."""
        result = self.retrieve(query, top_k=top_k, depth=depth)
        return result.get_graph_metadata(keys=keys, include_scores=include_scores,
                                         include_graph_attrs=include_graph_attrs)
