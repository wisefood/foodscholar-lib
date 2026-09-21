"""Retrieval execution entry point for KGGEN_extended.

Takes a **user query** and runs the hybrid retrieval pipeline implemented in
`retrieval_core.py`, printing the retrieved source passages with their score
breakdown.

The indexes and the embeddings are loaded from `cache/` when available; the
first run creates them and writes them to disk so later queries (and later
processes) reuse them.

Usage
-----
    python run_retrieval.py "What is the most common mineral in the body?"

    python run_retrieval.py "What health condition does vitamin D deficiency lead to in children?" \
        --top-k 5 --gpu 1 --save-json out.json

    # only the metadata of the retrieved passages:
    python run_retrieval.py "What is the most common mineral in the body?" \
        --metadata-only --metadata-keys file,title,page_number --metadata-json meta.json

    # ... read from the passage nodes of the aggregated graph instead:
    python run_retrieval.py "What is the most common mineral in the body?" \
        --metadata-only --metadata-source graph

Programmatic use
----------------
    from run_retrieval import run_retrieval, run_retrieval_metadata, run_retrieval_graph_metadata

    result = run_retrieval("What is the most common mineral in the body?")
    print(result.passage_texts)

    # metadata-only view of the same retrieval (from passages.json)
    metadata = run_retrieval_metadata("What is the most common mineral in the body?")
    print(metadata[0]["file"], metadata[0]["page_number"])

    # ... or from the passage nodes of the aggregated graph
    graph_metadata = run_retrieval_graph_metadata("What is the most common mineral in the body?")
    print(graph_metadata[0])
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval_core import (  # noqa: E402
    RetrievalConfig,
    RetrievalResult,
    Retriever,
    _configure_gpu,
)

# One shared retriever per process: indexes / embeddings are loaded only once,
# no matter how many queries are run.
_RETRIEVER: Optional[Retriever] = None
# Config the cached `_RETRIEVER` above was built with. Kept so that a caller
# asking for a *different* retriever (other graph / passages / cache / device)
# is detected and served a matching one instead of the stale cached instance.
_RETRIEVER_CONFIG: Optional[RetrievalConfig] = None


def _same_retriever_state(a: Optional[RetrievalConfig],
                          b: Optional[RetrievalConfig]) -> bool:
    """True if both configs describe the same retriever state.

    `verbose` only controls logging, so it is ignored: flipping it must not
    throw away the warm in-process retriever.
    """
    if a is None or b is None:
        return False
    left, right = asdict(a), asdict(b)
    left.pop("verbose", None)
    right.pop("verbose", None)
    return left == right


def get_retriever(config: Optional[RetrievalConfig] = None,
                  force_rebuild_indexes: bool = False,
                  force_rebuild_embeddings: bool = False) -> Retriever:
    """Return the process-wide `Retriever`, building (or loading) its state.

    The cached retriever is reused only when it was built with the same config
    as the one requested here; a different config (other graph/passages paths,
    cache dir, device, top_k, ...) rebuilds it. Without that check, the first
    call without a config — which therefore used the defaults — would keep being
    returned to every later caller that *did* pass a config.
    """
    global _RETRIEVER, _RETRIEVER_CONFIG

    config = config or RetrievalConfig()

    if (_RETRIEVER is None
            or force_rebuild_indexes
            or force_rebuild_embeddings
            or not _same_retriever_state(_RETRIEVER_CONFIG, config)):
        _configure_gpu(config)  # before torch / sentence_transformers import
        retriever = Retriever(config)
        if force_rebuild_indexes or force_rebuild_embeddings:
            from retrieval_core import build_embeddings, build_indexes
            retriever._indexes, retriever._passages = build_indexes(
                config, force_rebuild=force_rebuild_indexes
            )
            retriever._embeddings = build_embeddings(
                retriever._indexes, retriever._passages, config,
                force_rebuild=force_rebuild_embeddings,
            )
        _RETRIEVER = retriever
        _RETRIEVER_CONFIG = config
    return _RETRIEVER


def run_retrieval(query: str,
                  config: Optional[RetrievalConfig] = None,
                  top_k: Optional[int] = None,
                  depth: Optional[int] = None,
                  verbose: bool = True) -> RetrievalResult:
    """Run retrieval for a single user query and return the `RetrievalResult`."""
    if config is None:
        config = RetrievalConfig(verbose=verbose)
    else:
        config.verbose = verbose
    retriever = get_retriever(config)
    result = retriever.retrieve(query, top_k=top_k, depth=depth)
    if verbose:
        print_result(result)
    return result


def run_retrieval_metadata(query: str,
                           config: Optional[RetrievalConfig] = None,
                           top_k: Optional[int] = None,
                           depth: Optional[int] = None,
                           keys: Optional[Sequence[str]] = None,
                           include_scores: bool = True,
                           verbose: bool = True) -> List[Dict[str, Any]]:
    """Run retrieval for a user query and return **only** the retrieved metadata.

    Returns a list with one dict per retrieved passage (rank order) containing
    the passage id, its metadata fields and, optionally, the scores.
    """
    if config is None:
        config = RetrievalConfig(verbose=verbose)
    else:
        config.verbose = verbose
    retriever = get_retriever(config)
    metadata = retriever.retrieve_metadata(query, top_k=top_k, depth=depth,
                                           keys=keys, include_scores=include_scores)
    if verbose:
        print_metadata(metadata)
    return metadata


def run_retrieval_graph_metadata(query: str,
                                 config: Optional[RetrievalConfig] = None,
                                 top_k: Optional[int] = None,
                                 depth: Optional[int] = None,
                                 keys: Optional[Sequence[str]] = None,
                                 include_scores: bool = True,
                                 include_graph_attrs: bool = False,
                                 verbose: bool = True) -> List[Dict[str, Any]]:
    """Run retrieval and return the metadata stored on the graph's passage nodes.

    This reads the `metadata` attribute of the passage nodes of the aggregated
    graph (instead of the `passages.json` copy used by `run_retrieval_metadata`).
    """
    if config is None:
        config = RetrievalConfig(verbose=verbose)
    else:
        config.verbose = verbose
    retriever = get_retriever(config)
    metadata = retriever.retrieve_graph_metadata(
        query, top_k=top_k, depth=depth, keys=keys,
        include_scores=include_scores, include_graph_attrs=include_graph_attrs,
    )
    if verbose:
        print_metadata(metadata)
    return metadata


def print_metadata(metadata: List[Dict[str, Any]], max_col_width: int = 55) -> None:
    """Print a metadata list (as returned by `RetrievalResult.get_metadata`)."""
    if not metadata:
        print("No metadata.")
        return
    columns = list(metadata[0].keys())
    widths = {
        c: min(max_col_width, max(len(str(c)), *(len(str(e.get(c))) for e in metadata)))
        for c in columns
    }
    print(" | ".join(str(c).ljust(widths[c]) for c in columns))
    print("-+-".join("-" * widths[c] for c in columns))
    for entry in metadata:
        cells = []
        for c in columns:
            value = str(entry.get(c))
            if len(value) > widths[c]:
                value = value[: widths[c] - 1] + "…"
            cells.append(value.ljust(widths[c]))
        print(" | ".join(cells))


def print_result(result: RetrievalResult, max_chars: int = 500) -> None:
    """Pretty-print a retrieval result."""
    print("=" * 100)
    print(f"QUERY: {result.query}")
    print("=" * 100)
    print(f"Retrieved passages : {len(result.passage_ids)}")
    print(f"Top entities       : {result.top_entities}")
    print(f"Subgraph           : {result.subgraph_nodes} nodes / {result.subgraph_edges} edges")
    print(f"Time               : {result.elapsed_seconds:.2f}s")
    print("-" * 100)
    for rank, (pid, breakdown) in enumerate(zip(result.passage_ids, result.score_breakdown), 1):
        text = result.passage_records[rank - 1].get("text", "")
        print(f"[{rank}] {pid}")
        print(f"    final={breakdown['final_score']:.4f}  "
              f"(text={breakdown['text_sim']:.4f}, "
              f"triplet={breakdown['triplet_sim']:.4f}, "
              f"ppr={breakdown['ppr_score']:.4f})")
        print(f"    {text[:max_chars]}{'...' if len(text) > max_chars else ''}")
        print()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run KGGEN_extended hybrid passage retrieval for a user query."
    )
    parser.add_argument("query", type=str, help="the user query to retrieve passages for")
    parser.add_argument("--top-k", type=int, default=None,
                        help="number of passages to return (default: config value)")
    parser.add_argument("--depth", type=int, default=None,
                        help="k-hop depth of the subgraph (default: config value)")
    parser.add_argument("--gpu", type=int, default=1,
                        help="GPU index to use (-1 to disable CUDA_VISIBLE_DEVICES pinning)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="device for the encoder (default: auto)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="embedding batch size used when creating the embeddings")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="directory for the index / embedding caches")
    parser.add_argument("--rebuild-indexes", action="store_true",
                        help="ignore the cached indexes and rebuild them")
    parser.add_argument("--rebuild-embeddings", action="store_true",
                        help="ignore the cached embeddings and recompute them")
    parser.add_argument("--save-json", type=str, default=None,
                        help="optional path where the result is written as JSON")
    parser.add_argument("--metadata-only", action="store_true",
                        help="print only the metadata of the retrieved passages")
    parser.add_argument("--metadata-json", type=str, default=None,
                        help="optional path where the retrieved passage metadata is written as JSON")
    parser.add_argument("--metadata-keys", type=str, default=None,
                        help="comma-separated subset of metadata fields to keep, "
                             "e.g. 'file,title,page_number'")
    parser.add_argument("--metadata-source", type=str, default="passages",
                        choices=["passages", "graph"],
                        help="where to read the metadata from: 'passages' (passages.json) "
                             "or 'graph' (the passage-node attributes of the aggregated graph)")
    parser.add_argument("--no-scores", action="store_true",
                        help="exclude the scores from the metadata output")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = parser.parse_args(argv)

    config = RetrievalConfig(
        gpu_index=None if args.gpu < 0 else args.gpu,
        device=args.device,
        verbose=not args.quiet,
    )
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.cache_dir:
        config.cache_dir = Path(args.cache_dir)

    if config.verbose:
        print(config.describe())
        print("=" * 100)

    retriever = get_retriever(
        config,
        force_rebuild_indexes=args.rebuild_indexes,
        force_rebuild_embeddings=args.rebuild_embeddings,
    )
    result = retriever.retrieve(args.query, top_k=args.top_k, depth=args.depth)

    metadata_keys = None
    if args.metadata_keys:
        metadata_keys = tuple(k.strip() for k in args.metadata_keys.split(",") if k.strip())

    metadata = None
    if args.metadata_only or args.metadata_json:
        if args.metadata_source == "graph":
            metadata = result.get_graph_metadata(keys=metadata_keys,
                                                 include_scores=not args.no_scores)
        else:
            metadata = result.get_metadata(keys=metadata_keys,
                                           include_scores=not args.no_scores)

    if not args.quiet:
        if args.metadata_only:
            print_metadata(metadata or [])
        else:
            print_result(result)

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as fh:
            fh.write(result.to_json())
        print(f"Result saved to {args.save_json}")

    if args.metadata_json and metadata is not None:
        Path(args.metadata_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metadata_json, "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        print(f"Metadata saved to {args.metadata_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
