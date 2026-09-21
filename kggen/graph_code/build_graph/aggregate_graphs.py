#!/usr/bin/env python3
"""
Aggregate multiple ExtendedGraph JSON files into a single combined graph.

This script loads graphs from JSON (produced by extract_triplets_from_chunks.py or
extract_triplets_aggregate_extended.py), aggregates them into one combined graph,
deduplicates, and exports.

Usage:
    # Aggregate two or more JSON graph files
    python aggregate_graphs.py \\
        --graphs graph1.json graph2.json [...] \\
        --out-dir /path/to/aggregated_output \\
        [--model "mistral-small3.2:24b-instruct-2506-q8_0"]

    # Aggregate all aggregated_graph.json from subdirectories
    python aggregate_graphs.py \\
        --graphs-dir /path/to/kg_output_chunks/guides/ \\
        --out-dir /path/to/aggregated_guides \\
        [--model "mistral-small3.2:24b-instruct-2506-q8_0"]

Input formats accepted:
  - Individual .json files (ExtendedGraph serialization)
  - A directory with --graphs-dir: finds all **/aggregated_graph.json recursively

Output:
  - aggregated_graph.json   (full ExtendedGraph)
  - aggregated_graph.graphml (GraphML with passage nodes + Source edges)
  - triples.csv / triples.jsonl
  - passages.csv / passages.json
  - metrics.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Make the local `kggen_extended` package importable ──
# It lives in graph_code/ — the parent of this script's directory — and is not
# pip-installed, so graph_code/ must be on sys.path before importing it.
GRAPH_CODE_DIR = Path(__file__).resolve().parent.parent
if str(GRAPH_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_CODE_DIR))

# ── HuggingFace cache setup (must precede kggen_extended import) ──
hf_cache = Path("/mnt/data/huggingface_cache")
hf_cache.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(hf_cache)
os.environ["HF_HUB_CACHE"] = str(hf_cache / "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(hf_cache / "sentence-transformers")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from kggen_extended import ExtendedKGGen, ExtendedGraph, DeduplicateMethod, export_all
from kggen_extended.models import _triple_to_key, _key_to_triple
import litellm
litellm.disable_cache = True

# ── Configuration ─────────────────────────────────────────────────
MODEL_CATALOG = [
    {"name": "llama3.1:8b-instruct-fp16",                   "provider": "ollama",     "input_cost_per_1M": 0.0,   "output_cost_per_1M": 0.0},
    {"name": "mistral-small3.2:24b-instruct-2506-q8_0",     "provider": "ollama",     "input_cost_per_1M": 0.0,   "output_cost_per_1M": 0.0},
    {"name": "mistral-small-3.2-24b-instruct-2506",         "provider": "gpustack",   "input_cost_per_1M": 0.0,   "output_cost_per_1M": 0.0},
    {"name": "gpt-oss:20b",                                 "provider": "ollama",     "input_cost_per_1M": 0.0,   "output_cost_per_1M": 0.0},
]

MODEL_BY_NAME = {m["name"]: m for m in MODEL_CATALOG}
DEFAULT_MODEL = "mistral-small3.2:24b-instruct-2506-q8_0"

# ── API configuration ─────────────────────────────────────────────
OPENROUTER_API_BASE = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "None")
GPUSTACK_API_BASE = os.environ.get("GPUSTACK_API_BASE", "http://test6.magellan2.imsi.athenarc.gr/v1")
GPUSTACK_API_KEY = os.environ.get("GPUSTACK_API_KEY", "")  # was a hardcoded key; rotate it


def _resolve_kggen_runtime(model_info):
    """Resolve provider-specific model and endpoint settings."""
    provider = model_info.get("provider", "openrouter")
    model_name = model_info["name"]

    if provider == "ollama":
        return {
            "provider": provider,
            "model_name": model_name,
            "litellm_model": f"ollama_chat/{model_name}",
            "api_base": OLLAMA_API_BASE,
            "api_key": OLLAMA_API_KEY,
        }

    if provider == "gpustack":
        return {
            "provider": provider,
            "model_name": model_name,
            "litellm_model": f"openai/{model_name}",
            "api_base": GPUSTACK_API_BASE,
            "api_key": GPUSTACK_API_KEY,

        }

    return {
        "provider": provider,
        "model_name": model_name,
        "litellm_model": f"openrouter/{model_name}",
        "api_base": OPENROUTER_API_BASE,
        "api_key": OPENROUTER_API_KEY,
    }


def load_graphs_from_paths(graph_paths):
    """Load ExtendedGraph objects from a list of JSON file paths.

    Returns:
        list of ExtendedGraph, list of (path, error) for failed loads.
    """
    graphs = []
    errors = []

    for gp in graph_paths:
        gp = Path(gp)
        if not gp.exists():
            errors.append((str(gp), "File not found"))
            continue
        try:
            graph = ExtendedGraph.from_file(str(gp))
            graphs.append(graph)
            print(f"  ✓ Loaded: {gp.name}  ({len(graph.entities)} entities, {len(graph.relations)} relations)")
        except Exception as e:
            errors.append((str(gp), str(e)))
            print(f"  ✗ Failed: {gp.name} — {e}")

    return graphs, errors


def find_aggregated_graphs(base_dir, pattern="aggregated_graph.json"):
    """Recursively find all files matching `pattern` under `base_dir`.

    Skips directories named _aggregated_* to avoid re-aggregating previous outputs.
    """
    base = Path(base_dir)
    if not base.is_dir():
        print(f"ERROR: Not a directory: {base}")
        return []

    paths = []
    for p in base.rglob(pattern):
        # Skip previously aggregated outputs
        if any(parent.name.startswith("_aggregated") for parent in p.parents):
            continue
        paths.append(p)

    paths.sort()
    return paths


class UsageTracker:
    """Minimal token-tracking wrapper (token counts are zero for aggregation-only runs)."""
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def track_callback(self, kwargs, completion_response, start_time, end_time):
        pass


def _print_metrics(model_info, elapsed_s, tracker, output_dir, n_input_graphs):
    """Print and save metrics."""
    cost_input = (tracker.prompt_tokens / 1_000_000) * model_info["input_cost_per_1M"]
    cost_output = (tracker.completion_tokens / 1_000_000) * model_info["output_cost_per_1M"]
    total_cost = cost_input + cost_output

    metrics = {
        "model": model_info["name"],
        "provider": model_info.get("provider", "openrouter"),
        "elapsed_s": round(elapsed_s, 1),
        "n_input_graphs": n_input_graphs,
        "prompt_tokens": tracker.prompt_tokens,
        "completion_tokens": tracker.completion_tokens,
        "total_tokens": tracker.total_tokens,
        "cost_input_usd": round(cost_input, 6),
        "cost_output_usd": round(cost_output, 6),
        "cost_total_usd": round(total_cost, 6),
    }

    print("\n─ Metrics ─" + "─" * 50)
    print(f"  Input graphs       : {n_input_graphs}")
    print(f"  Wall time          : {elapsed_s:.1f}s")
    print(f"  Total tokens       : {tracker.total_tokens:,}")
    print(f"  Cost (total)       : ${total_cost:.6f}")
    print("─" * 60)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple ExtendedGraph JSON files into a single combined graph."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--graphs",
        nargs="+",
        help="One or more ExtendedGraph JSON files to aggregate.",
    )
    group.add_argument(
        "--graphs-dir",
        type=str,
        help="Directory to search recursively for aggregated_graph.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for the aggregated graph and exports.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model for deduplication embeddings (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available models with pricing and exit.",
    )
    return parser.parse_args()


def run_aggregation(graph_paths, out_dir, model_info):
    """Load graphs from paths, aggregate, deduplicate, and export."""
    runtime = _resolve_kggen_runtime(model_info)
    model_name = runtime["model_name"]

    if runtime["provider"] == "openrouter":
        os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
        os.environ["OPENROUTER_API_BASE"] = OPENROUTER_API_BASE
    elif runtime["provider"] == "gpustack":
        os.environ["GPUSTACK_API_KEY"] = GPUSTACK_API_KEY
        os.environ["GPUSTACK_API_BASE"] = GPUSTACK_API_BASE
    else:
        os.environ["OLLAMA_API_BASE"] = OLLAMA_API_BASE
        os.environ["OLLAMA_API_KEY"] = OLLAMA_API_KEY

    tracker = UsageTracker()
    litellm.success_callback = [tracker.track_callback]

    t_start = time.time()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"Graph Aggregation — Combine Multiple ExtendedGraphs")
    print("=" * 70)
    print(f"Model      : {model_name}")
    print(f"Provider   : {runtime['provider']}")
    print(f"Input files: {len(graph_paths)}")
    print(f"Out dir    : {out_dir}")
    print("=" * 70)

    # ── Load all graphs ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1/4 — Load input graphs")
    print("=" * 60)
    graphs, errors = load_graphs_from_paths(graph_paths)

    if errors:
        print(f"\n  {len(errors)} file(s) failed to load:")
        for path, err in errors:
            print(f"    - {path}: {err}")

    if not graphs:
        print("ERROR: No graphs loaded. Aborting.")
        return

    total_entities = sum(len(g.entities) for g in graphs)
    total_relations = sum(len(g.relations) for g in graphs)
    print(f"\n  Total across {len(graphs)} graphs: {total_entities} entities, {total_relations} relations\n")

    # ── Init ExtendedKGGen (needed for aggregate() and deduplicate()) ──
    print("=" * 60)
    print("STEP 2/4 — Initialise ExtendedKGGen for aggregation")
    print("=" * 60)
    if runtime["provider"] in ["openrouter", "gpustack"]:
        kg = ExtendedKGGen(
                model=runtime["litellm_model"],
                max_tokens=8192,
                api_base=runtime["api_base"],
                api_key=runtime["api_key"],
                temperature=0.0,
                disable_cache=True,
            )
    else:
        kg = ExtendedKGGen(
                model=runtime["litellm_model"],
                max_tokens=8192,
                api_base=runtime["api_base"],
                temperature=0.0,
                disable_cache=True,
            )
    print(f"  Model: {model_name}")
    print(f"  LiteLLM: {runtime['litellm_model']}\n")

    # ── Aggregate all graphs ──────────────────────────────────────
    print("=" * 60)
    print("STEP 3/4 — Aggregate all graphs (passage-aware)")
    print("=" * 60)
    combined_graph = kg.aggregate(graphs)
    combined_graph.stats("Combined Aggregated Graph")
    print("  Sample relations (first 5):")
    for i, rel in enumerate(list(combined_graph.relations)[:5]):
        tkey = _triple_to_key(rel)
        pids = combined_graph.triplet_passages.get(tkey, set()) if combined_graph.triplet_passages else set()
        print(f"    {rel[0]} → [{rel[1]}] → {rel[2]}  [passages: {len(pids)}]")
    print()

    # ── Deduplicate ───────────────────────────────────────────────
    print("=" * 60)
    print("STEP 4/4 — Deduplicate (passage-aware semhash)")
    print("=" * 60)
    print(f"  Before — Entities: {len(combined_graph.entities)}, "
          f"Relations: {len(combined_graph.relations)}")

    deduped_graph = kg.deduplicate(
        combined_graph,
        method=DeduplicateMethod.SEMHASH,
        semhash_similarity_threshold=0.95,
    )

    deduped_graph.stats("Deduped Aggregated Graph")
    if deduped_graph.entity_clusters:
        n_clusters = sum(1 for v in deduped_graph.entity_clusters.values() if len(v) > 0)
        print(f"  Entity clusters with aliases: {n_clusters}")
        for canonical, aliases in list(deduped_graph.entity_clusters.items())[:5]:
            if aliases:
                print(f"    {canonical} ← {aliases}")
    print()

    # ── Show multi-passage triplets ───────────────────────────────
    if deduped_graph.triplet_passages:
        multi = {
            tkey: pids
            for tkey, pids in deduped_graph.triplet_passages.items()
            if len(pids) > 1
        }
        print(f"  Triplets with >1 passage: {len(multi)}")
        for tkey, pids in list(multi.items())[:5]:
            s, p, o = _key_to_triple(tkey)
            print(f"    {s} → [{p}] → {o}  [passages: {len(pids)}]")

    # ── Export all formats ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Export all formats")
    print("=" * 60)
    export_all(deduped_graph, str(out_dir))

    # ── Save full graph JSON ──────────────────────────────────────
    graph_json_path = out_dir / "aggregated_graph.json"
    deduped_graph.to_file(str(graph_json_path))
    print(f"  JSON:   {graph_json_path}")

    # ── Metrics ───────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    _print_metrics(model_info, total_elapsed, tracker, out_dir, len(graphs))

    print(f"\n{'=' * 60}")
    print(f"Done in {total_elapsed:.1f}s")
    print(f"{'=' * 60}")


def main():
    args = parse_args()

    if args.list_models:
        print("Available models (OpenRouter and Ollama):\n")
        print(f"{'Model':<48} {'Provider':<12} {'Input/M tok':>12} {'Output/M tok':>13}")
        print("-" * 89)
        for m in MODEL_CATALOG:
            print(f"  {m['name']:<46} {m.get('provider', 'openrouter'):<12} "
                  f"${m['input_cost_per_1M']:>10.2f}  ${m['output_cost_per_1M']:>11.2f}")
        print(f"\nDefault: {DEFAULT_MODEL}")
        return

    model_name = args.model
    if model_name not in MODEL_BY_NAME:
        print(f"Unknown model '{model_name}'.")
        print(f"Available models: {[m['name'] for m in MODEL_CATALOG]}")
        print("Use --list-models to see details.")
        return
    model_info = MODEL_BY_NAME[model_name]

    # ── Resolve graph paths ───────────────────────────────────────
    if args.graphs:
        graph_paths = [str(Path(p).resolve()) for p in args.graphs]
    else:
        # --graphs-dir
        graph_paths = find_aggregated_graphs(args.graphs_dir)
        if not graph_paths:
            print(f"No aggregated_graph.json files found under: {args.graphs_dir}")
            return
        graph_paths = [str(p) for p in graph_paths]

    run_aggregation(graph_paths, args.out_dir, model_info)


if __name__ == "__main__":
    main()
