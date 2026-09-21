#!/usr/bin/env python3
"""
Extract KG triples from a single chunks CSV using KGGEN Extended.

Usage:
    python extract_triplets_from_chunks.py \\
        --csv /path/to/chunks.csv \\
        --out-dir /path/to/output \\
        [--model "mistral-small3.2:24b-instruct-2506-q8_0"] \\
        [--max-texts 5]

Pipeline (per CSV):
  1. Load the chunks CSV
  2. Generate a separate ExtendedGraph per chunk (passage_id = chunk UUID)
  3. Aggregate all chunk graphs → one combined ExtendedGraph
  4. Deduplicate entities (semhash, passage-aware remapping)
  5. Export GraphML + triples CSV/JSONL + passages CSV/JSON + metrics.json
     + aggregated_graph.json
"""

import argparse
import ast
import json
import time
import sys
import pandas as pd
from pathlib import Path

GRAPH_CODE_DIR = Path(__file__).resolve().parent.parent
if str(GRAPH_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_CODE_DIR))

# ── HuggingFace cache setup (must precede kggen_extended import) ──
import os
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
    {"name": "mistral-small-3.2-24b-instruct-2506",          "provider": "gpustack",     "input_cost_per_1M": 0.0,   "output_cost_per_1M": 0.0},
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

# ── Column names in input CSV ────────────────────────────────────
TEXT_COL = "chunk_text"
ID_COL = "chunk_id"
META_COL = "chunk_metadata"
MAX_TEXTS = None                    # set to None for all texts, or a number for testing
KG_CHUNK_SIZE = 5000


# ── Token-tracking wrapper ────────────────────────────────────────

class UsageTracker:
    """Accumulates prompt / completion token counts via LiteLLM callbacks."""
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def track_callback(self, kwargs, completion_response, start_time, end_time):
        usage = None
        if hasattr(completion_response, 'usage') and completion_response.usage:
            usage = completion_response.usage
        elif isinstance(completion_response, dict):
            usage = completion_response.get('usage')
        if usage:
            self.prompt_tokens += getattr(usage, 'prompt_tokens', 0) or usage.get('prompt_tokens', 0)
            self.completion_tokens += getattr(usage, 'completion_tokens', 0) or usage.get('completion_tokens', 0)


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


def _print_metrics(model_info, elapsed_s, tracker, output_dir):
    """Print and return a metrics dict, and save to output_dir/metrics.json."""
    cost_input = (tracker.prompt_tokens / 1_000_000) * model_info["input_cost_per_1M"]
    cost_output = (tracker.completion_tokens / 1_000_000) * model_info["output_cost_per_1M"]
    total_cost = cost_input + cost_output

    metrics = {
        "model": model_info["name"],
        "provider": model_info.get("provider", "openrouter"),
        "elapsed_s": round(elapsed_s, 1),
        "prompt_tokens": tracker.prompt_tokens,
        "completion_tokens": tracker.completion_tokens,
        "total_tokens": tracker.total_tokens,
        "cost_input_usd": round(cost_input, 6),
        "cost_output_usd": round(cost_output, 6),
        "cost_total_usd": round(total_cost, 6),
    }

    print("\n─ Metrics ─" + "─" * 50)
    print(f"  Wall time          : {elapsed_s:.1f}s")
    print(f"  Prompt tokens      : {tracker.prompt_tokens:,}")
    print(f"  Completion tokens  : {tracker.completion_tokens:,}")
    print(f"  Total tokens       : {tracker.total_tokens:,}")
    print(f"  Cost (input)       : ${cost_input:.6f}")
    print(f"  Cost (output)      : ${cost_output:.6f}")
    print(f"  Cost (total)       : ${total_cost:.6f}")
    print("─" * 60)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics


def _parse_metadata(meta_str):
    """Parse chunk_metadata string to a dict."""
    if pd.isna(meta_str) or not meta_str:
        return {}
    try:
        return ast.literal_eval(meta_str)
    except (ValueError, SyntaxError):
        try:
            return json.loads(meta_str)
        except (json.JSONDecodeError, TypeError):
            return {"raw": str(meta_str)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract KG triples from a single chunks CSV using KGGEN Extended."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to a single chunks CSV file.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for this CSV's KG results.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL}). Use --list-models to see all choices.",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=MAX_TEXTS,
        help="Maximum number of texts to process (for testing). Default: all.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available models with pricing and exit.",
    )
    return parser.parse_args()


def run_extraction(csv_path, out_dir, model_info):
    """Run the full KGGEN Extended pipeline on a single CSV."""
    runtime = _resolve_kggen_runtime(model_info)
    model_name = runtime["model_name"]

    # ── Set provider-specific env vars for LiteLLM ────────────────
    if runtime["provider"] == "openrouter":
        os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
        os.environ["OPENROUTER_API_BASE"] = OPENROUTER_API_BASE
    elif runtime["provider"] == "gpustack":
        os.environ["GPUSTACK_API_KEY"] = GPUSTACK_API_KEY
        os.environ["GPUSTACK_API_BASE"] = GPUSTACK_API_BASE
    else:
        os.environ["OLLAMA_API_BASE"] = OLLAMA_API_BASE
        os.environ["OLLAMA_API_KEY"] = OLLAMA_API_KEY

    # ── Wire up token tracking ───────────────────────────────────
    tracker = UsageTracker()
    litellm.success_callback = [tracker.track_callback]

    t_start = time.time()

    # ── Load CSV ──────────────────────────────────────────────────
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return

    print(f"\nLoading chunks from {csv_path} ...")
    df = pd.read_csv(csv_path)

    missing = [c for c in [ID_COL, TEXT_COL] if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in CSV: {missing}")
        return

    print(f"  {len(df)} chunks loaded.\n")

    # ── Create output dir ─────────────────────────────────────────
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"KGGEN Extended — Passage-Aware KG Extraction (Single CSV)")
    print("=" * 70)
    print(f"CSV        : {csv_path}")
    print(f"Model      : {model_name}")
    print(f"Provider   : {runtime['provider']}")
    print(f"Chunks     : {len(df)}")
    print(f"API base   : {runtime['api_base']}")
    print(f"Out dir    : {out_dir}")
    print("=" * 70)

    # ── Init ExtendedKGGen ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1/5 — Initialise ExtendedKGGen")
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
    print(f"  LiteLLM: {runtime['litellm_model']}")
    print(f"  API base: {runtime['api_base']}\n")

    # ── Generate per-chunk graphs with passage tracking ───────────
    print("=" * 60)
    print("STEP 2/5 — Generate per-chunk graphs (with passage tracking)")
    print("=" * 60)

    # ── Checkpoint directory for incremental saves ────────────────
    chunks_dir = out_dir / "chunk_graphs"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Resume: load already-processed chunk IDs from the checkpoint dir
    processed_ids = set()
    if chunks_dir.exists():
        for f in chunks_dir.glob("*.json"):
            processed_ids.add(f.stem)
    if processed_ids:
        print(f"  Found {len(processed_ids)} already-processed chunks in {chunks_dir}")
        print(f"  Will skip them and resume from the last processed chunk.\n")

    chunk_graphs = []
    t_gen = time.time()
    skipped = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        chunk_id = str(row[ID_COL])
        chunk_text = str(row[TEXT_COL]).strip()
        if not chunk_text:
            continue

        # ── Checkpoint: skip if already saved ────────────────────
        checkpoint_path = chunks_dir / f"{chunk_id}.json"
        if chunk_id in processed_ids:
            try:
                graph = ExtendedGraph.from_file(str(checkpoint_path))
                chunk_graphs.append(graph)
                skipped += 1
                continue
            except Exception as e:
                print(f"    ⚠️  Corrupt checkpoint {chunk_id[:8]}..., re-extracting: {e}")

        # Parse metadata
        chunk_metadata = {}
        if META_COL in row.index:
            chunk_metadata = _parse_metadata(row[META_COL])

        print(f"  Chunk {idx}: {chunk_id[:8]}... ({len(chunk_text)} chars) → generating ...", flush=True)

        t1 = time.time()
        try:
            graph = kg.generate(
                input_data=chunk_text,
                chunk_size=KG_CHUNK_SIZE,
                passage_id=chunk_id,
                passage_text=chunk_text,
                passage_metadata=chunk_metadata,
            )
        except Exception as e:
            print(f"    ⚠️  Chunk {idx} failed: {e}")
            continue

        elapsed_chunk = time.time() - t1

        if hasattr(graph, "entities"):
            graph.entity_metadata = {
                entity: {chunk_id}
                for entity in graph.entities
            }

            # ── SAVE CHECKPOINT immediately ──────────────────────
            graph.to_file(str(checkpoint_path))

            print(f"    ✓ {len(graph.relations)} relations ({elapsed_chunk:.1f}s)" +
                  f" [saved {chunk_id[:8]}...]")
            chunk_graphs.append(graph)
        else:
            print(f"    ⚠️  Unexpected graph type: {type(graph)}")

    gen_elapsed = time.time() - t_gen
    if skipped:
        print(f"\n  → {skipped} chunk(s) loaded from savepoints.")
    print(f"  → {len(chunk_graphs)} / {len(df)} chunks successful in {gen_elapsed:.1f}s\n")

    if not chunk_graphs:
        print("ERROR: No chunk graphs were generated. Exiting.")
        return

    # ── Aggregate all chunk graphs ────────────────────────────────
    print("=" * 60)
    print("STEP 3/5 — Aggregate all chunk graphs (passage-aware)")
    print("=" * 60)
    combined_graph = kg.aggregate(chunk_graphs)
    combined_graph.stats("Combined Graph")
    print("  Sample relations (first 5):")
    for i, rel in enumerate(list(combined_graph.relations)[:5]):
        tkey = _triple_to_key(rel)
        pids = combined_graph.triplet_passages.get(tkey, set()) if combined_graph.triplet_passages else set()
        print(f"    {rel[0]} → [{rel[1]}] → {rel[2]}  [passages: {len(pids)}]")
    print()

    # ── Deduplicate aggregated graph ──────────────────────────────
    print("=" * 60)
    print("STEP 4/5 — Deduplicate (passage-aware semhash)")
    print("=" * 60)
    print(f"  Before — Entities: {len(combined_graph.entities)}, "
          f"Relations: {len(combined_graph.relations)}")

    deduped_graph = kg.deduplicate(
        combined_graph,
        method=DeduplicateMethod.SEMHASH,
        semhash_similarity_threshold=0.95,
    )

    deduped_graph.stats("Deduped Graph")
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
    print("STEP 5/5 — Export all formats")
    print("=" * 60)
    export_all(deduped_graph, str(out_dir))

    # ── Save full graph JSON (for later reload/aggregation) ──────
    graph_json_path = out_dir / "aggregated_graph.json"
    deduped_graph.to_file(str(graph_json_path))
    print(f"  JSON:   {graph_json_path}")

    # ── Metrics ───────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    _print_metrics(model_info, total_elapsed, tracker, out_dir)

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

    run_extraction(args.csv, args.out_dir, model_info)


if __name__ == "__main__":
    main()
