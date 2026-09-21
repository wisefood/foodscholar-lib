"""
scripts/run_scifoodner_inference.py
------------------------------------
SciFoodNER inference script — run with scifoodner conda environment.
Called as a subprocess from notebooks/nel_ner_evaluation.ipynb.

Usage:
    /mnt/data/makis/conda_envs/scifoodner/bin/python \
        scripts/run_scifoodner_inference.py \
        --input  notebook_artifacts/scifoodner_input.json \
        --output notebook_artifacts/scifoodner_results.json

Input JSON:
    {"passage": "...", "ground_truth": ["entity1", "entity2", ...]}

Output JSON:
    {
      "variant_F": {NER results from cafeteria model},
      "variant_G": {NER+NEL results from foodon model}
    }
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCIFOODNER_BASE = "/mnt/data/vpitsilou/scifoodner"
NER_DATA_BASE   = "/mnt/data/vpitsilou/stelar/SciFoodNER/scifoodner_final"


def get_dataset_labels(dataset: str) -> list:
    import pandas as pd
    if dataset == "cafeteria":
        df = pd.read_csv(f"{NER_DATA_BASE}/NER_data/cafeteria.tsv", sep="\t").dropna()
    else:
        df = pd.read_csv(f"{NER_DATA_BASE}/NER_data/{dataset}.csv", index_col=[0])
    df.columns = ["word", "tag"]
    df["word"] = df["word"].astype(str)
    df["tag"]  = df["tag"].astype(str)
    return list(df["tag"].drop_duplicates().values)


def split_tag(tag: str):
    tag = tag.strip()
    if tag in ("O", "B", "I"):
        return tag, None
    if "-" in tag:
        prefix, value = tag.split("-", 1)
        return prefix, value
    return tag, None


def parse_predictions(sentence_predictions: list) -> list:
    entities, current_tokens, current_label = [], [], None

    def flush():
        nonlocal current_tokens, current_label
        if current_tokens:
            links = [x.strip() for x in current_label.split(";") if x.strip()] if current_label else []
            entities.append({"text": " ".join(current_tokens), "links": links})
        current_tokens, current_label = [], None

    for item in sentence_predictions:
        token, tag = list(item.items())[0]
        token = token.strip()
        prefix, value = split_tag(tag)
        if prefix == "O":
            flush()
        elif prefix == "B":
            flush(); current_tokens = [token]; current_label = value
        elif prefix == "I":
            if value is not None:
                if current_tokens and current_label == value:
                    current_tokens.append(token)
                else:
                    flush(); current_tokens = [token]; current_label = value
            else:
                if current_tokens: current_tokens.append(token)
                else: current_tokens = [token]; current_label = None
        else:
            flush()
    flush()
    return entities


def evaluate_ner(extracted: set, ground_truth: set) -> dict:
    ext_lower = {s.lower() for s in extracted}
    gt_lower  = {s.lower() for s in ground_truth}
    tp = len(ext_lower & gt_lower)
    fp = len(ext_lower - gt_lower)
    fn = len(gt_lower - ext_lower)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    jaccard   = len(ext_lower & gt_lower) / len(ext_lower | gt_lower) if (ext_lower | gt_lower) else 1.0
    return {"extracted": len(ext_lower), "precision": precision,
            "recall": recall, "f1": f1, "jaccard": jaccard,
            "tp": tp, "fp": fp, "fn": fn}


def run_model(dataset: str, passage: str) -> tuple:
    from simpletransformers.ner import NERModel, NERArgs
    import torch

    labels    = get_dataset_labels(dataset)
    model_dir = f"{SCIFOODNER_BASE}/{dataset}/best"
    args      = NERArgs()
    args.use_multiprocessing = False

    use_cuda = torch.cuda.is_available()
    print(f"  Loading {dataset} model (CUDA={use_cuda})...", file=sys.stderr)
    model = NERModel("bert", model_dir, args=args, labels=labels, use_cuda=use_cuda)

    t0 = time.time()
    predictions, _ = model.predict([passage])
    runtime = time.time() - t0
    print(f"  {dataset} inference: {runtime:.2f}s", file=sys.stderr)

    entities = parse_predictions(predictions[0])
    return entities, runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    passage      = data["passage"]
    ground_truth = set(data["ground_truth"])

    # Variant F — cafeteria (plain NER)
    print("Running Variant F (cafeteria — NER only)...", file=sys.stderr)
    entities_F, runtime_F = run_model("cafeteria", passage)
    extracted_F = {e["text"] for e in entities_F}
    ner_metrics_F = evaluate_ner(extracted_F, ground_truth)

    # Variant G — foodon (NER + NEL)
    print("Running Variant G (foodon — NER+NEL)...", file=sys.stderr)
    entities_G, runtime_G = run_model("foodon", passage)
    extracted_G = {e["text"] for e in entities_G}
    nel_results_G = {}
    for e in entities_G:
        if e["text"] not in nel_results_G:
            nel_results_G[e["text"]] = e["links"][0] if e["links"] else None
    ner_metrics_G = evaluate_ner(extracted_G, ground_truth)

    # NEL metrics for G
    gt_lower       = {g.lower() for g in ground_truth}
    linked_G       = {e for e, u in nel_results_G.items() if u}
    gt_covered_G   = {e for e in linked_G if e.lower() in gt_lower}
    nel_gt_cov_G   = len(gt_covered_G) / len(ground_truth) if ground_truth else 0.0
    pipeline_cov_G = len({e for e, u in nel_results_G.items()
                          if u and e.lower() in gt_lower}) / len(ground_truth)

    output = {
        "variant_F": {
            "name":             "F — SciFoodNER cafeteria",
            "ner_metrics":      ner_metrics_F,
            "ner_runtime":      runtime_F,
            "extracted_entities": list(extracted_F),
            "entity_uri_pairs": {e["text"]: None for e in entities_F},
        },
        "variant_G": {
            "name":                  "G — SciFoodNER foodon",
            "ner_metrics":           ner_metrics_G,
            "ner_runtime":           runtime_G,
            "nel_metrics":           {"linked": len(linked_G),
                                      "nil": len(nel_results_G) - len(linked_G),
                                      "gt_coverage": nel_gt_cov_G},
            "pipeline_gt_coverage":  pipeline_cov_G,
            "extracted_entities":    list(extracted_G),
            "entity_uri_pairs":      nel_results_G,
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output}", file=sys.stderr)
    print(f"F: Jaccard={ner_metrics_F['jaccard']:.3f}, {len(extracted_F)} entities", file=sys.stderr)
    print(f"G: Jaccard={ner_metrics_G['jaccard']:.3f}, NEL GT Cov={nel_gt_cov_G:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
