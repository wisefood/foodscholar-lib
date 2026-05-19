"""Build the SapBERT dense-index cache (.npz) — meant to run on a GPU box.

Embedding ~29k FoodOn terms with SapBERT is slow on CPU. Run this once on a
GPU server, then copy the resulting .npz back; the linker's dense tier loads
it directly (the cache fingerprint matches as long as the same foodon.owl and
the same model are used).

Usage:
    python scripts/build_dense_index.py --config config.local.yaml

Reads ontology.foodon_path / ontology.cache_path / ontology.prefix_filter and
annotate.linker.dense_model / dense_cache_path from the config. Writes the
.npz to dense_cache_path (data/sapbert_terms.npz by default).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the SapBERT dense-index cache")
    ap.add_argument("--config", default="config.local.yaml")
    args = ap.parse_args()

    from foodscholar.annotate.dense_index import DenseIndex
    from foodscholar.annotate.embedder import SapBERTEmbedder
    from foodscholar.config import load_config
    from foodscholar.ontology.api import FoodOnAPI
    from foodscholar.ontology.foodon import load_ontology

    cfg = load_config(args.config)
    linker_cfg = cfg.annotate.linker
    if not linker_cfg.dense_model:
        raise SystemExit("annotate.linker.dense_model is not set in the config")
    cache_path = linker_cfg.dense_cache_path or "data/sapbert_terms.npz"

    print(f"config        : {args.config}")
    print(f"foodon        : {cfg.ontology.foodon_path}")
    print(f"prefix_filter : {cfg.ontology.prefix_filter}")
    print(f"dense_model   : {linker_cfg.dense_model}")
    print(f"cache_path    : {cache_path}")

    print("\nloading ontology ...")
    terms = load_ontology(cfg.ontology.foodon_path, cache_path=cfg.ontology.cache_path)
    api = FoodOnAPI(terms, prefix_filter=tuple(cfg.ontology.prefix_filter))
    n_terms = sum(1 for t in api if not t.obsolete)
    print(f"  {len(api)} terms ({n_terms} non-obsolete to embed)")

    print(f"\nbuilding SapBERT embedder ({linker_cfg.dense_model}) ...")
    embedder = SapBERTEmbedder(linker_cfg.dense_model)

    print("embedding terms + building dense index (this is the slow part) ...")
    t0 = time.time()
    index = DenseIndex.build(api, embedder, cache_path=cache_path)
    dt = time.time() - t0
    print(f"\ndone — {index.size} term vectors in {dt:.1f}s")

    out = Path(cache_path)
    if out.exists():
        print(f"cache written : {out}  ({out.stat().st_size / 1e6:.1f} MB)")
        print("\nCopy this .npz back to the same data/ path on the gate machine.")
    else:
        print("WARNING: cache file was not written — check dense_cache_path.")


if __name__ == "__main__":
    main()
