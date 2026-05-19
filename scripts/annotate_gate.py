"""BRIEF §17 annotate sanity gate — run against real FoodOn.

Three checks the brief requires before the annotate output may feed Layer A:

  1. Entity-linking coverage >= 70%   (gold set, real FOODON: ids)
  2. Top-N most-linked FoodOn terms "look like nutrition" (eyeball)
  3. A sample of random links printed for hand-checking

Run:
    GROQ_API_KEY=... python scripts/annotate_gate.py --config config.local.yaml

The gate exits non-zero if the coverage check fails. Checks 2 and 3 are
human-judgement — the script prints the evidence, the reviewer signs off.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GOLD = REPO / "tests" / "fixtures" / "linker_gold_foodon.jsonl"
COVERAGE_GATE = 0.70
SAMPLE_LINKS = 50
TOP_N = 100


def run(config_path: str) -> int:
    from foodscholar import FoodScholar
    from foodscholar.annotate.embedder import HashEmbedder
    from foodscholar.evaluation.linker import evaluate, load_gold

    # The §17 gate checks linking quality — coverage, the FoodOn frequency
    # list, hand-checked links. Chunk embeddings are not one of those checks,
    # so we pass HashEmbedder explicitly to skip the ~1.3GB SPECTER2/BGE load.
    # The linker still resolves against real FoodOn — only the (irrelevant
    # here) chunk-embedding step is mocked.
    fs = FoodScholar.from_config(config_path, embedder=HashEmbedder())

    info = fs.info()
    print("=" * 70)
    print("ANNOTATE GATE — BRIEF §17")
    print("=" * 70)
    print(f"config:   {config_path}")
    print(f"ner:      {info.get('ner')}")
    print(f"linker:   {fs.linker.linker_id}")
    print(f"embedder: {info.get('embedder')}")
    print(f"llm:      {info.get('llm')}")
    print(f"ontology: {info.get('ontology')}")

    # ---- Check 1: linking coverage on the real-FoodOn gold set ----------
    print("\n" + "-" * 70)
    print("CHECK 1 — entity-linking coverage (gate: >= %.0f%%)" % (COVERAGE_GATE * 100))
    print("-" * 70)
    gold = load_gold(GOLD)
    report = evaluate(fs.linker, gold)
    s = report.summary()
    print(f"  gold records : {s['total']}")
    print(f"  coverage     : {s['coverage']:.1%}")
    print(f"  accuracy     : {s['accuracy']:.1%}")
    print(f"  by tier      : {s['by_tier']}")
    if report.misses:
        print("  misses (text -> expected | got):")
        for text, exp, got in report.misses:
            print(f"    {text!r:28} {exp} | {got}")
    coverage_pass = report.coverage >= COVERAGE_GATE
    print(f"  => {'PASS' if coverage_pass else 'FAIL'}")

    # ---- run the annotate phase over the chunk sample -------------------
    print("\n" + "-" * 70)
    print("running fs.annotate() over the chunk sample ...")
    print("-" * 70)
    fs.load_chunks(fs.config.corpus.chunks_path)
    meta = fs.annotate()
    chunks = fs.chunk_store.scan()
    print(f"  annotated {meta.record_count} chunks")

    # ---- Check 2: top-N most-linked FoodOn terms ------------------------
    print("\n" + "-" * 70)
    print(f"CHECK 2 — top-{TOP_N} most-linked FoodOn terms (eyeball: nutrition?)")
    print("-" * 70)
    freq: Counter[str] = Counter()
    all_links = []
    for ch in chunks:
        for link in ch.entity_links:
            freq[link.ontology_id] += 1
            all_links.append((ch.chunk_id, link))
    if not all_links:
        print("  NO LINKS PRODUCED — annotate yielded zero entity links.")
        return 1 if not coverage_pass else 0
    for oid, n in freq.most_common(TOP_N):
        print(f"  {n:4d}  {oid:18} {fs.ontology.id_to_label(oid)}")

    # ---- Check 3: random sample of links for hand-check -----------------
    print("\n" + "-" * 70)
    n_sample = min(SAMPLE_LINKS, len(all_links))
    print(f"CHECK 3 — {n_sample} random links for hand-checking")
    print("-" * 70)
    rng = random.Random(17)
    for chunk_id, link in rng.sample(all_links, n_sample):
        m = link.mention
        label = fs.ontology.id_to_label(link.ontology_id)
        print(
            f"  [{chunk_id}] {m.text!r:24} ({m.entity_type}) "
            f"-> {link.ontology_id} {label!r} "
            f"[{link.method}, {link.confidence:.2f}]"
        )

    print("\n" + "=" * 70)
    print("Checks 2 and 3 are human-judgement — review the lists above.")
    print(f"Check 1 (coverage): {'PASS' if coverage_pass else 'FAIL'}")
    print("=" * 70)
    return 0 if coverage_pass else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="BRIEF §17 annotate gate")
    ap.add_argument("--config", default="config.local.yaml")
    args = ap.parse_args()
    sys.exit(run(args.config))


if __name__ == "__main__":
    main()
