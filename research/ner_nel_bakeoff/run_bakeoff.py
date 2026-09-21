#!/usr/bin/env python3
"""Run the 2x2 NER x encoder bake-off and report upstream + downstream metrics.

Archived research, not shipped. See README.md for why the published numbers do
not transfer unexamined, and for what the output has to show before a default
is flipped.

Usage:
    python research/ner_nel_bakeoff/run_bakeoff.py --config config.yaml \
        --sample 2000 --out research/ner_nel_bakeoff/results
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from foodscholar import FoodScholar

NER_BACKENDS = ["gliner", "gliner2"]
ENCODERS = ["biolord", "sapbert"]


@dataclass
class CellResult:
    """One (ner, encoder) cell of the bake-off."""

    ner: str
    encoder: str
    n_chunks: int = 0
    n_mentions: int = 0
    n_links: int = 0
    n_distinct_uris: int = 0
    mentions_per_chunk: float = 0.0
    link_rate: float = 0.0
    # downstream — the part the original benchmark could not measure
    n_entities: int = 0
    n_entities_with_facet: int = 0
    n_shelves: int | None = None
    n_shelves_supported: int | None = None
    n_themes: int | None = None
    precision_sample: list[dict] = field(default_factory=list)
    error: str | None = None


def _sample_chunks(fs: FoodScholar, n: int, seed: int) -> list:
    chunks = fs.chunk_store.scan()
    if n and n < len(chunks):
        rng = random.Random(seed)
        chunks = rng.sample(chunks, n)
    return chunks


def run_cell(
    config_path: str,
    ner: str,
    encoder: str,
    *,
    sample: int,
    seed: int,
    precision_sample_size: int,
    downstream: bool,
) -> CellResult:
    result = CellResult(ner=ner, encoder=encoder)
    try:
        fs = FoodScholar.from_config(config_path)
        fs.config.annotate.ner = ner
        fs.config.annotate.linker.nel_encoder = encoder
        # Index path is derived from the encoder, so the two coexist on disk.
        fs.config.annotate.linker.nel_index_path = None
        fs.config.annotate.linker.nel_metadata_path = None

        chunks = _sample_chunks(fs, sample, seed)
        result.n_chunks = len(chunks)

        mentions_per_chunk = fs.ner.extract_batch([c.text for c in chunks])
        flat = [m for ms in mentions_per_chunk for m in ms]
        result.n_mentions = len(flat)
        result.mentions_per_chunk = len(flat) / max(len(chunks), 1)

        links = fs.linker.link_many(flat)
        linked = [ln for ln in links if ln is not None]
        result.n_links = len(linked)
        result.n_distinct_uris = len({ln.ontology_id for ln in linked})
        result.link_rate = len(linked) / max(len(flat), 1)

        # Hand-scoring queue: the sample a human marks correct/incorrect.
        rng = random.Random(seed)
        pool = rng.sample(linked, min(precision_sample_size, len(linked)))
        result.precision_sample = [
            {
                "surface": ln.mention.text,
                "entity_type": ln.mention.entity_type,
                "ontology_id": ln.ontology_id,
                "confidence": round(ln.confidence, 4),
                "verdict": None,  # <- fill in by hand
            }
            for ln in pool
        ]

        if downstream:
            fs.annotate()
            fs.build_entities()
            entities = fs.entity_store.scan()
            result.n_entities = len(entities)
            result.n_entities_with_facet = sum(1 for e in entities if e.facet_hint)
            fs.build_layer_a()
            fs.attach()
            shelves = fs.graph_store.list_shelves()
            result.n_shelves = len(shelves)
            floor = fs.config.layer_b.min_chunks_per_shelf
            result.n_shelves_supported = sum(1 for s in shelves if s.chunk_count >= floor)
            fs.build_layer_b(facet="foods")
            result.n_themes = len(fs.graph_store.list_themes())
    except Exception as e:  # a failed cell must not kill the run
        result.error = f"{type(e).__name__}: {e}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample", type=int, default=2000, help="0 = whole corpus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision-sample", type=int, default=100)
    parser.add_argument("--out", default="research/ner_nel_bakeoff/results")
    parser.add_argument(
        "--no-downstream",
        action="store_true",
        help="Skip Layer A/B rebuilds — much faster, but misses the point.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[CellResult] = []
    for ner in NER_BACKENDS:
        for encoder in ENCODERS:
            print(f"\n=== cell: ner={ner} encoder={encoder} ===", flush=True)
            cell = run_cell(
                args.config,
                ner,
                encoder,
                sample=args.sample,
                seed=args.seed,
                precision_sample_size=args.precision_sample,
                downstream=not args.no_downstream,
            )
            results.append(cell)
            path = out_dir / f"{ner}__{encoder}.json"
            path.write_text(json.dumps(asdict(cell), indent=2))
            print(f"  -> {path}", flush=True)

    header = (
        f"{'ner':<10} {'encoder':<10} {'ment/chunk':>11} {'link_rate':>10} "
        f"{'uris':>7} {'entities':>9} {'faceted':>8} {'shelves':>8} "
        f"{'supported':>10} {'themes':>7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        if r.error:
            print(f"{r.ner:<10} {r.encoder:<10}  ERROR: {r.error}")
            continue
        print(
            f"{r.ner:<10} {r.encoder:<10} {r.mentions_per_chunk:>11.2f} "
            f"{r.link_rate:>10.3f} {r.n_distinct_uris:>7} {r.n_entities:>9} "
            f"{r.n_entities_with_facet:>8} {r.n_shelves!s:>8} "
            f"{r.n_shelves_supported!s:>10} {r.n_themes!s:>7}"
        )
    print(
        "\nHand-score the `precision_sample` entries in each JSON before "
        "deciding. Link rate is not accuracy."
    )
    (out_dir / "summary.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2)
    )


if __name__ == "__main__":
    main()
