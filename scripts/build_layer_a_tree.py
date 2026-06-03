"""Re-run Layer B for `foods` in per-shelf Pass-1 mode, then render the
interactive Layer A tree to data/viz/layer_a_tree_foods.html.

Per-shelf Pass 1 yields shelf-scoped similarity candidates that can align with
the per-shelf relatedness pass, so a real `merged` origin bucket appears
(global mode produced zero, and its themes smear across hundreds of shelves).

Run from the repo root with live Neo4j + Elasticsearch up and the corpus
embedded (same stores build_graph.ipynb uses):

    python scripts/build_layer_a_tree.py
"""

from __future__ import annotations

from pathlib import Path

from foodscholar import FoodScholar

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "corpus": {
        "chunks_path": str(REPO_ROOT / "data" / "foodscholar" / "corpus"),
        "annotated_snapshot_path": str(REPO_ROOT / "data" / "annotated.parquet"),
    },
    "ontology": {
        "foodon_path": str(REPO_ROOT / "data" / "foodon.owl"),
        "cache_path": str(REPO_ROOT / "data" / "foodon_cache.parquet"),
        "prefix_filter": ["FOODON:"],
    },
    "llm": {"primary": {"provider": "groq", "model": "llama-3.1-8b-instant"}},
    "storage": {
        "chunk_store": {
            "backend": "elastic",
            "url": "http://localhost:9200",
            "index": "foodscholar_chunks",
        },
        "graph_store": {
            "backend": "neo4j",
            "url": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password",
        },
    },
}


def main() -> None:
    fs = FoodScholar.from_config(CONFIG)

    # Switch Pass 1 to per-shelf and rebuild Layer B for foods (replaces themes).
    fs.config.layer_b.pass1_mode = "per_shelf"
    artifact = fs.build_layer_b(facet="foods", dry_run=False)
    print("Layer B rebuilt:", artifact)

    by_pass: dict[str, int] = {"merged": 0, "global_similarity": 0, "relatedness": 0}
    for t in fs.graph_store.list_themes():
        by_pass[t.discovery_pass] = by_pass.get(t.discovery_pass, 0) + 1
    print("themes by pass:", by_pass)

    out = REPO_ROOT / "data" / "viz" / "layer_a_tree_foods.html"
    path = fs.viz.layer_a_tree("foods").render("tree", output=out)
    print("wrote", path)


if __name__ == "__main__":
    main()
