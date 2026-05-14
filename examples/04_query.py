"""Demonstrates the retrieval-shape against in-memory stores.

This is the *shape* of the future `foodscholar.retrieval.answer()` call —
the in-memory backend lets you exercise the contracts before the real
implementations land.
"""

from foodscholar.io.graph import Shelf, Theme
from foodscholar.retrieval import Answer
from foodscholar.storage import InMemoryChunkStore, InMemoryGraphStore


def main() -> None:
    chunk_store = InMemoryChunkStore()
    graph_store = InMemoryGraphStore()

    graph_store.upsert_shelves(
        [Shelf(shelf_id="s-med", label="Mediterranean", facet="dietary_patterns", depth=1)]
    )
    graph_store.upsert_themes(
        [
            Theme(
                theme_id="t-olive",
                label="Olive oil",
                shelf_ids=["s-med"],
                discovered_by="leiden",
                discovery_version="demo",
            )
        ]
    )

    hits = chunk_store.search("olive oil", k=5)
    answer = Answer(
        text="(retrieval pipeline lands in the retrieval milestone)",
        tips=[],
        cited_chunks=[h.chunk_id for h in hits],
        cited_cards=[],
        activated_shelves=["s-med"],
        activated_themes=["t-olive"],
        grounding_passed=True,
        llm_model="mock",
        prompt_version="v1",
    )
    print(answer.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
