"""Placeholder — annotation pipeline lands in the annotate milestone.

The annotate phase will:
  1. Run SciFoodNER to produce Mention objects per chunk
  2. Link mentions to FoodOn ids (lexical → SapBERT fallback)
  3. Compute SPECTER2 / BGE embeddings per source_type
"""

if __name__ == "__main__":
    raise SystemExit(
        "annotate phase is not implemented yet. See BRIEF.md §12 step 8+."
    )
