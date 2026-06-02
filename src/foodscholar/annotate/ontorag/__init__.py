"""OntoRAG-derived tri-hybrid ontology retriever.

Adapted from https://github.com/jan3657/onto_rag — the retrieval half only.
Three retrieval arms over the FoodOn term set:

  - Whoosh BM25 (lexical, over label + synonyms + definition)
  - FAISS kNN on MiniLM embeddings (general semantic)
  - FAISS kNN on SapBERT embeddings (biomedical synonymy)

merged by Reciprocal Rank Fusion into one ranked candidate list.

Per the design decision (docs/DESIGN_agentic_annotate.md): this is
**retrieval only** — it returns ranked `RetrievedCandidate`s. Selection is
done by the agent / linker, not here; OntoRAG's own LLM selector, scorer and
synonym-retry loop are deliberately not adopted.

Gated by the `[ontorag]` extra (`whoosh`, `faiss-cpu`, `sentence-transformers`).
"""

from foodscholar.annotate.ontorag.index import OntoRagIndex, build_index
from foodscholar.annotate.ontorag.retriever import OntoRagRetriever, RetrievedCandidate

__all__ = [
    "OntoRagIndex",
    "OntoRagRetriever",
    "RetrievedCandidate",
    "build_index",
]
