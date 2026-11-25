"""FoodScholar — hierarchical knowledge graph over nutrition literature.

Public names resolve lazily (PEP 562): ``import foodscholar.chunking`` loads the
chunker and nothing else, while ``from foodscholar import FoodScholar`` still
works exactly as before — it just imports the facade on first use.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.3.0"

_LAZY: dict[str, str] = {
    "FoodScholarConfig": "foodscholar.config",
    "RetrievalConfig": "foodscholar.config",
    "load_config": "foodscholar.config",
    "FoodScholar": "foodscholar.facade",
    "CardHandle": "foodscholar.graph_view",
    "GraphView": "foodscholar.graph_view",
    "ShelfHandle": "foodscholar.graph_view",
    "ThemeHandle": "foodscholar.graph_view",
    "ArtifactMeta": "foodscholar.io",
    "Card": "foodscholar.io",
    "Chunk": "foodscholar.io",
    "EntityLink": "foodscholar.io",
    "Mention": "foodscholar.io",
    "OntologyTerm": "foodscholar.io",
    "Shelf": "foodscholar.io",
    "Theme": "foodscholar.io",
    "FallbackLLMClient": "foodscholar.llm",
    "build_llm": "foodscholar.llm",
    "FoodOnAPI": "foodscholar.ontology",
    "load_ontology": "foodscholar.ontology",
    "KGGenRetriever": "foodscholar.retrieval",
    "RetrievalHit": "foodscholar.retrieval",
    "RetrievalTrace": "foodscholar.retrieval",
    "ChunkStore": "foodscholar.storage",
    "Embedder": "foodscholar.storage",
    "GraphStore": "foodscholar.storage",
    "InMemoryChunkStore": "foodscholar.storage",
    "InMemoryGraphStore": "foodscholar.storage",
    "LLMClient": "foodscholar.storage",
    "config_hash": "foodscholar.versioning",
    "make_artifact_meta": "foodscholar.versioning",
}

__all__ = [
    "ArtifactMeta",
    "Card",
    "CardHandle",
    "Chunk",
    "ChunkStore",
    "Embedder",
    "EntityLink",
    "FallbackLLMClient",
    "FoodOnAPI",
    "FoodScholar",
    "FoodScholarConfig",
    "GraphStore",
    "GraphView",
    "InMemoryChunkStore",
    "InMemoryGraphStore",
    "KGGenRetriever",
    "LLMClient",
    "Mention",
    "OntologyTerm",
    "RetrievalConfig",
    "RetrievalHit",
    "RetrievalTrace",
    "Shelf",
    "ShelfHandle",
    "Theme",
    "ThemeHandle",
    "__version__",
    "build_llm",
    "config_hash",
    "load_config",
    "load_ontology",
    "make_artifact_meta",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'foodscholar' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value  # cache: later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


if TYPE_CHECKING:  # static type checkers and IDEs see the real names
    from foodscholar.config import FoodScholarConfig, RetrievalConfig, load_config
    from foodscholar.facade import FoodScholar
    from foodscholar.graph_view import CardHandle, GraphView, ShelfHandle, ThemeHandle
    from foodscholar.io import (
        ArtifactMeta,
        Card,
        Chunk,
        EntityLink,
        Mention,
        OntologyTerm,
        Shelf,
        Theme,
    )
    from foodscholar.llm import FallbackLLMClient, build_llm
    from foodscholar.ontology import FoodOnAPI, load_ontology
    from foodscholar.retrieval import KGGenRetriever, RetrievalHit, RetrievalTrace
    from foodscholar.storage import (
        ChunkStore,
        Embedder,
        GraphStore,
        InMemoryChunkStore,
        InMemoryGraphStore,
        LLMClient,
    )
    from foodscholar.versioning import config_hash, make_artifact_meta
