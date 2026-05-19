"""FoodScholar — hierarchical knowledge graph over nutrition literature."""

__version__ = "0.1.0"

from foodscholar.config import FoodScholarConfig, load_config
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
from foodscholar.storage import (
    ChunkStore,
    Embedder,
    GraphStore,
    InMemoryChunkStore,
    InMemoryGraphStore,
    LLMClient,
)
from foodscholar.versioning import config_hash, make_artifact_meta

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
    "LLMClient",
    "Mention",
    "OntologyTerm",
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
