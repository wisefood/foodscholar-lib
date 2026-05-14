"""Pydantic config model + YAML loader with ${ENV} substitution."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from foodscholar.io.graph import Facet

_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunks_path: Path


class OntologyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    foodon_path: Path
    cache_path: Path | None = None
    include_imports: bool = False


class LinkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lexical_threshold: float = 0.85
    dense_threshold: float = 0.78
    semantic_type_gate: bool = True


class AnnotateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ner_model: str = "sci_food_ner_v1"
    scientific_embedder: str = "allenai/specter2_base"
    general_embedder: str = "BAAI/bge-large-en-v1.5"
    linker: LinkerConfig = Field(default_factory=LinkerConfig)


class LayerAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_support: int = 20
    max_depth: int = 5
    collapse_single_child_chains: bool = True
    blacklist_terms: list[str] = Field(default_factory=list)
    facets: list[Facet] = Field(
        default_factory=lambda: [
            "foods",
            "health",
            "sustainability",
            "dietary_patterns",
            "allergies",
            "nutrients",
        ]
    )


class LayerBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_chunks_per_shelf: int = 50
    algorithm: Literal["leiden", "hdbscan", "bertopic"] = "leiden"
    resolution: float = 1.0
    recurse_threshold: int = 200


class LayerCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm_model: str = "claude-sonnet-4-6"
    prompt_version: str = "v1"
    sample_size: int = 12
    grounding_check: Literal["strict", "lenient", "off"] = "strict"
    safety_sensitive_facets: list[Facet] = Field(default_factory=lambda: ["allergies"])


class ChunkStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["elastic", "memory"] = "elastic"
    url: str | None = None
    index: str | None = None


class GraphStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["neo4j", "memory"] = "neo4j"
    url: str | None = None
    user: str | None = None
    password: str | None = None


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_store: ChunkStoreConfig = Field(default_factory=ChunkStoreConfig)
    graph_store: GraphStoreConfig = Field(default_factory=GraphStoreConfig)


class FoodScholarConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corpus: CorpusConfig
    ontology: OntologyConfig | None = None
    annotate: AnnotateConfig = Field(default_factory=AnnotateConfig)
    layer_a: LayerAConfig = Field(default_factory=LayerAConfig)
    layer_b: LayerBConfig = Field(default_factory=LayerBConfig)
    layer_c: LayerCConfig = Field(default_factory=LayerCConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


def _substitute_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            var = match.group(1)
            return os.environ.get(var, match.group(0))

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


def load_config(path: str | Path) -> FoodScholarConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping")
    resolved = _substitute_env(raw)
    return FoodScholarConfig.model_validate(resolved)
