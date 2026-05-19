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
    prefix_filter: list[str] | None = ["FOODON:"]
    """Term-id prefix whitelist for the loaded ontology. Real FoodOn .owl files
    embed NCBITaxon/CHEBI/BFO/ENVO terms inline; the default keeps only FOODON:
    so the linker doesn't match food queries against unrelated ontologies. Set
    to ``null`` to disable filtering (useful for synthetic test fixtures with
    custom prefixes like ``TEST:``)."""


class LinkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lexical_threshold: float = 0.85
    dense_threshold: float = 0.78
    semantic_type_gate: bool = True

    # Dense tier (BRIEF §2). Empty disables it — the linker is then lexical-only.
    dense_model: str = ""
    """Embedding model for the dense tier, e.g.
    'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'. Empty string = dense tier off."""
    dense_cache_path: Path | None = None
    """Optional .npz path for the precomputed term-embedding matrix."""

    # LLM-select tier (deviation from BRIEF §2; see BRIEF §3.5). Opt-in.
    llm_select: bool = False
    """Enable the 4th tier: when no deterministic tier produces a confident
    hit, an LLM picks from the top-k candidates (or rejects). Off by default —
    keeps the linker deterministic unless explicitly opted in."""
    llm_select_threshold: float = 0.90
    """Deterministic-tier confidence below which the LLM tier is consulted."""
    llm_candidate_k: int = 8
    """Number of candidates shown to the LLM selector."""


class AnnotateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ner: Literal["keyword", "agentic"] = "keyword"
    """NER strategy. `keyword` = deterministic ontology-keyword matcher (no
    LLM, the safe default). `agentic` = LLM extracts mentions via the
    configured `llm:` client (needs the `[llm]` extra + a provider)."""

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


LLMProvider = Literal["anthropic", "openai", "groq", "gemini", "ollama"]


class ProviderConfig(BaseModel):
    """One LLM provider + model. API keys are read from the environment
    (ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY) — never
    placed in config files. Ollama needs no key, just a running daemon."""

    model_config = ConfigDict(extra="forbid")
    provider: LLMProvider
    model: str
    host: str | None = None  # ollama only — daemon URL


class LLMConfig(BaseModel):
    """LLM client configuration: a primary provider plus an ordered fallback
    chain. The chain is tried in order; each entry is attempted only if all
    earlier ones errored (timeout, rate limit, auth, service down)."""

    model_config = ConfigDict(extra="forbid")
    primary: ProviderConfig
    fallbacks: list[ProviderConfig] = Field(default_factory=list)
    timeout_s: float = 30.0
    max_retries: int = 2


class FoodScholarConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corpus: CorpusConfig
    ontology: OntologyConfig | None = None
    annotate: AnnotateConfig = Field(default_factory=AnnotateConfig)
    layer_a: LayerAConfig = Field(default_factory=LayerAConfig)
    layer_b: LayerBConfig = Field(default_factory=LayerBConfig)
    layer_c: LayerCConfig = Field(default_factory=LayerCConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    llm: LLMConfig | None = None  # None → facade uses the built-in mock LLM


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
