"""Pydantic config model + YAML / dict loader with ${ENV} substitution."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from foodscholar.io.chunk import SourceType
from foodscholar.io.graph import Facet

_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


# Default GLiNER label vocabulary — kept here so it ships with the package and
# does not require a sister YAML file. Matches gliner.py's production list.
_GLINER_DEFAULT_LABELS: list[str] = [
    "food",
    "nutrient",
    "micronutrient",
    "macronutrient",
    "food component",
    "dietary supplement",
    "dietary pattern",
    "medical condition",
    "biomarker",
    "Country",
    "Measurement",
    "Population",
    "Time expression",
]


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunks_path: Path
    annotated_snapshot_path: Path | None = None
    """Optional parquet path. When set, `FoodScholar.load_and_annotate` writes
    a snapshot of annotated chunks here after upsert, and skips processing if
    the file already exists and is non-empty (idempotent reruns)."""
    ignore_source_types: list[SourceType] = Field(default_factory=list)
    """Source types to skip at ingest time (`abstract`, `textbook`, `guide`).
    Chunks whose `source_type` is in this set are dropped before upsert — their
    NEL annotations and embeddings are skipped too. The `ignore_source_types=`
    kwarg on `FoodScholar.ingest` / `FoodScholar.load_and_annotate` overrides
    this default per call."""


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


class GLinerConfig(BaseModel):
    """GLiNER-bio NER configuration. Defaults match the validated prototype."""

    model_config = ConfigDict(extra="forbid")
    model_id: str = "urchade/gliner_large_bio-v0.1"
    threshold: float = 0.4
    flat_ner: bool = True
    max_length: int = 2048
    batch_size: int = 16
    labels: list[str] = Field(default_factory=lambda: list(_GLINER_DEFAULT_LABELS))


class LinkerConfig(BaseModel):
    """Entity-linking configuration.

    Backend = `hnsw` (local hnswlib index, default) or `elastic` (ES dense_vector,
    opt-in). The HNSW index is built on first use from the loaded FoodOn
    ontology and cached to `nel_index_path` / `nel_metadata_path` (auto-derived
    when those are left unset).
    """

    model_config = ConfigDict(extra="forbid")
    nel_backend: Literal["hnsw", "elastic"] = "hnsw"
    nel_encoder: Literal["sapbert", "biolord", "minilm", "mpnet"] = "biolord"
    nel_top_k: int = 1
    nel_min_sim: float = 0.70
    nel_index_path: Path | None = None
    """Path for the cached hnswlib index file. Auto-derived from the encoder
    and ontology when unset (e.g. data/foodon_hnsw_biolord.bin)."""
    nel_metadata_path: Path | None = None
    """Sister JSON file holding the ordered ontology metadata for the index."""

    # Elastic-only knobs (ignored when nel_backend != "elastic")
    es_index: str | None = None


class AnnotateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ner: Literal["gliner"] = "gliner"
    """NER strategy. Only `gliner` is supported in v0.1 — `keyword` (deterministic
    ontology keyword match) and `agentic` (LLM-extracted) were removed when the
    library switched to GLiNER + HNSW + BioLORD."""

    gliner: GLinerConfig = Field(default_factory=GLinerConfig)
    scientific_embedder: str = "allenai/specter2_base"
    general_embedder: str = "BAAI/bge-large-en-v1.5"
    linker: LinkerConfig = Field(default_factory=LinkerConfig)
    batch_size: int = 16
    """Chunks processed per NER batch in the annotate runner."""


class FacetConfig(BaseModel):
    """Per-facet override on top of `LayerAConfig` globals.

    Every field is optional — a None value falls back to the matching global.
    Use this to set tighter thresholds for noisy facets (foods) without changing
    defaults for sparser ones (allergies, dietary_patterns).
    """

    model_config = ConfigDict(extra="forbid")
    min_support: int | None = None
    max_depth: int | None = None
    collapse_single_child_chains: bool | None = None
    blacklist_terms: list[str] | None = None
    whitelist: list[str] = Field(default_factory=list)
    min_link_confidence: float | None = None
    umbrella_direct_share_max: float | None = None
    umbrella_lifted_share_min: float | None = None


class _ResolvedFacetConfig(BaseModel):
    """Fully-resolved per-facet config — every field non-None. Produced by
    `LayerAConfig.resolve_facet()` so pruner code never reaches back to the
    globals dict."""

    model_config = ConfigDict(extra="forbid")
    min_support: int
    max_depth: int
    collapse_single_child_chains: bool
    blacklist_terms: list[str]
    whitelist: list[str]
    min_link_confidence: float
    umbrella_direct_share_max: float
    umbrella_lifted_share_min: float


_DEFAULT_BLACKLIST: list[str] = [
    # Generic upper-ontology scaffolding that the umbrella rule can't catch
    # (these get linked rarely enough that direct-share isn't a clean signal).
    "material entity",
    "physical object",
    "manufactured product",
]
# Note: FoodOn organizational classes (`food material`, `plant food product`,
# the `* by *` axes, `mammal material`, …) are NOT in the static blacklist.
# The umbrella rule (LayerAConfig.umbrella_direct_share_max /
# umbrella_lifted_share_min) catches them by structure: their direct support
# is near-zero relative to their lifted support, so the rule drops them
# automatically. This keeps the blacklist short and stable across FoodOn
# releases — new organizational classes are caught the same way without YAML
# changes.


class LayerAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_support: int = 20
    max_depth: int = 5
    collapse_single_child_chains: bool = True
    blacklist_terms: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_BLACKLIST)
    )
    min_link_confidence: float = 0.70
    umbrella_direct_share_max: float = 0.05
    """Drop a shelf when `direct/chunk_count` is below this AND `lifted_share`
    is above `umbrella_lifted_share_min` — i.e. almost nobody mentions it
    directly and almost all its support came from descendants. Catches FoodOn
    organizational classes (`plant food product`, `vegetable food product`,
    etc.) by structure rather than by name. Set to 0.0 to disable."""
    umbrella_lifted_share_min: float = 0.85
    """Companion threshold for the umbrella rule (above)."""
    """Discard EntityLinks below this cosine before counting support. Defaults
    to the linker's `nel_min_sim` so projection is no stricter than ingestion
    unless the user explicitly tightens it."""
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
    facet_overrides: dict[Facet, FacetConfig] = Field(default_factory=dict)
    """Per-facet overrides on top of the globals above. A facet not in this
    dict uses globals verbatim."""

    def resolve_facet(self, facet: Facet) -> _ResolvedFacetConfig:
        """Return the fully-resolved (no-None) config for one facet."""
        override = self.facet_overrides.get(facet)
        if override is None:
            return _ResolvedFacetConfig(
                min_support=self.min_support,
                max_depth=self.max_depth,
                collapse_single_child_chains=self.collapse_single_child_chains,
                blacklist_terms=list(self.blacklist_terms),
                whitelist=[],
                min_link_confidence=self.min_link_confidence,
                umbrella_direct_share_max=self.umbrella_direct_share_max,
                umbrella_lifted_share_min=self.umbrella_lifted_share_min,
            )
        return _ResolvedFacetConfig(
            min_support=override.min_support
            if override.min_support is not None
            else self.min_support,
            max_depth=override.max_depth
            if override.max_depth is not None
            else self.max_depth,
            collapse_single_child_chains=override.collapse_single_child_chains
            if override.collapse_single_child_chains is not None
            else self.collapse_single_child_chains,
            blacklist_terms=list(override.blacklist_terms)
            if override.blacklist_terms is not None
            else list(self.blacklist_terms),
            whitelist=list(override.whitelist),
            min_link_confidence=override.min_link_confidence
            if override.min_link_confidence is not None
            else self.min_link_confidence,
            umbrella_direct_share_max=override.umbrella_direct_share_max
            if override.umbrella_direct_share_max is not None
            else self.umbrella_direct_share_max,
            umbrella_lifted_share_min=override.umbrella_lifted_share_min
            if override.umbrella_lifted_share_min is not None
            else self.umbrella_lifted_share_min,
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
    api_key: str | None = None
    """Optional Elasticsearch API key. If unset, the ES client falls back to
    the `ELASTICSEARCH_API_KEY` environment variable. Anonymous access is used
    if neither is provided (suitable for an unauthenticated local cluster)."""
    username: str | None = None
    password: str | None = None
    """Optional HTTP-basic credentials for ES. Take precedence over `api_key`."""
    bulk_size: int = 500
    """How many documents per ES `_bulk` request. Drives both the chunk index
    and the paired entity index. Larger values are faster on healthy clusters
    but each request carries more memory + retry weight; ES rejects requests
    above ~100 MB by default. Sweet spot for chunk-sized text docs is usually
    1000-5000."""


class GraphStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["neo4j", "memory"] = "neo4j"
    url: str | None = None
    user: str | None = None
    password: str | None = None
    """Plaintext Neo4j password. If unset, the driver falls back to the
    `NEO4J_PASSWORD` environment variable. Use `${VAR}` in YAML to inject env
    values without committing secrets to the file."""


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_store: ChunkStoreConfig = Field(default_factory=ChunkStoreConfig)
    graph_store: GraphStoreConfig = Field(default_factory=GraphStoreConfig)


LLMProvider = Literal["anthropic", "openai", "groq", "gemini", "ollama"]


class ProviderConfig(BaseModel):
    """One LLM provider + model.

    API keys can be supplied either in this section (`api_key:` — useful for
    in-code configs and Docker secrets) or via the provider's standard
    environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
    `GROQ_API_KEY`, `GEMINI_API_KEY`). The config value wins when both are
    set. Ollama needs no key — just a running daemon at `host`.
    """

    model_config = ConfigDict(extra="forbid")
    provider: LLMProvider
    model: str
    api_key: str | None = None
    host: str | None = None  # ollama daemon URL; ignored for other providers


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


def resolve_config(
    config: str | Path | dict[str, Any] | FoodScholarConfig,
) -> FoodScholarConfig:
    """Normalize any supported config source into a `FoodScholarConfig`.

    Accepts a YAML file path, a Python dict, or an already-validated config
    object. ${ENV} substitution runs over dicts and strings too, so in-code
    configs can carry env placeholders the same way YAML can.
    """
    if isinstance(config, FoodScholarConfig):
        return config
    if isinstance(config, dict):
        return FoodScholarConfig.model_validate(_substitute_env(config))
    if isinstance(config, (str, Path)):
        return load_config(config)
    raise TypeError(
        f"unsupported config type: {type(config).__name__} "
        "(want str | Path | dict | FoodScholarConfig)"
    )
