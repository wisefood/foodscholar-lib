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
    embedder: str = "BAAI/bge-base-en-v1.5"
    linker: LinkerConfig = Field(default_factory=LinkerConfig)
    batch_size: int = 16
    """Chunks processed per NER batch in the annotate runner."""


class LinkBlocklistEntry(BaseModel):
    """A (surface form, ontology id) pair filtered before support collection.

    Catches NEL drift on polysemous surface forms — e.g. the prototype linker
    pairs "fish" with FOODON:00002281 (fish food = aquarium feed) when the
    text is about fish as human food. The pair is matched case-insensitively
    on surface; the ontology_id must match exactly.
    """

    model_config = ConfigDict(extra="forbid")
    surface: str
    ontology_id: str


_DEFAULT_LINK_BLOCKLIST: list[LinkBlocklistEntry] = [
    # The classic NEL-drift example: "fish" in food prose almost always means
    # fish-as-human-food, but the upstream linker pairs it with FOODON:00002281
    # (fish food = aquarium feed). Verified on the audit sample.
    LinkBlocklistEntry(surface="fish", ontology_id="FOODON:00002281"),
]


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
    umbrella_min_count: int | None = None
    link_blocklist: list[LinkBlocklistEntry] | None = None


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
    umbrella_min_count: int
    link_blocklist: list[LinkBlocklistEntry]


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


class SemanticConsolidationConfig(BaseModel):
    """Semantic shelf consolidation — embedding + LLM-as-judge merge pass.

    Runs as a standalone phase *after* `fs.attach()` (so the judge can ground
    on real sample chunks). Catches semantic-duplicate shelves that share
    meaning but not lexical stem — invisible to the structural single-child
    collapse in `prune.py`. Off by default; opt in per config. See
    CONSOLIDATION.md for the full design.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    """Master switch. When False, `fs.semantic_consolidate()` still runs on
    demand but the pipeline never invokes it automatically."""
    facets: list[Facet] = Field(default_factory=lambda: ["foods"])
    """Facets to consolidate. Each is processed independently — pairs never
    cross a facet boundary."""
    cosine_threshold: float = 0.94
    """Minimum cosine similarity for a pair to become a candidate. On a real
    BGE-embedded foods facet, 0.88 chains into one giant hairball and even 0.92
    groups merely-related foods (apple/pear/rice in one cluster). 0.94 keeps
    clusters tight — near-identical labels only — which is what the identity
    judge needs. Use the notebook sweep to retune per corpus; expect
    0.93-0.96."""
    max_candidates_per_shelf: int = 5
    """Cap on candidate pairs touching any single shelf, to bound LLM cost."""
    max_cluster_size: int = 12
    """Hard cap on shelves judged in one LLM call. A connected component larger
    than this is split — weakest cosine edges dropped first — until every piece
    fits. Prevents a transitive hairball from (a) blowing the model's JSON
    output budget and (b) asking the judge to reason over dozens of unrelated
    shelves at once."""
    subtype_patterns: list[str] = Field(
        default_factory=lambda: [
            "canadian",
            "turkey",
            "beef",
            "imitation",
            "red",
            "white",
            "green",
            "silken",
            "extra firm",
            "soft",
            "firm",
        ]
    )
    """Subtype-prefix safety net. If exactly one label of a pair starts with a
    listed word, the pair is excluded — they're parallel siblings (e.g.
    'turkey bacon' vs 'bacon'), never duplicates."""
    exclude_scaffolding: bool = True
    """Drop FoodOn organizational umbrella terms from consolidation entirely.
    These ('food product', 'food consumer group', 'food modification process',
    'dietary supplement') cluster together at high cosine because they're all
    abstract food-ish classes — but none should ever merge; they're navigation
    scaffolding, not duplicate foods. A shelf is treated as scaffolding when it
    has NO FoodOn synonyms AND its label ends in a classifier word
    (`classifier_suffixes`). Real foods almost always carry synonyms, so this
    rarely touches them."""
    classifier_suffixes: list[str] = Field(
        default_factory=lambda: [
            "product",
            "products",
            "process",
            "group",
            "form",
            "analog",
            "analogue",
            "supplement",
            "food",
            "foods",
            "ingredient",
            "ingredients",
            "substance",
            "material",
        ]
    )
    """Trailing words that mark an organizational class rather than a food.
    Used only for the scaffolding filter (combined with the no-synonym test)."""
    judge_enabled: bool = True
    """When False, stop after candidate generation — no LLM calls. Useful for a
    zero-cost candidate preview before trusting the judge."""
    auto_merge_confidence: float = 0.80
    """A shelf placed in a merge group below this confidence is logged as
    'uncertain' and NOT auto-applied. Set to 0.80 because instruct-tuned models
    (e.g. Llama) report bimodal confidence — 0.80 is their 'yes', not a
    marginal score — so a 0.85 gate silently drops obvious merges."""
    include_related_synonyms: bool = False
    """Whether to fold RELATED (not just EXACT) synonyms into the embedding
    text."""
    max_synonyms: int = 5
    """Cap synonyms per shelf in the embedding text to avoid label bloat."""
    sample_chunks_per_shelf: int = 3
    """How many sample chunks to pull per shelf as judge grounding."""
    permanent_block_list: list[tuple[str, str]] = Field(default_factory=list)
    """Pairs of FoodOn ids (matching `shelf.foodon_id`, e.g.
    'FOODON:03310387') that must NEVER merge, regardless of what the judge
    says. Catches systematic polysemy traps the embedder + LLM both fall for
    (oil/fat, olive-oil/vegetable-oil). Order-independent. Grow it whenever a
    bad merge slips through."""
    use_few_shot: bool = True
    """Prepend calibration examples to the judge prompt. Cheap (~300 tokens)
    and the most reliable lever for fixing an over-merging judge — it anchors
    the model on what counts as a meaningful distinction (whole vs skim milk,
    oil vs fat) versus a true duplicate (yoghurt vs yogurt)."""


class LayerAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_support: int = 20
    max_depth: int = 5
    collapse_single_child_chains: bool = True
    blacklist_terms: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_BLACKLIST)
    )
    min_link_confidence: float = 0.70
    umbrella_direct_share_max: float = 0.10
    """The umbrella rule drops a shelf iff **all three** conditions hold
    simultaneously (AND-chained, not OR):

      1. `chunk_count >= umbrella_min_count`            (size guard fires first)
      2. `direct / chunk_count < umbrella_direct_share_max`
      3. `lifted / chunk_count > umbrella_lifted_share_min`

    The size guard prevents small niche shelves (where direct_share has high
    variance) from being mistaken for umbrellas. Conditions 2+3 identify
    FoodOn organizational classes — almost nobody mentions them directly,
    almost all their support is from descendants. Set
    `umbrella_direct_share_max=0.0` to disable the rule entirely (condition 2
    becomes unsatisfiable)."""
    umbrella_lifted_share_min: float = 0.85
    """Companion threshold for the umbrella rule (above)."""
    umbrella_min_count: int = 25
    """Minimum chunk_count for the umbrella rule to apply. Below this, the
    threshold pass alone decides. Prevents the rule from chewing into small
    niche shelves where direct_share has high variance. The min_support
    threshold (default 20-25) is the lower guard; this floor sits above it
    so the umbrella rule has a stable denominator.

    When `umbrella_min_count <= min_support` the guard is a no-op (every
    threshold-survivor is also umbrella-eligible) — that's the default
    configuration. Raise this above min_support only if a corpus has small
    legitimate niche shelves you want spared from umbrella detection."""
    link_blocklist: list[LinkBlocklistEntry] = Field(
        default_factory=lambda: list(_DEFAULT_LINK_BLOCKLIST)
    )
    """Pre-filter EntityLinks before support collection. Each entry pairs a
    lowercased surface form with a specific ontology_id; matching pairs are
    dropped. Catches NEL drift on polysemous surface forms (e.g. "fish" being
    linked to FOODON:00002281 'fish food' = aquarium feed) without
    re-annotating the corpus."""
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
    semantic_consolidation: SemanticConsolidationConfig = Field(
        default_factory=SemanticConsolidationConfig
    )
    """Embedding + LLM-as-judge merge pass. Off by default; runs as a separate
    phase after attach. See `SemanticConsolidationConfig`."""

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
                umbrella_min_count=self.umbrella_min_count,
                link_blocklist=list(self.link_blocklist),
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
            umbrella_min_count=override.umbrella_min_count
            if override.umbrella_min_count is not None
            else self.umbrella_min_count,
            link_blocklist=list(override.link_blocklist)
            if override.link_blocklist is not None
            else list(self.link_blocklist),
        )


class SimilarityConfig(BaseModel):
    """Pass 1 (similarity) graph + algorithm knobs.

    `algorithm` is restricted to `"leiden"` in v1 — HDBSCAN is documented as
    a fallback in the brief but cut from v1 per the implementation plan."""

    model_config = ConfigDict(extra="forbid")
    knn_k: int = 15
    edge_threshold: float = 0.55
    require_mutual: bool = True
    algorithm: Literal["leiden"] = "leiden"


class RelatednessConfig(BaseModel):
    """Pass 2 (relatedness) graph knobs.

    - `tau_strict`: minimum entity-link confidence to participate in edges.
    - `min_shared_ids`: edge created iff >= this many shared FoodOn IDs.
    - `max_doc_frequency`: entities appearing in > this fraction of the
      shelf's chunks are dropped (they carry no discriminative signal).
    - `always_exclude_iris`: never-edge-creators. The umbrella class
      FOODON:00001002 ('food product') is the default exclusion — it
      survived Layer A and gets ancestor-propagated onto almost every chunk.
    """

    model_config = ConfigDict(extra="forbid")
    tau_strict: float = 0.80
    min_shared_ids: int = 2
    max_doc_frequency: float = 0.40
    algorithm: Literal["leiden"] = "leiden"
    always_exclude_iris: list[str] = Field(
        default_factory=lambda: ["FOODON:00001002"]
    )


class LeidenConfig(BaseModel):
    """Shared by both passes. `random_state` is the determinism contract —
    same chunks + same seed = identical theme assignment across runs."""

    model_config = ConfigDict(extra="forbid")
    resolution: float = 1.0
    n_iterations: int = 10
    min_community_size: int = 15
    random_state: int = 42


class MergeConfig(BaseModel):
    """Greedy pair-assignment merge. `combined_similarity =
    chunk_weight * J(chunks) + entity_weight * J(entities)` per
    (sim_i, rel_j); pairs at or above `dedupe_threshold` collapse into
    `discovery_pass="merged"` themes."""

    model_config = ConfigDict(extra="forbid")
    chunk_weight: float = 0.6
    entity_weight: float = 0.4
    dedupe_threshold: float = 0.70


class LabelingConfig(BaseModel):
    """Theme labeling. `"llm"` is v1 default — navigation labels need to read
    well and per-run cost is ~$0.60. `"keyword"` (pure c-TF-IDF) is a free
    deterministic fallback. c-TF-IDF is always computed and fed to the LLM
    as keyword context."""

    model_config = ConfigDict(extra="forbid")
    strategy: Literal["keyword", "llm"] = "llm"
    top_keywords: int = 5
    llm_max_tokens: int = 32  # 3-5 word labels


class LayerBAuditConfig(BaseModel):
    """WARN-level gates emitted by `audit_layer_b()`. None of these flip
    `LayerBAuditReport.passed` (which only checks CRITICAL invariants); they
    surface in the notebook to drive tuning."""

    model_config = ConfigDict(extra="forbid")
    target_themes_per_shelf_min: int = 3
    target_themes_per_shelf_max: int = 12
    merged_rate_min: float = 0.20
    merged_rate_max: float = 0.80


class LayerBConfig(BaseModel):
    """Layer B (theme discovery) — dual-pass + merge per the brief.

    See `layer_b_construction_brief.md` §5 for the full knob list and the
    accompanying plan for the v1 decisions (Leiden-only, LLM labels by
    default, embedded-fraction gate, etc.).
    """

    model_config = ConfigDict(extra="forbid")
    min_chunks_per_shelf: int = 50
    min_embedded_fraction: float = 0.80
    """Skip shelves where < this fraction of chunks have embeddings —
    clustering a biased subsample is worse than not clustering at all."""

    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    relatedness: RelatednessConfig = Field(default_factory=RelatednessConfig)
    leiden: LeidenConfig = Field(default_factory=LeidenConfig)
    merge: MergeConfig = Field(default_factory=MergeConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    audit: LayerBAuditConfig = Field(default_factory=LayerBAuditConfig)
    global_similarity_max_chunks: int = 50_000
    """Safety cap: if the global similarity pass would see more chunks than this,
    fall back to per-shelf Pass 1 and emit a warning."""


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
