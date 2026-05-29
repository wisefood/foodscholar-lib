"""LLM-as-judge: one call per candidate cluster.

Each cluster's shelves are rendered as numbered blocks (label, FoodOn
synonyms, and real sample chunks pulled from the store — this phase runs after
attach, so `chunk.shelf_ids` is populated). The model answers by 1-based index,
which we map back to shelf ids. Output is a `ClusterDecision`: which members
merge into which groups, and which stay alone.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from foodscholar.layer_a.semantic_consolidation.models import (
    ClusterDecision,
    MergeGroup,
)
from foodscholar.layer_a.semantic_consolidation.prompts import (
    FEW_SHOT_EXAMPLES,
    JUDGE_CLUSTER_PROMPT,
    JUDGE_CLUSTER_SCHEMA,
    PROMPT_VERSION,
)
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.config import SemanticConsolidationConfig
    from foodscholar.io.graph import Shelf
    from foodscholar.ontology import FoodOnAPI
    from foodscholar.storage.protocols import ChunkStore, LLMClient

_log = get_logger("foodscholar.semantic_consolidation")


def judge_clusters(
    clusters: list[list[str]],
    shelves_by_id: dict[str, Shelf],
    ontology: FoodOnAPI,
    chunk_store: ChunkStore,
    llm: LLMClient,
    cfg: SemanticConsolidationConfig,
) -> list[ClusterDecision]:
    """Judge every cluster, one LLM call each."""
    samples_cache: dict[str, list[str]] = {}
    decisions: list[ClusterDecision] = []
    for members in clusters:
        prompt = _build_prompt(
            members, shelves_by_id, ontology, chunk_store, cfg, samples_cache
        )
        try:
            obj = llm.generate_json(
                prompt, JUDGE_CLUSTER_SCHEMA, max_tokens=_token_budget(len(members))
            )
        except Exception as exc:  # malformed output / provider error
            _log.warning(
                "semantic_consolidation.judge_error",
                cluster=members,
                error=str(exc),
            )
            obj = {}
        decision = _parse_cluster(members, obj, llm.model_id)
        _log.debug(
            "semantic_consolidation.cluster_judged",
            cluster_size=len(members),
            merge_groups=len(decision.merge_groups),
            keep_alone=len(decision.keep_alone),
        )
        decisions.append(decision)
    return decisions


def _token_budget(cluster_size: int) -> int:
    """Output-token budget scaled to cluster size.

    A cluster can emit one merge group (members + canonical_name + rationale)
    per ~2 shelves plus a keep_alone list, so the response grows with N. The
    old flat 1024 truncated big clusters mid-JSON, which groq's structured
    mode rejects as invalid (empty `failed_generation`). Budget generously —
    output tokens are cheap and truncation costs a whole cluster's decision.
    """
    return max(1024, 256 + cluster_size * 200)


def _build_prompt(
    members: list[str],
    shelves_by_id: dict[str, Shelf],
    ontology: FoodOnAPI,
    chunk_store: ChunkStore,
    cfg: SemanticConsolidationConfig,
    samples_cache: dict[str, list[str]],
) -> str:
    blocks = []
    for idx, shelf_id in enumerate(members, start=1):
        shelf = shelves_by_id[shelf_id]
        syns = _syns(shelf, ontology, cfg) or "(none)"
        samples = _samples_block(shelf_id, chunk_store, cfg, samples_cache)
        blocks.append(
            f"Shelf {idx}: {shelf.label!r}\n"
            f"  synonyms: {syns}\n"
            f"  sample chunks:\n{samples}"
        )
    return JUDGE_CLUSTER_PROMPT.format(
        n=len(members),
        few_shot=("\n" + FEW_SHOT_EXAMPLES) if cfg.use_few_shot else "",
        shelf_blocks="\n\n".join(blocks),
    )


def _syns(shelf: Shelf, ontology: FoodOnAPI, cfg: SemanticConsolidationConfig) -> str:
    if not shelf.foodon_id:
        return ""
    syns = ontology.id_to_synonyms(
        shelf.foodon_id, include_related=cfg.include_related_synonyms
    )
    return ", ".join(syns[: cfg.max_synonyms])


def _samples_block(
    shelf_id: str,
    chunk_store: ChunkStore,
    cfg: SemanticConsolidationConfig,
    cache: dict[str, list[str]],
) -> str:
    if shelf_id not in cache:
        try:
            hits = chunk_store.search(
                query="",
                shelf_ids=[shelf_id],
                k=cfg.sample_chunks_per_shelf,
                use_vector=False,
            )
            cache[shelf_id] = [c.text[:300] for c in hits]
        except Exception:  # store can't serve samples — judge on labels alone
            cache[shelf_id] = []
    samples = cache[shelf_id]
    if not samples:
        return "    (no sample chunks available)"
    return "\n".join(f"    - {s}" for s in samples)


def _parse_cluster(
    members: list[str], obj: dict[str, object], llm_id: str
) -> ClusterDecision:
    """Map the index-based LLM response back onto shelf ids.

    Defensive: indices out of range or repeated across groups are dropped; any
    member the model never mentioned is forced into `keep_alone` so every
    shelf is accounted for exactly once.
    """
    n = len(members)
    decided_at = datetime.now(UTC).isoformat()

    def to_ids(indices: object) -> list[str]:
        out: list[str] = []
        if isinstance(indices, list):
            for i in indices:
                if isinstance(i, int) and 1 <= i <= n:
                    sid = members[i - 1]
                    if sid not in out:
                        out.append(sid)
        return out

    seen: set[str] = set()
    groups: list[MergeGroup] = []
    raw_groups = obj.get("merge_groups", [])
    if isinstance(raw_groups, list):
        for g in raw_groups:
            if not isinstance(g, dict):
                continue
            ids = [sid for sid in to_ids(g.get("members")) if sid not in seen]
            if len(ids) < 2:  # a group needs 2+ real, unseen members
                continue
            seen.update(ids)
            conf = _clamp(g.get("confidence"))
            groups.append(
                MergeGroup(
                    members=ids,
                    canonical_name=str(g.get("canonical_name", "")).strip()
                    or "(unnamed)",
                    confidence=conf,
                    rationale=str(g.get("rationale", "")).strip() or "(no rationale)",
                )
            )

    keep = [sid for sid in to_ids(obj.get("keep_alone")) if sid not in seen]
    seen.update(keep)
    # Any member the model forgot defaults to keep-alone (never silently merged).
    for sid in members:
        if sid not in seen:
            keep.append(sid)

    return ClusterDecision(
        cluster_members=members,
        merge_groups=groups,
        keep_alone=keep,
        llm_id=llm_id,
        prompt_version=PROMPT_VERSION,
        decided_at=decided_at,
    )


def _clamp(value: object) -> float:
    try:
        conf = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, conf))
