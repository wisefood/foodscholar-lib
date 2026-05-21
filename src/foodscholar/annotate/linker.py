"""HNSW-backed entity linker.

Single tier: encode the mention's surface form, kNN against the FoodOn term
index (default: BioLORD over hnswlib `ip`), accept the top-1 hit if its
cosine ≥ `min_sim`. This replaces the previous lexical/fuzzy/dense/LLM
4-tier linker — the validated prototype showed pure dense linking is both
faster and more accurate on the real FoodOn term set, and the fuzzy tier
was the source of the §17 audit's wrong-link findings.

The heavy lifting (encoding + index build/load + cache) lives in
`nel_index.HNSWNELIndex`; this class is thin orchestration over a
`NELIndex`, so it works against `ElasticNELIndex` too once that lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.io.chunk import EntityLink, Mention
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.annotate.nel_index import NELIndex

_log = get_logger("foodscholar.annotate.linker")


class HNSWLinker:
    """Single-tier dense linker over a `NELIndex`."""

    linker_id = "hnsw-linker-v1"

    def __init__(self, nel_index: NELIndex, *, min_sim: float = 0.70) -> None:
        self._nel = nel_index
        self._min_sim = min_sim
        self.linker_id = f"hnsw-linker-v1({nel_index.backend_id})"

    # ------------------------------------------------------------------ Linker protocol

    def link(self, mention: Mention) -> EntityLink | None:
        text = mention.text.strip()
        if not text:
            return None
        hit = self._nel.link(text)
        return self._to_link(mention, hit)

    def link_many(self, mentions: list[Mention]) -> list[EntityLink | None]:
        """Batched path used by the runner. Goes through `NELIndex.link_batch`
        so one encode + one kNN call covers all mentions in a chunk batch.
        """
        if not mentions:
            return []
        surfaces = [m.text.strip() for m in mentions]
        hits = self._nel.link_batch(surfaces)
        return [self._to_link(m, h) for m, h in zip(mentions, hits, strict=True)]

    def dry_run(self, text: str) -> EntityLink | None:
        """Convenience for notebooks: build a Mention from raw text and link it."""
        return self.link(
            Mention(text=text, start=0, end=len(text), score=1.0, ner_model_version="dry-run")
        )

    # ------------------------------------------------------------------ helpers

    def _to_link(
        self, mention: Mention, hit: tuple[str, float] | None
    ) -> EntityLink | None:
        if hit is None:
            return None
        term_id, score = hit
        if score < self._min_sim:
            return None
        return EntityLink(
            mention=mention,
            ontology_id=term_id,
            confidence=score,
            method="dense",
            linker_version=self.linker_id,
        )
