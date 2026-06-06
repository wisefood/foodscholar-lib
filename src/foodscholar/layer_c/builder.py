"""Layer C orchestrator: iterate Layer B themes, summarize each into a Card.

For each theme: gather member chunk texts → Stage 1 (map-reduce extractive) →
Stage 2 (LLM refinement) → Card. Persists via `persist_cards` unless `dry_run`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from foodscholar.layer_c.models import LayerCReport
from foodscholar.layer_c.persist import persist_cards
from foodscholar.layer_c.registry import build_summarizer
from foodscholar.layer_c.stage1 import run_stage1
from foodscholar.layer_c.stage2 import run_stage2

if TYPE_CHECKING:
    from foodscholar.facade import FoodScholar


@dataclass
class _ThemeAdapter:
    """Expose the `_ThemeLike` shape Stage 2 expects from a ThemeHandle."""

    theme_id: str
    label: str
    facet: str
    keyword_terms: list[str]


def build_layer_c(
    fs: FoodScholar,
    *,
    facet: str = "foods",
    dry_run: bool = False,
) -> LayerCReport:
    """Build one Card per Layer B theme of `facet`."""
    cfg = fs.config.layer_c
    summarizer = build_summarizer(cfg.stage1_method, cfg)

    themes = [t for t in fs.graph.themes() if t.model.facet == facet]
    cards = []
    skipped = failed = 0
    strat: dict[str, int] = {}

    for th in themes:
        chunk_ids = list(fs.graph_store.get_chunks_for_theme(th.theme_id))
        texts = [c.text for c in fs.chunk_store.get_many(chunk_ids)]
        if not any(t and t.strip() for t in texts):
            skipped += 1
            continue

        s1 = run_stage1(
            texts, summarizer,
            map_reduce_threshold=cfg.map_reduce_threshold,
            group_char_budget=cfg.group_char_budget,
        )
        adapter = _ThemeAdapter(
            theme_id=th.theme_id, label=th.label, facet=th.model.facet,
            keyword_terms=list(th.model.keyword_terms),
        )
        try:
            card = run_stage2(fs.llm, s1, adapter, chunk_ids, cfg)
        except Exception:
            failed += 1
            continue

        cards.append(card)
        strat[s1.strategy] = strat.get(s1.strategy, 0) + 1

    if not dry_run:
        persist_cards(cards, fs.graph_store)

    return LayerCReport(
        n_themes=len(themes), n_cards=len(cards),
        n_skipped=skipped, n_failed=failed, strategy_counts=strat,
    )
