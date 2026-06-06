"""Read-only evaluation harness: run every extractive method over a theme's
chunks and emit per-method metrics for side-by-side comparison. No LLM, no
persistence. Used to pick/tune `config.layer_c.stage1_method`."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from foodscholar.layer_c.models import MethodResult
from foodscholar.layer_c.registry import all_summarizers

if TYPE_CHECKING:
    from foodscholar.facade import FoodScholar


def benchmark_theme(fs: "FoodScholar", theme_id: str) -> list[MethodResult]:
    """Run all registered methods over `theme_id`'s chunks (single pass)."""
    cfg = fs.config.layer_c
    chunk_ids = list(fs.graph_store.get_chunks_for_theme(theme_id))
    texts = [c.text for c in fs.chunk_store.get_many(chunk_ids)]
    input_chars = sum(len(t) for t in texts)

    results: list[MethodResult] = []
    for summ in all_summarizers(cfg):
        t0 = time.perf_counter()
        summary = summ.summarize(texts)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        results.append(MethodResult(
            method=summ.name, summary=summary,
            input_chunks=len(texts), input_chars=input_chars,
            execution_time_ms=elapsed_ms, summary_length_chars=len(summary),
        ))
    return results


def benchmark_facet(
    fs: "FoodScholar",
    *,
    facet: str = "foods",
    themes: int = 5,
    out: str | None = None,
) -> dict[str, list[MethodResult]]:
    """Benchmark the `themes` largest themes of `facet`; write combined JSON."""
    cfg = fs.config.layer_c
    handles = sorted(
        (t for t in fs.graph.themes() if t.model.facet == facet),
        key=lambda t: t.chunk_count, reverse=True,
    )[:themes]

    by_theme = {h.theme_id: benchmark_theme(fs, h.theme_id) for h in handles}

    out_path = Path(out or cfg.benchmark_out_dir) / f"benchmark_{facet}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {tid: [r.model_dump() for r in rs] for tid, rs in by_theme.items()},
        indent=2,
    ), encoding="utf-8")
    return by_theme
