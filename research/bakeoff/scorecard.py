"""Assemble + render the method scorecard."""

from __future__ import annotations

from bakeoff import metrics as M
from bakeoff.result import MethodResult

_COLUMNS = [
    "method", "coverage", "find_median", "find_p90", "find_pct_within_k",
    "nameability", "fanout_max", "depth_max", "spec_mean",
    "faithfulness_is_a", "faithfulness_fabricated", "llm_calls",
]


def build_scorecard(
    results: list[MethodResult],
    *,
    mentioned_leaves: set[str],
    query_leaves: list[str],
    k: int,
    llm=None,
    nameability_sample: int = 25,
) -> list[dict]:
    rows: list[dict] = []
    for r in results:
        find = M.findability(r, query_leaves, k=k)
        faith = M.faithfulness(r)
        fo_max, _ = M.fan_out(r)
        d_max, _ = M.tree_depth(r)
        spec_mean, _ = M.specificity(r)
        rows.append({
            "method": r.name,
            "coverage": M.coverage(r, mentioned_leaves),
            "find_median": find["median_clicks"],
            "find_p90": find["p90_clicks"],
            "find_pct_within_k": find["pct_within_k"],
            "nameability": (M.nameability(r, llm, sample=nameability_sample)
                            if llm is not None else None),
            "fanout_max": fo_max,
            "depth_max": d_max,
            "spec_mean": spec_mean,
            "faithfulness_is_a": faith["is-a"],
            "faithfulness_fabricated": faith["fabricated"],
            "llm_calls": r.llm_calls,
        })
    return rows


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def render_scorecard_markdown(rows: list[dict]) -> str:
    header = "| " + " | ".join(_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    body = "\n".join(
        "| " + " | ".join(_fmt(row.get(c)) for c in _COLUMNS) + " |"
        for row in rows
    )
    return f"{header}\n{sep}\n{body}"


def render_scorecard_html(rows: list[dict]) -> str:
    head = "".join(f"<th>{c}</th>" for c in _COLUMNS)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(row.get(c))}</td>" for c in _COLUMNS) + "</tr>"
        for row in rows
    )
    return f"<table class='scorecard'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
