"""Regenerate NB3 flat-vs-graph compare figures from the saved JSON artifacts.

The original `compare_fig` drew each delta hit's reason chip on a line *below*
the excerpt (y - 0.04) while rows were only `step` (0.083) apart — so on dense
delta columns (dietary_fibre, 7/10) the chips overwrote the rows beneath them.

This re-renders from {anchor}_flat.json / {anchor}_graph.json / {anchor}_delta.json
(no retrieval re-run) with the reason chip placed *inline*, right of the
excerpt, so nothing collides regardless of how many delta hits a column has.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import cs_common as cs

cs.init_mpl()
OUT = cs.artifacts_dir("nb3_retrieval")
K = cs.CONFIG["retrieval_k"]

REASON_COLOR = {
    "different-document": cs.COLORS["amber"],
    "different-food-entity": cs.COLORS["purple"],
    "paraphrase": cs.COLORS["teal"],
    "unresolved": cs.COLORS["grey"],
}


def regen(key):
    flat = json.loads((OUT / f"{key}_flat.json").read_text())
    graph = json.loads((OUT / f"{key}_graph.json").read_text())
    delta = json.loads((OUT / f"{key}_delta.json").read_text())

    flat_ids = flat["ids"]
    graph_ids = graph["ids"]
    flat_set = set(flat_ids)
    query = flat["query"]

    # excerpt for every id: flat hits carry their own excerpt; graph-only ids
    # get theirs from the delta rows. Together they cover every graph id.
    excerpts = {h["chunk_id"]: h["excerpt"] for h in flat["hits"]}
    for r in delta["delta"]:
        excerpts[r["chunk_id"]] = r["excerpt"]
    reason_by_id = {r["chunk_id"]: r["reason"] for r in delta["delta"]}
    n_delta = len(delta["delta"])

    fig, ax = cs.slide(
        f"Flat vs graph-expanded retrieval — {key}",
        eyebrow="lens 3 · M3",
        caption=f'Query: "{query}"',
    )
    cs.kpi(ax, 0.82, 0.99, f"{n_delta}/{K}", "graph-only hits",
           color=cs.COLORS["purple"], w=0.17, h=0.24, value_fontsize=24)

    def column(ids, x0, head, color, mark_delta):
        cs.chip(ax, x0, 0.96, head, color=color, fontsize=14)
        top = 0.84
        step = 0.083
        for i, cid in enumerate(ids):
            y = top - i * step
            is_delta = mark_delta and cid not in flat_set
            ax.text(x0, y, f"{i+1:2d}.", fontsize=11, weight="bold",
                    color=cs.COLORS["muted"], va="top", family="DejaVu Sans Mono")
            # delta rows leave room for an inline reason chip; flat rows run full width
            elen = 44 if is_delta else 58
            ax.text(x0 + 0.028, y, cs.excerpt(excerpts.get(cid, cid), elen),
                    fontsize=10.5, color=cs.COLORS["ink"], va="top")
            if is_delta:
                rsn = reason_by_id.get(cid, "")
                # inline, right of the (shorter) excerpt — never overlaps the row below
                cs.chip(ax, x0 + 0.345, y - 0.012, rsn,
                        color=REASON_COLOR.get(rsn, cs.COLORS["grey"]),
                        fontsize=8.5, weight="normal", pad=0.25)

    column(flat_ids, 0.0, "flat — BM25 + kNN (RRF)", cs.COLORS["teal"], mark_delta=False)
    column(graph_ids, 0.50, "graph-expanded", cs.COLORS["purple"], mark_delta=True)
    cs.save_slide(fig, OUT / f"{key}_compare.png")
    print("wrote", OUT / f"{key}_compare.png")


if __name__ == "__main__":
    keys = sys.argv[1:] or list(cs.CONFIG["queries"].keys())
    for k in keys:
        regen(k)
