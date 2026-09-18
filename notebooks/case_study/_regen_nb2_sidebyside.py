"""Regenerate NB2 side-by-side theme figures from the saved JSON artifacts.

The original `sidebyside` cell packed each theme as label (at y) + keyword line
(at y - 0.052) with a row step as small as 0.80/6 ≈ 0.133, so the keyword line of
one theme crowded the label of the next — labels and keywords visually merged
(dietary_fibre, both 6-theme columns). The kernel also crashed mid-run.

This re-renders from {anchor}_leiden.json / {anchor}_bertopic.json (no clustering
re-run) with a fixed, generous row layout: label and keyword line sit close to
each other but each theme block is well separated from the next.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import cs_common as cs

cs.init_mpl()
OUT = cs.artifacts_dir("nb2_themes")
MAX_ROWS = 6  # original cap for legibility

# anchor -> shelf label (the JSON carries shelf_id, not the human label)
SHELF_LABEL = {
    "olive_oil": "olive oil",
    "legume": "legume food product",
    "fish": "fish food product",
    "dietary_fibre": "dietary fibre",
}


def sidebyside(anchor):
    L = json.loads((OUT / f"{anchor}_leiden.json").read_text())
    B = json.loads((OUT / f"{anchor}_bertopic.json").read_text())
    shelf_label = SHELF_LABEL.get(anchor, anchor.replace("_", " "))
    Lthemes, Bthemes = L["themes"], B["themes"]

    fig, ax = cs.slide(
        f"Themes on '{shelf_label}'",
        eyebrow="lens 2 · M2 · per-anchor themes (default config)",
        caption=("Leiden carves more, finer communities; BERTopic/HDBSCAN yields fewer, "
                 "larger topics over the same subtree — both on-topic at this scale."),
    )

    def column(themes, x0, color, head):
        cs.chip(ax, x0, 0.95, head, color=color, fontsize=14)
        if not themes:
            ax.text(x0, 0.80, "— no themes at this shelf —", fontsize=13,
                    color=cs.COLORS["muted"], style="italic", va="top")
            return
        shown = themes[:MAX_ROWS]
        top = 0.82
        # one block = bold label + keyword line; blocks evenly spread over the band.
        step = 0.80 / MAX_ROWS  # fixed pitch (0.133) regardless of how many shown
        for i, t in enumerate(shown):
            y = top - i * step
            kw = ", ".join(t["keyword_terms"][:5]) or "—"
            ax.text(x0, y, cs.excerpt(t["label"], 34), fontsize=13,
                    weight="bold", color=cs.COLORS["ink"], va="top")
            ax.text(x0, y - 0.045, kw, fontsize=10.5,
                    color=cs.COLORS["muted"], va="top")
            cs.chip(ax, x0 + 0.40, y - 0.012, f"{t['chunk_count']} chunks",
                    color=cs.TINTS.get("teal", cs.COLORS["panel"]),
                    fg=cs.COLORS["ink"], fontsize=10.5, weight="normal", pad=0.3)

    column(Lthemes, 0.0, cs.COLORS["teal"], f"Leiden · {len(Lthemes)} themes")
    column(Bthemes, 0.52, cs.COLORS["purple"], f"BERTopic · {len(Bthemes)} themes")
    cs.save_slide(fig, OUT / f"{anchor}_sidebyside.png")
    print("wrote", OUT / f"{anchor}_sidebyside.png")


if __name__ == "__main__":
    keys = sys.argv[1:] or ["olive_oil", "legume", "fish", "dietary_fibre"]
    for k in keys:
        sidebyside(k)
