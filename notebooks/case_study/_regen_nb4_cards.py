"""Regenerate NB4 card figures from the saved JSON artifacts.

The original card figure showed the card title only as the slide heading and
printed each citation as a raw `source_doc` (a bare DOI URL or a long PDF
filename) with no source *type* — so a reader couldn't see at a glance whether a
card is abstract-driven (DOIs) or textbook-driven (PDFs), which is the whole
point of the abstract-expanded anchors.

This re-renders from {anchor}_card.json + {anchor}_provenance.json (no LLM,
no kernel) and adds:
  - the card title as a bold heading on the card body,
  - a one-line source-mix line under "cited evidence" (N abstracts · M textbooks),
  - a per-citation source-type chip (abstract / textbook / web).

Source type is inferred from source_doc: a doi.org / http URL -> journal
abstract; a *.pdf -> textbook; anything else -> web. (The provenance JSON does
not carry chunk.source_type, and inference here is exact for this corpus.)
"""
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, ".")
import cs_common as cs

cs.init_mpl()
OUT = cs.artifacts_dir("nb4_cards")

BADGE = {"high": cs.COLORS["hierarchy"], "medium": cs.COLORS["amber"],
         "low": cs.COLORS["grey"], "debated": cs.COLORS["purple"],
         "unclear": cs.COLORS["grey"]}

SRC_COLOR = {"abstract": cs.COLORS["teal"], "textbook": cs.COLORS["amber"],
             "web": cs.COLORS["grey"]}


def source_type(doc):
    if not doc:
        return "web"
    d = doc.lower()
    if "doi.org" in d or d.startswith("http"):
        return "abstract"
    if d.endswith(".pdf"):
        return "textbook"
    return "web"


def card_fig(k):
    card = cs.load_json(OUT / f"{k}_card.json")
    prov = cs.load_json(OUT / f"{k}_provenance.json")
    resolved = [c for c in prov["citations"] if c["resolved"]]
    cited = resolved[:3]
    pct = (prov["cited_resolved"] / prov["cited_total"] * 100) if prov["cited_total"] else 0.0

    # source-mix over ALL resolved citations (not just the 3 shown)
    from collections import Counter
    mix = Counter(source_type(c["source_doc"]) for c in resolved)
    mix_str = " · ".join(f"{n} {t}{'s' if n != 1 else ''}" for t, n in mix.most_common())

    fig, ax = cs.slide(
        card["title"],
        eyebrow=f"lens 4 · M4 · {k} card",
        caption=f"model: {card['llm_model']} · provenance {prov['cited_resolved']}/{prov['cited_total']} resolved ({pct:.0f}%)",
    )
    eq = str(card["evidence_quality"])
    cs.chip(ax, 0.84, 0.98, eq.upper(), color=BADGE.get(eq, cs.COLORS["grey"]), fontsize=12)

    y = 0.93
    # title on the card body (not only the slide heading) + source-mix subline
    ax.text(0.0, y, card["title"], fontsize=16, weight="bold",
            color=cs.COLORS["ink"], va="top")
    y -= 0.052
    ax.text(0.0, y, f"sources: {mix_str}", fontsize=11,
            color=cs.COLORS["muted"], va="top", style="italic")
    y -= 0.055

    for line in textwrap.wrap(card["summary"], 100)[:5]:
        ax.text(0.0, y, line, fontsize=12.5, va="top", color=cs.COLORS["ink"]); y -= 0.05
    if card.get("tip"):
        y -= 0.015
        for line in textwrap.wrap("TIP — " + card["tip"].strip(), 96)[:2]:
            ax.text(0.0, y, line, fontsize=12, va="top", color=cs.COLORS["teal"], style="italic"); y -= 0.05
    # confidence + controversy notes (were never drawn in the original figure)
    if card.get("controversy_note"):
        y -= 0.012
        for line in textwrap.wrap("CONTROVERSY — " + card["controversy_note"], 100)[:2]:
            ax.text(0.0, y, line, fontsize=11, va="top", color=cs.COLORS["purple"]); y -= 0.044
    if card.get("confidence_note"):
        y -= 0.012
        for line in textwrap.wrap("CONFIDENCE — " + card["confidence_note"], 100)[:3]:
            ax.text(0.0, y, line, fontsize=11, va="top", color=cs.COLORS["muted"]); y -= 0.044
    y -= 0.02
    ax.text(0.0, y, f"cited evidence ({prov['cited_resolved']}/{prov['cited_total']} resolved · {mix_str})",
            fontsize=12.5, weight="bold", color=cs.COLORS["muted"], va="top"); y -= 0.05
    for c in cited:
        st = source_type(c["source_doc"])
        cs.chip(ax, 0.0, y - 0.012, st, color=SRC_COLOR[st], fontsize=9,
                weight="normal", pad=0.3)
        # excerpt wraps to the right of the type chip; source_doc shown after it
        body = f"[{c['source_doc']}] {c['excerpt']}"
        for j, line in enumerate(textwrap.wrap(body, 98)[:2]):
            ax.text(0.085, y, line, fontsize=9.5, va="top",
                    family="DejaVu Sans Mono", color=cs.COLORS["ink"]); y -= 0.040
        y -= 0.014
    cs.save_slide(fig, OUT / f"{k}_card.png")
    print("wrote", OUT / f"{k}_card.png")


if __name__ == "__main__":
    keys = sys.argv[1:] or ["olive_oil", "legume", "fish", "dietary_fibre"]
    for k in keys:
        card_fig(k)
