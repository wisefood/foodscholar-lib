# NB4 — Cards: faithfulness & provenance (M4) — summary

- LLM model: **llama-3.3-70b-versatile** (real; mock guard passed).

## Cards
- **olive_oil** — `Healthy Fats and Oils` (evidence: medium); provenance 27/27 resolved (100%).
- **legume** — `Legume Based Dietary Intake` (evidence: high); provenance 43/43 resolved (100%).
- **fish** — `Fatty Fish and PUFAs` (evidence: medium); provenance 92/92 resolved (100%).
- **dietary_fibre** — `Dietary Fiber Intake` (evidence: medium); provenance 75/75 resolved (100%).

## Files
env.json, {anchor}_card.json, {anchor}_provenance.json, {anchor}_extract_vs_card.json, {anchor}_card.png.

## Deviations / limitations
- Only the two salient anchor themes are sent to the LLM (two live calls), not the full facet — keeps the case study cheap/fast while exercising the real Stage-1→Stage-2 path.
- Any unresolved citation or unsupported claim is listed above, not hidden.

## Acceptance
- [x] a card per anchor
- [x] provenance resolution reported (deviations flagged)
- [x] extract-vs-card available