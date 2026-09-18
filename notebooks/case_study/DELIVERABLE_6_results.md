# §6 Results — draft prose (6.1.2, 6.1.3)

> Drafted from the verified case-study artifacts under `artifacts/`. Every number
> below is reproducible from the notebooks (NB1–NB4) over the abstract-expanded
> corpus. Figure callouts point at the slide-ready PNGs in `artifacts/`.
>
> **Scope note (agreed):** §6.1.1's small NER/NEL test corpus (≈494 chunks) is
> deliberately tiny and cannot showcase the *graph's* value — a hierarchy over a
> few hundred chunks is degenerate. §6.1.2–6.1.3 therefore demonstrate the
> knowledge graph at scale, on a **34,359-chunk** corpus, where the backbone,
> evidence lifting, thematic structure, and graph-aware retrieval become
> meaningful. The two subsections share the same constructed graph.

---

## 6.1.2 Hierarchical graph construction with FoodOn as backbone

We run the full pipeline over a corpus of **34,359 semantically coherent chunks**
spanning three source types — nutrition **textbooks** (12,194 chunks), public
dietary **guides** (1,150 chunks), and peer-reviewed journal **abstracts** (21,015
chunks; only abstracts whose entities were successfully linked are admitted). Each
chunk is NER/NEL-annotated against FoodOn (and the wider OBO family); **25,188
chunks (73 %) carry at least one FoodOn entity** and thus contribute to the food
backbone, while the remainder remain retrievable as text.

**Why scale matters (and why §6.1.1's small corpus cannot show it).** The value
of a *hierarchical* organisation only emerges once enough evidence populates the
backbone. We make this concrete by building the same graph on two corpora — the
guides+textbooks alone (13,344 chunks) and the full corpus with abstracts added
(34,359 chunks). Adding peer-reviewed abstracts **quadruples** the FoodOn-linked
evidence (6,290 → 25,188 chunks), grows the backbone from 256 to 403 populated
classes, and multiplies the thematic richness of individual classes (the olive-oil
class goes from 1 to 6 themes, the legume class from 3 to 6). On a few-hundred-chunk
corpus most FoodOn classes are empty and the hierarchy is degenerate; at corpus
scale it becomes a richly-populated, browsable structure. *(Figure:
`scale_comparison.png` — 13k vs 34k on evidence, classes and themes;
`scale_comparison.json`.)*

**Backbone shape.** Layer A projects the raw FoodOn ontology onto a *navigable*
backbone for the `foods` facet, yielding **403 FoodOn classes (shelves)**
organised into six depth tiers (the full graph holds 408 shelves: these 403 plus
one stub root for each of the five other, unpopulated facets). The projection is
the central construction step:
raw FoodOn is a deep, sparsely-branching DAG that is unusable for browsing,
whereas the projected backbone is shallow and balanced. For the food *olive oil*
(`FOODON:03301826`), the raw ontology buries the class **16 ancestors deep**
through a chain of abstract intermediates (`food material component → … →
extract, concentrate or isolate of plant or animal → olive oil`); the projected
backbone collapses this to **4 navigable tiers** (`Foods → plant food product →
plant lipid food product → plant fat or oil refined food product → olive oil`).
The legume class (`FOODON:00001264`) is similarly reduced from depth 7 to depth 3.
*(Figure: `m1_olive_oil.png`, `m1_legume.png` — raw-vs-projected depth and
ancestor chains.)*

**Chunks per FoodOn class.** Evidence is highly concentrated at the top of the
backbone and thins toward the leaves, exactly as expected of a real corpus. The
synthetic `Foods` root accumulates **13,173 chunks**; first-tier classes such as
`plant food product` (8,305) and `animal food product` (6,397) carry several
thousand each; the **median populated class holds 86 chunks**. The full depth
distribution (number of classes per tier) is 1 / 9 / 34 / 69 / 134 / 156 for
depths 0–5 — i.e. the backbone broadens steadily with depth rather than exploding
at any single level. *(Figure: `depth_hist.png` — classes per depth;
`stats.json`/`stats.csv` — per-facet counts.)*

**Direct vs lifted support — why the hierarchy adds value.** A chunk is attached
to the most specific FoodOn class it mentions, and that evidence is then *lifted*
up the backbone so every ancestor class accumulates the evidence of its whole
subtree. This is what makes intermediate classes useful entry points. The legume
class illustrates it sharply: only **19 chunks mention "legume" directly**, but
**1,436 are lifted** from its descendants (beans, lentils, soybeans, …), giving
the class **1,455 chunks** of browsable evidence it would otherwise lack. The same
mechanism lets the `Foods` root see all 13,173 food-linked chunks. *(Figure:
`support_legume.png`, `support_olive_oil.png` — direct vs lifted support along
each backbone path; `olive_oil_breadcrumb.png`, `legume_breadcrumb.png` — the
root→class navigation path.)*

**Thematic sub-structure (Layer B).** Within a class, Layer B discovers themes
that partition its evidence into coherent sub-topics. Over the abstract-expanded
corpus, the *olive oil* class resolves into **6 Leiden themes** (e.g. *oil/fat/
olive*, *adherence/dietary/diet*, *postprandial/oil/EVOO*) covering 88 % of its
chunks, and the *legume* class into **6 themes** (e.g. *beans/cooked/cup*,
*foods/protein/food*, *postprandial/flour/protein*). The scientific abstracts
visibly enrich these themes beyond what the guides alone supported. We additionally
benchmark two clustering methods (Leiden community detection vs BERTopic/HDBSCAN)
over the *identical* chunk set: at the default granularity both now recover
comparable topics at the food-specific class (keyword overlap up to Jaccard 0.33,
rising further when the HDBSCAN floor is matched), confirming the themes are a
property of the data rather than the algorithm. *(Figure: `comparable_AB.png`;
`olive_oil_sidebyside.png`, `legume_sidebyside.png`.)*

**Complementary retrieval (graph value, made concrete).** The constructed graph is
not just for browsing — it changes *retrieval*. For the query *"How do legumes and
dietary fibre affect blood sugar and cholesterol?"*, a flat hybrid baseline
(BM25 + dense kNN) and a graph-expanded retriever (flat hits enriched with
theme-sibling and class-attached chunks) return measurably different evidence:
**7 of the top 10 graph-expanded results are not in the flat top-10**, and those
graph-only hits are dominated by primary journal abstracts (e.g. *"Diets high in
pulses and legumes have been associated with improved cardiometabolic…"*) that the
flat search ranked below the fold. The graph thus surfaces complementary,
higher-evidence sources by exploiting the class/theme structure. *(Figure:
`legume_compare.png`, `olive_oil_compare.png` — flat vs graph-expanded ranked
lists with each graph-only hit annotated by reason; `*_delta.json` — machine-
readable deltas.)*

---

## 6.1.3 Construction of summaries

For selected classes we generate **evidence cards** (Layer C): a two-stage
pipeline first produces an extractive summary over the class's most
query-relevant theme (LexRank; map-reduce for larger themes), then an LLM
(`llama-3.3-70b-versatile`) rewrites it into a concise, grounded card carrying a
title, summary, practical tip, an evidence-quality rating, and explicit citations
back to source chunks.

We generate three cards, one per anchor class, each from the class's most
query-relevant theme. They span **two evidence regimes** — one card drawn from the
textbook/guide literature and two from journal abstracts — demonstrating that the
same pipeline produces faithful, fully-traceable summaries from didactic text and
from primary scientific evidence alike:

- **Olive oil** — *"Healthy Oils and Fats"* (evidence: medium; **textbook/guide-
  sourced** — 22 textbook + 5 guide of 27 cited chunks). The card explains that the
  monounsaturated and polyunsaturated fats in olive (and other vegetable) oils are
  "good" fats that lower harmful cholesterol and benefit heart health. *(Figure:
  `olive_oil_card.png`.)*

- **Legumes** — *"Dietary Intake of Pulses"* (evidence: medium; **100 % abstract-
  sourced**, 43/43). The card links pulse consumption to improved cardiovascular
  risk markers and higher micronutrient (e.g. folate) intake. *(Figure:
  `legume_card.png`.)*

- **Fish** — *"Fish DHA and Health"* (evidence: medium; **100 % abstract-sourced**,
  92/92). This card is the clearest demonstration of the abstract corpus driving a
  summary end-to-end: the class `fish food product` accumulates 789 chunks (only 2
  attached directly; 787 lifted from species-level descendants), Layer B isolates
  the omega-3 theme (`fish · dha · fish oil · pufas`) from among 15 themes, and
  Layer C distils it into a card on dietary DHA/EPA raising the blood Omega-3 Index.
  *(Figure: `fish_card.png`.)*

**Faithfulness and provenance.** Every claim on a card is traceable: each card
cites the specific chunks it was built from, and we resolve those citations back to
their source documents, reporting the resolution rate. All three cards achieve
**100 % citation resolution** (olive oil 27/27, legumes 43/43, fish 92/92), with
**zero unresolved citations** — the abstract-sourced cards are exactly as traceable
as the textbook-sourced one. *(Data: `*_provenance.json`.)*

**Extractive-vs-LLM refinement.** Placing the Stage-1 extractive extract beside the
final LLM card shows the refinement step condensing without drifting from the
evidence — the extracts (≈0.9–1.7 k chars; `single` for the smallest theme,
`map-reduce` for larger ones) are distilled to ≈440–590-char cards that preserve
the source claims. *(Data: `*_extract_vs_card.json`.)*

> **Reproduction note.** The three cards (`{olive_oil,legume,fish}_card.*`) are
> generated by `NB4_cards.ipynb`, which makes one live LLM call per anchor. It
> requires `GROQ_API_KEY` in the environment and aborts loudly on the mock LLM
> (never fabricates a card). All three cards are generated.

---

### Figure inventory for §6.1.2–6.1.3

| Claim | Figure | Data |
|-------|--------|------|
| **Graph value grows with scale (13k→34k)** | `scale_comparison.png` | `scale_comparison.json` |
| Raw→projected backbone (depth 16→4) | `m1_olive_oil.png`, `m1_legume.png`, `m1_fish.png` | `m1_raw_vs_projected.json` |
| Navigation path to a class | `{olive_oil,legume,fish}_breadcrumb.png` | `anchor_shelves.json` |
| Classes per depth tier | `depth_hist.png` | `stats.json` |
| Direct vs lifted support | `support_{legume,olive_oil,fish}.png` | `stats.json`, `anchor_shelves.json` |
| Themes per class | `{olive_oil,legume,fish}_sidebyside.png` | `*_leiden.json` |
| Clustering methods comparable | `comparable_AB.png` | `intrinsic_same_chunkset.json`, `bertopic_retuned_comparison.json` |
| Flat vs graph retrieval | `{legume,olive_oil,fish}_compare.png` | `*_delta.json` |
| Evidence cards (3: 1 textbook-sourced + 2 abstract-sourced, all 100% provenance) | `{olive_oil,legume,fish}_card.png` | `*_card.json`, `*_provenance.json` |
