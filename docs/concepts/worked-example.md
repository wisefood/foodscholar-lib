# A worked example, end to end

This page follows **one chunk** all the way through the pipeline, using real values from
a `foods` build, so every concept on the other pages has something concrete to point at.

Our chunk:

> **`tb_0421`** *(textbook)* — "One cup of milk provides about 300 mg of calcium, and the
> lactose it contains aids calcium absorption. Fortified milk is also a source of
> vitamin D."

## 1. Annotation — mentions → FoodOn ids

GLiNER finds the mention spans; the dense linker resolves each to a FoodOn id by
embedding it (BioLORD) and taking its nearest ontology term (cosine ≥ `nel_min_sim`):

| mention | → FoodOn id | label |
|---|---|---|
| `milk` | `FOODON:03310029` | cow milk |
| `calcium` | `CHEBI:…` *(nutrient)* | calcium |
| `lactose` | … | lactose |

The chunk is also embedded once with BGE-base (768-d) for retrieval and Layer B Pass 1.
See [Annotation](annotation.md).

## 2. Layer A — which shelves it lands on

The linked id `cow milk` is walked **up** the real FoodOn is-a chain. Each class on the
way gains *lifted* support from this chunk, and the chunk **attaches** to the surviving
shelves on that path:

```{mermaid}
flowchart TD
    A[cow milk] --> B[mammalian milk product<br/>FOODON:03315150]
    B --> C[milk or milk based food product<br/>FOODON:00001257]
    C --> D[dairy food product]
    D --> E[vertebrate food product]
    E --> F[animal food product]
    F --> G[Foods]
```

So `tb_0421` attaches to **`mammalian milk product`** (and its ancestors) — it is one of
that shelf's 649 chunks. That shelf's support tells the story: **direct 1, lifted 648** —
almost nobody writes the literal phrase "mammalian milk product", but the corpus is full
of its descendants (`cow milk`, `goat milk`, …). Contrast a *genuine topic* like
`fruit produce` (**633 direct, 0 lifted**), which the corpus names outright. See
[Layer A](layer-a-backbone.md) and `support_direct` vs `support_lifted` in the
[glossary](glossary.md).

## 3. Layer B — which theme it joins

Within the `mammalian milk product` shelf, Layer B's two passes cluster the chunks. Our
chunk — calcium, lactose, absorption — lands in the **relatedness** theme below (it
shares the entities `cow milk`, `calcium`, `lactose` with its theme-mates). Real themes
on this shelf, by origin:

| `discovery_pass` | theme label | chunks | top keywords |
|---|---|---|---|
| `relatedness` | milk calcium lactose | 167 | milk, calcium, lactose, vitamin, fat |
| `relatedness` | milk protein foods | 103 | milk, protein, foods, eggs |
| `global_similarity` | calcium vitamin milk | 123 | calcium, vitamin, milk, protein |
| `global_similarity` | breast milk breast infant | 98 | breast milk, breast, infant, formula |

`tb_0421` joins **"milk calcium lactose"**. (This build produced no `merged` themes — a
real, instructive outcome: it means the embedding and entity passes didn't overlap enough
to cross the merge threshold, which is a [tuning](../guides/tuning-layer-b.md) signal, not
a bug.) See [Layer B](layer-b-themes.md).

## 4. Layer C — the card that cites it

`build_layer_c` writes a cited card for the shelf/theme. Every sentence must be grounded
in member chunks — including ours:

> **Calcium in dairy milk** — *evidence quality: high*
> Milk is a major dietary calcium source, ~300 mg per cup `[tb_0421]`, and its lactose
> content supports calcium absorption `[tb_0421, ab_088]`. Fortified milk additionally
> supplies vitamin D `[tb_0421]`.
> *cited_chunk_ids:* `[tb_0421, ab_088, gd_0203]`

See [Layer C](layer-c-cards.md).

## 5. Retrieval — answering a query

Now a user asks *"Is dairy a good source of calcium?"*:

1. **Hybrid search (Elasticsearch):** BM25 + kNN over the query, fused by RRF, filtered to
   `shelf_ids ∋ mammalian milk product`. `tb_0421` ranks high.
2. **Theme expansion (Neo4j → ES):** follow `tb_0421`'s `theme_ids` to the
   *milk calcium lactose* theme and pull its sibling chunks — adding complementary
   evidence (absorption studies, fortification) that worded things differently.
3. **Present:** the chunks, plus the shelf's Layer C card, with full provenance
   `chunk → shelf → theme → source doc`.

That provenance trail is the whole point: the answer can cite exactly where every claim
came from. See [Architecture](architecture.md) for the two-store machinery underneath.
```{note}
Steps 1–3 use the values shown above verbatim from a real `foods` build; the chunk text
and the card are representative (Layer C wasn't built in this snapshot), but the shelf,
its support numbers, the is-a chain, and the themes are exactly as the graph contains them.
```
