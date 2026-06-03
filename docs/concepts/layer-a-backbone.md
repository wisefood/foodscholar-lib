# Layer A — Backbone

Layer A is the **navigation skeleton**: a curated, multi-facet menu of *shelves*
projected from the FoodOn ontology, populated only with the parts of the ontology
that the corpus actually talks about.

## From mentions to shelves

```{mermaid}
flowchart LR
    Ch[Chunk] -->|NER| Me[Mentions]
    Me -->|dense linker| Ids[FoodOn IDs]
    Ids -->|walk ancestors| Sup[Support table]
    Sup -->|project + prune| Sh[Shelves]
    Ch -.attach.-> Sh
```

1. **Link.** Each chunk's mentions are linked to FoodOn IDs by the
   [dense linker](annotation.md). FoodOn is a ~39k-term ontology with a real
   `is-a` hierarchy: `olive oil → vegetable oil → … → food product`.
2. **Collect support.** For every linked ID, walk its FoodOn ancestors and tally how
   many chunks mention each class **directly** vs. via a **descendant** (*lifted*
   support). This is the evidence each candidate shelf carries.
3. **Project & prune.** Build a navigable tree from the supported classes, keeping
   only those with real corpus evidence. ~39k FoodOn terms collapse to a few hundred
   shelves per facet — the ones this corpus justifies.

A chunk can attach to **multiple shelves** (a passage about "salmon poached in olive
oil" attaches to both `salmon` and `olive oil`). This multi-label attachment matters
later — it's why Layer B themes must be tied to an *origin shelf* rather than the
union of their chunks' shelves (see [](layer-b-themes.md)).

## Facets

Shelves are grouped into six **facets**, each a separate projection:

`foods` · `health` · `nutrients` · `dietary_patterns` · `allergies` · `sustainability`

A chunk's links route to the relevant facet(s). In the current corpus the `foods`
facet is by far the richest (a few hundred shelves); the others are sparser.

## The projection method

Layer A's construction method is selected by `config.layer_a.projection`. The
production default is **`"backbone"`** — the *1a+ backbone projection* (the name is from
the method [bake-off](glossary.md): method "1a" plus refinements):

- Start from the facet root's **supported children** (the backbone).
- Expand down the *real* FoodOn tiers, but **collapse single-child [filing
  tiers](glossary.md)** — organizational classes with one child that add depth without
  aiding navigation.
- Place every node under a **single parent**, **cap fan-out** (`backbone_max_children`;
  overflow children fold into the nearest kept ancestor, never dropped silently), and
  **prune empty dead-ends**.

The result is **faithful**: every shelf is a real FoodOn class, the tree's edges are
real `is-a` relations, and original labels/IDs are untouched.

```{mermaid}
flowchart LR
    subgraph raw[Raw FoodOn subtree]
      r0[food product] --> r1[…186 siblings…]
      r0 --> r2[milk or milk based food product]
      r2 --> r3[dairy food product]
      r3 --> r4[mammalian milk product]
    end
    subgraph bb[Backbone projection]
      b0[Foods] --> b1[milk or milk based food product]
      b1 --> b2[mammalian milk product]
    end
    raw -->|collapse filing tiers,<br/>cap fan-out, prune| bb
```

```{admonition} Why not just use FoodOn's tree as-is?
:class: tip
The raw ontology is unbalanced for browsing — a flat ~186-wide `foods` blob in
places, deep filing chains in others, and an "umbrella" class (`food product`) that
absorbs generic mentions. The backbone projection re-cuts it into a navigable shape
*without* leaving FoodOn's ID space.
```

### The fallback prune cascade

`projection="prune"` selects the earlier top-down method, kept as a non-default
alternative. It applies, in order: blacklist → **umbrella rule** (drop inflated
organizational classes whose support is almost entirely *lifted*) → whitelist →
support threshold (`min_support`) → depth cap → single-child collapse. Both methods
are `is-a`-faithful; `backbone` is the validated production choice. A third opt-in
method, bottom-up LLM grouping (`bottom_up_grouping.enabled`), exists for facets that
benefit from it.

```{note}
How the method was chosen — the metrics (coverage, findability, depth, reproducibility)
and the bake-off that compared the candidates — is preserved as research provenance
under the repo's `research/` directory.
```

## Aliasing

FoodOn labels are often jargon (`Citrus sinensis (whole, raw)`). After projection, an
optional **LLM aliasing pass** (`config.layer_a.alias_shelves`, on by default when an
LLM is configured) gives jargon shelves a friendly `display_label`. This is purely
**additive** — it never changes a shelf's `label`, `foodon_id`, or position in the
tree, so the projection stays auditable while the UI stays readable.

## The Shelf record

Each shelf carries its identity, position, and the evidence behind it:

```python
class Shelf(BaseModel):
    shelf_id: ShelfId            # e.g. "foodon:FOODON:03309927"
    label: str                   # FoodOn label
    display_label: str | None    # LLM alias for the UI (additive)
    facet: Facet                 # foods | health | nutrients | ...
    depth: int                   # projection-relative depth (0 = root)
    foodon_id: str | None        # the ontology class this shelf represents
    parent_shelf_id: ShelfId | None
    chunk_count: int             # total attached chunks (incl. descendants)
    support_direct: int          # chunks mentioning this exact ID
    support_lifted: int          # support inherited from descendants
    see_also: list[str]          # IDs collapsed into this shelf
```

`support_direct` vs. `support_lifted` is the key diagnostic: a shelf with high lifted
but low direct support is mostly an organizational umbrella; one with high direct
support is a genuine topic the corpus discusses by name.

### Reading real shelves

Actual `foods` shelves from a build make the diagnostic concrete:

| shelf | chunks | direct | lifted | reading |
|---|---|---|---|---|
| `fruit produce` | 633 | **633** | 0 | genuine topic — named outright |
| `vegetable` | 542 | **542** | 0 | genuine topic |
| `cheese` | 239 | **239** | 0 | genuine topic |
| `meat (raw)` | 478 | 407 | 71 | mostly named, some lifted |
| `mammalian milk product` | 649 | 1 | **648** | category — evidence is all descendants |
| `vertebrate food product` | 2016 | 0 | **2016** | pure umbrella — nobody writes the phrase |
| `Foods` | 3893 | 0 | **3893** | the facet root |

The umbrellas (`Foods`, `vertebrate food product`) carry the most chunks but ~zero direct
support — they exist to *organize*, not to be browsed to. The genuine topics are where a
user actually lands.

## How chunks attach

A chunk's linked FoodOn ids are resolved to **surviving** shelves: a direct hit attaches
to that shelf; otherwise the id is lifted up the is-a tree to the nearest kept ancestor.
Because a chunk usually carries several ids — and each lifts independently — **one chunk
attaches to multiple shelves** (see {term}`lifted attachment`). This multi-attachment is
recorded as `shelf_ids` on the chunk (denormalized to Elasticsearch) and is exactly why
Layer B must tie a theme to its [origin shelf](layer-b-themes.md) rather than the union of
its chunks' shelves.

## Building it

```python
fs.build_layer_a()      # project shelves for every configured facet (+ aliasing)
fs.attach()             # attach chunks to shelves (writes shelf_ids denorm)
fs.graph.shelves(facet="foods")          # browse the result
fs.viz.layer_a_tree("foods").render("tree", output="tree.html")   # see it
```

The **Guides** section covers the end-to-end build pipeline and the `fs.graph` read API.
