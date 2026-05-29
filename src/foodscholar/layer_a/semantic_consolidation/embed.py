"""Embedding step: turn each shelf into a vector for similarity search.

The real `Shelf` has no text beyond `label`, so the embedding signal is
reconstructed from the ontology: `label` plus the FoodOn synonyms keyed by the
shelf's `foodon_id`. (FoodOn `OntologyTerm` carries no definition, so the
brief's `definition` component is omitted.) Shelves with no `foodon_id` — the
synthetic facet roots — are excluded: nothing should ever merge into them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.layer_a.semantic_consolidation.models import ShelfEmbedding

if TYPE_CHECKING:
    from foodscholar.config import SemanticConsolidationConfig
    from foodscholar.io.graph import Shelf
    from foodscholar.ontology import FoodOnAPI
    from foodscholar.storage.protocols import Embedder


def shelf_embed_text(
    shelf: Shelf, ontology: FoodOnAPI, cfg: SemanticConsolidationConfig
) -> str:
    """Build the text fed to the embedder: ``label | syn1 | syn2 ...``."""
    parts = [shelf.label]
    if shelf.foodon_id:
        syns = ontology.id_to_synonyms(
            shelf.foodon_id, include_related=cfg.include_related_synonyms
        )
        parts.extend(syns[: cfg.max_synonyms])
    return " | ".join(parts)


def is_scaffolding(
    shelf: Shelf, ontology: FoodOnAPI, cfg: SemanticConsolidationConfig
) -> bool:
    """True if the shelf is a FoodOn organizational class, not a food.

    Heuristic: it has NO exact synonyms AND its label's last word is a generic
    classifier (`product`, `process`, `group`, `supplement`, …). Real foods
    nearly always carry a synonym (`olive oil` → 'EVOO'), so requiring BOTH
    conditions keeps false positives low. Synthetic roots (no `foodon_id`)
    aren't this function's concern — they're filtered earlier.
    """
    if not shelf.foodon_id:
        return False
    if ontology.id_to_synonyms(shelf.foodon_id, include_related=True):
        return False
    last_word = shelf.label.lower().split()[-1] if shelf.label.split() else ""
    return last_word in {s.lower() for s in cfg.classifier_suffixes}


def embed_shelves(
    shelves: list[Shelf],
    ontology: FoodOnAPI,
    embedder: Embedder,
    cfg: SemanticConsolidationConfig,
) -> list[ShelfEmbedding]:
    """Embed every shelf that carries a `foodon_id` and isn't scaffolding.

    Synthetic facet roots (no `foodon_id`) are skipped — nothing should merge
    into them. When `cfg.exclude_scaffolding` is set, FoodOn organizational
    umbrella terms are skipped too (see `is_scaffolding`): they cluster at high
    cosine but never merge, so embedding them only produces noise.

    The returned vectors are whatever the embedder produces; the candidate
    step normalizes before comparison, so unnormalized embedders are fine.
    """
    eligible = [
        s
        for s in shelves
        if s.foodon_id is not None
        and not (cfg.exclude_scaffolding and is_scaffolding(s, ontology, cfg))
    ]
    if not eligible:
        return []
    texts = [shelf_embed_text(s, ontology, cfg) for s in eligible]
    vectors = embedder.embed(texts)
    return [
        ShelfEmbedding(
            shelf_id=s.shelf_id,
            foodon_id=s.foodon_id,  # type: ignore[arg-type]  # filtered non-None above
            text=t,
            embedding=list(v),
            embedder_id=embedder.model_id,
        )
        for s, t, v in zip(eligible, texts, vectors, strict=True)
    ]
