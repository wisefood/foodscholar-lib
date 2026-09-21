"""Unit tests for the NEL index module.

`HNSWNELIndex` needs `hnswlib` + `sentence-transformers`, both behind the
`@pytest.mark.slow` integration suite. Here we cover the parts that can be
tested without those deps: signature derivation, encoder name validation,
path resolution, and the elastic stub.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foodscholar.annotate.nel_index import (
    ENCODER_IDS,
    ENCODER_POOLING,
    ElasticNELIndex,
    HNSWNELIndex,
    NELIndex,
    _cache_is_reusable,
    _ontology_signature,
    _term_text,
)
from foodscholar.io.ontology import OntologyTerm
from foodscholar.ontology import FoodOnAPI


def _api(terms: list[OntologyTerm]) -> FoodOnAPI:
    return FoodOnAPI(terms, prefix_filter=None)


def test_encoder_ids_includes_biolord_and_sapbert() -> None:
    assert "biolord" in ENCODER_IDS
    assert "sapbert" in ENCODER_IDS
    assert "minilm" in ENCODER_IDS
    assert "mpnet" in ENCODER_IDS


def test_ontology_signature_is_deterministic() -> None:
    a = _api([OntologyTerm(id="T:1", label="alpha"), OntologyTerm(id="T:2", label="beta")])
    b = _api([OntologyTerm(id="T:2", label="beta"), OntologyTerm(id="T:1", label="alpha")])
    assert _ontology_signature(a) == _ontology_signature(b)


def test_ontology_signature_changes_with_content() -> None:
    a = _api([OntologyTerm(id="T:1", label="alpha")])
    b = _api([OntologyTerm(id="T:1", label="beta")])
    assert _ontology_signature(a) != _ontology_signature(b)


def test_ontology_signature_ignores_obsolete_terms() -> None:
    a = _api([OntologyTerm(id="T:1", label="alpha")])
    b = _api(
        [
            OntologyTerm(id="T:1", label="alpha"),
            OntologyTerm(id="T:DEAD", label="ghost", obsolete=True),
        ]
    )
    assert _ontology_signature(a) == _ontology_signature(b)


def test_term_text_joins_label_and_synonyms() -> None:
    t = OntologyTerm(id="T:1", label="vitamin C", synonyms=("ascorbic acid", "ascorbate"))
    assert _term_text(t) == "vitamin C ; ascorbic acid ; ascorbate"


def test_hnsw_rejects_unknown_encoder() -> None:
    api = _api([OntologyTerm(id="T:1", label="x")])
    with pytest.raises(ValueError, match="unknown NEL encoder"):
        HNSWNELIndex(api, encoder="not-a-real-encoder")  # type: ignore[arg-type]


def test_elastic_nel_stub_raises_until_implemented() -> None:
    with pytest.raises(NotImplementedError, match="ElasticNELIndex is not implemented"):
        ElasticNELIndex(url="http://localhost:9200", index="foodscholar_nel")


def test_protocol_runtime_checkable_with_fake_index() -> None:
    """NELIndex is a runtime_checkable Protocol — duck-typing must work."""

    class _Fake:
        backend_id = "fake"

        def link(self, surface):  # type: ignore[no-untyped-def]
            return None

        def link_batch(self, surfaces):  # type: ignore[no-untyped-def]
            return [None] * len(surfaces)

    assert isinstance(_Fake(), NELIndex)


def test_hnsw_resolve_paths_uses_signature_in_filename(tmp_path: Path) -> None:
    """When paths are not provided, the index name encodes encoder + signature."""
    # We can't construct an HNSWNELIndex without ML deps, but we can call the
    # path resolver via a subclass that skips the build/load step.

    class _NoBuild(HNSWNELIndex):
        def __init__(self, ontology: FoodOnAPI, *, encoder: str, cache_dir: Path) -> None:
            self._encoder_name = encoder
            self._encoder_model_id = ENCODER_IDS[encoder]
            self._signature = _ontology_signature(ontology)
            paths = self._resolve_paths(None, None, cache_dir)
            self.resolved_paths = paths

    api = _api([OntologyTerm(id="T:1", label="alpha")])
    sub = _NoBuild(api, encoder="biolord", cache_dir=tmp_path)
    idx_path, meta_path = sub.resolved_paths
    assert idx_path.parent == tmp_path
    assert "foodon_hnsw_biolord_" in idx_path.name
    assert idx_path.suffix == ".bin"
    assert meta_path.suffix == ".json"


# --------------------------------------------------------------------------
# Encoder pooling. SapBERT is a plain HF checkpoint trained with a CLS
# objective; sentence-transformers would otherwise default it to mean pooling.
# --------------------------------------------------------------------------


def test_sapbert_declares_cls_pooling() -> None:
    assert ENCODER_POOLING["sapbert"] == "cls"


def test_sentence_transformer_encoders_declare_no_override() -> None:
    """BioLORD, MiniLM and MPNet ship their own pooling config — leave them alone."""
    for encoder in ("biolord", "minilm", "mpnet"):
        assert encoder not in ENCODER_POOLING


def _meta(**overrides: object) -> dict:
    base = {
        "encoder": "FremyCompany/BioLORD-2023",
        "pooling": None,
        "signature": "sig-1",
        "terms": [{"uri": "T:1", "label": "alpha"}],
    }
    base.update(overrides)
    return base


def test_cache_is_reusable_when_everything_matches() -> None:
    assert _cache_is_reusable(
        _meta(),
        encoder_model_id="FremyCompany/BioLORD-2023",
        pooling=None,
        signature="sig-1",
    )


def test_cache_rejected_when_pooling_differs() -> None:
    """The regression this guards: a mean-pooled SapBERT index reused for CLS.

    Both poolings are 768-dim, so the dim check downstream cannot catch it and
    the linker would return quietly wrong links forever.
    """
    stale = _meta(encoder="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    assert stale["pooling"] is None  # written before ENCODER_POOLING existed

    assert not _cache_is_reusable(
        stale,
        encoder_model_id="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        pooling="cls",
        signature="sig-1",
    )


def test_cache_without_pooling_key_still_matches_encoders_with_no_override() -> None:
    """An old BioLORD index must not be rebuilt for nothing."""
    legacy = _meta()
    del legacy["pooling"]

    assert _cache_is_reusable(
        legacy,
        encoder_model_id="FremyCompany/BioLORD-2023",
        pooling=None,
        signature="sig-1",
    )


def test_cache_rejected_on_signature_or_encoder_change() -> None:
    assert not _cache_is_reusable(
        _meta(),
        encoder_model_id="FremyCompany/BioLORD-2023",
        pooling=None,
        signature="sig-2",
    )
    assert not _cache_is_reusable(
        _meta(),
        encoder_model_id="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        pooling="cls",
        signature="sig-1",
    )


def test_cache_rejected_when_terms_missing() -> None:
    assert not _cache_is_reusable(
        _meta(terms="not-a-list"),
        encoder_model_id="FremyCompany/BioLORD-2023",
        pooling=None,
        signature="sig-1",
    )
